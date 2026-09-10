"""
ARC-2 CYCLE 9 / COUNTER-AXIS #2: ICL-MICRO — few-shot in-context mapping
(the canonical transformer strength: induction heads learn a per-context
random mapping from examples in the context).

Task: per-context random BIJECTION key->value (16 keys, 16 values).
  n example pairs (k_i, mapping[k_i]) + test key q + target mapping[q].
  Mapping entropy = log2(16!) ~ 44 bits -> storage boundary test:
    tf_rope   (stateless attention; induction-heads regime)  predicted EXACT
    ssm_d16_1 (16 floats ~ 44 bits, borderline)              predicted FAIL/degrade
    ssm_d64_1 (64 floats ~ enough)                           capacity control
  Train L=64 (31 examples), eval 64/512/2048/4096 (2047 examples).
Oracle: key ln16; value ln(16-assigned) on a key's first occurrence, 0 after;
query ln16; target 0.
USAGE: OMP_NUM_THREADS=1 python3 -u icl_micro.py
"""
import json, math, os, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
NK = NV = 4
VOCAB = NK + NV
LN16 = math.log(4.0)
CFG = dict(steps=2500, batch=32, train_len=64,
           eval_lens=[64, 512, 2048, 4096], eval_reps=2)
print(f"[setup] icl-micro cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- data
def gen_icl(batch, length, rng):
    assert length % 2 == 0
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        n = (length - 2) // 2
        mapping = list(range(NK))
        rng.shuffle(mapping)                 # random bijection 0..15 -> 0..15
        x, nll, seen = [], [], set()
        for i in range(n):
            k = rng.randrange(NK)
            v = NK + mapping[k]
            if k not in seen:
                nll.append(LN16 + math.log(NV - len(seen)))
                seen.add(k)
            else:
                nll.append(LN16)
            x.append(k); x.append(v)
        q = rng.randrange(NK)
        x.append(q); nll.append(LN16)
        x.append(NK + mapping[q]); nll.append(0.0)
        xs.append(x[:-1]); ys.append(x[1:]); os_.append(nll[1:])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

# ---------------------------------------------------------------- models
def n_params(m):
    return sum(p.numel() for p in m.parameters())

class SSMBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wx = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.log_a = nn.Parameter(-3.0 * torch.ones(d))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        x = self.Wx(self.n1(h))
        a = 1.0 - torch.exp(-F.softplus(self.log_a))
        B, L, d = x.shape
        S = torch.zeros(B, d, device=x.device)
        outs = []
        for t in range(L):
            S = a * S + x[:, t]
            outs.append(self.Wo(S))
        o = torch.stack(outs, 1)
        h = h + o
        return h + self.mlp(self.n2(h))

class SSMHost(nn.Module):
    def __init__(self, d, nblk=1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([SSMBlock(d) for _ in range(nblk)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

class TransformerLM(nn.Module):
    def __init__(self, d_model, n_layers, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB)
        self.head.weight = self.emb.weight

    @staticmethod
    def sinusoidal(L, d):
        p = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        i = torch.arange(0, d, 2).float()
        p[:, 0::2] = torch.sin(pos * torch.exp(-9 * i / d))
        p[:, 1::2] = torch.cos(pos * torch.exp(-9 * i / d))
        return p

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) + self.sinusoidal(L, self.d_model).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(DEVICE)
        return self.head(self.enc(h, mask))

def train_model(model, seed=0, tag=""):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        x, y, o = gen_icl(CFG["batch"], CFG["train_len"], rng)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 500 == 0:
            print(f"  [{tag}] step {step}/{CFG['steps']} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_dce(model, L, reps):
    model.eval()
    bs = max(1, min(8, 4096 // L))
    ce = orc = n = 0.0
    tgt_ce = tgt_n = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gen_icl(bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return round((ce - orc) / n, 4), round(tgt_ce / tgt_n, 4)

# ---------------------------------------------------------------- experiment
ALL = {}
for name, mk in [("ssm_d16_1", lambda: SSMHost(16, 1)),
                 ("ssm_d64_1", lambda: SSMHost(64, 1)),
                 ("tf_rope", lambda: TransformerLM(64, 2))]:
    torch.manual_seed(0)
    m = mk()
    print(f"[run] {name} params={n_params(m)}", flush=True)
    train_model(m, 0, name)
    torch.save(m.state_dict(), f"icl4_{name}_s0.pt")
    ALL[name] = {f"L{L}": eval_dce(m, L, CFG["eval_reps"]) for L in CFG["eval_lens"]}
    del m

print("\n" + "=" * 84)
print("ICL-4X4  dCE nats/token (0 = oracle) | target-dCE = CE on the FINAL value")
print("=" * 84)
print(f"{'run':<14}" + "".join(f"{f'L{L} (total|target)':<26}" for L in CFG["eval_lens"]), flush=True)
for run, r in ALL.items():
    print(f"{run:<14}" + "".join(f"{str(r[f'L{L}']):<26}" for L in CFG["eval_lens"]), flush=True)
print("=" * 84)
final = {"tag": "ARC2-C9-ICL-4X4", "runs": ALL,
         "note": "total_dce | target_dCE per length",
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
