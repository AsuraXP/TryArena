"""
ARC-2 CYCLE 9 / COUNTER-AXIS - NEEDLE: content-addressed retrieval (the niche
attention is mechanistically built for). If the linear host loses here, the
honest map is: each mechanism owns a niche; our next mutation = a lookup (SRAM)
organ. If it wins/ties, the claim strengthens.

Task: n random (key, value) pairs + 1 query key + target value.
  - 256 key tokens, 256 value tokens; key->value mapping is PER-CONTEXT random
    (re-drawn each sequence) but CONSISTENT within a sequence (a repeated key
    keeps its value), so no learned global table can solve it: the association
    must be read from the CONTEXT (stateless retrieval for attention, 2048-bit
    per-context storage for fixed state).
  - Pairs sampled with replacement; query = uniform draw over DISTINCT keys
    present; target = its value (nll 0: deterministic given context).
  - Train L=64 (n=31 pairs), eval 64/512/2048/4096 (n ~ 2047, every key seen
    ~8x, needle at random depth up to 4000 positions back).

Arms (same budget): tf_rope (predicted WIN - content-matched attention, O(1)
state), ssm_d16_1 (2.8k, the documented host; 16 floats << 2048 bits ->
predicted failure).
Oracle = exact per-token conditional nll (key ln256; value ln256 on first
occurrence, 0 after; query ln(distinct); target 0).
USAGE: OMP_NUM_THREADS=1 python3 -u needle.py
"""
import json, math, os, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
NK = 256                       # key tokens 0..255
NV = 256                       # value tokens 256..511
VOCAB = NK + NV
LNK, LNV, LND = math.log(NK), math.log(NV), None
CFG = dict(steps=2500, batch=32, train_len=64,
           eval_lens=[64, 512, 2048, 4096], eval_reps=2,
           d_model=16, tf_d=64, tf_layers=2)
print(f"[setup] needle cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- data
def gen_needle(batch, length, rng):
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        assert length % 2 == 0
        n = (length - 2) // 2          # n pairs + query + target = length tokens
        x, nll, mapping = [], [], {}
        for i in range(n):
            k = rng.randrange(NK)
            if k not in mapping:
                v = 256 + rng.randrange(NV)
                mapping[k] = v
                nll.append(LNK + LNV)   # key + fresh value
            else:
                v = mapping[k]
                nll.append(LNK + 0.0)   # key + known value
            x.append(k); x.append(v)
        distinct = list(mapping.keys())
        q = rng.choice(distinct)
        x.append(q)
        nll.append(math.log(len(distinct)))
        x.append(mapping[q])           # target: deterministic given context
        nll.append(0.0)
        xs.append(x[:-1]); ys.append(x[1:])
        os_.append(nll[1:])
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
    def __init__(self, d, nblk):
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
        h = self.emb(x) + self.sinusoidal(L, self.emb.weight.shape[1]).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(DEVICE)
        return self.head(self.enc(h, mask))

def train_model(model, seed=0, tag=""):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        x, y, o = gen_needle(CFG["batch"], CFG["train_len"], rng)
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
        x, y, o = gen_needle(bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return round((ce - orc) / n, 4), round(tgt_ce / tgt_n, 4)

# ---------------------------------------------------------------- experiment
ALL = {}
torch.manual_seed(0)
ssm = SSMHost(CFG["d_model"], 1)
print(f"[run] ssm_d16_1 params={n_params(ssm)}", flush=True)
train_model(ssm, 0, "ssm")
torch.save(ssm.state_dict(), "needle_ssm_s0.pt")
ALL["ssm_d16_1"] = {f"L{L}": eval_dce(ssm, L, CFG["eval_reps"]) for L in CFG["eval_lens"]}
del ssm

torch.manual_seed(0)
tf = TransformerLM(CFG["tf_d"], CFG["tf_layers"])
print(f"[run] tf_rope params={n_params(tf)}", flush=True)
train_model(tf, 0, "tf")
torch.save(tf.state_dict(), "needle_tf_s0.pt")
ALL["tf_rope"] = {f"L{L}": eval_dce(tf, L, CFG["eval_reps"]) for L in CFG["eval_lens"]}

print("\n" + "=" * 84)
print("NEEDLE  dCE nats/token (0 = oracle) | target-dCE = CE on the FINAL value token")
print("=" * 84)
print(f"{'run':<14}" + "".join(f"{f'L{L} (total|target)':<26}" for L in CFG["eval_lens"]), flush=True)
for run, r in ALL.items():
    print(f"{run:<14}" + "".join(f"{str(r[f'L{L}']):<26}" for L in CFG["eval_lens"]), flush=True)
print("=" * 84)
final = {"tag": "ARC2-C9-NEEDLE", "runs": ALL,
         "note": "total_dce | target_dCE per length",
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
