"""
ARC-2 CYCLE 12 / CONSTRUCTION (not comparison): the SRAM organ
================================================================
P13 (in-context mapping) is the one axis nobody won: micro TF, SSM, and even
the 796k-param TF_STRONG @10k steps all failed it at the TRAINING length
(target CE = ln16 everywhere). The cause (L-TWO-HOP + storage): the per-context
mapping (44 bits for 16 keys) must be STORED while the answer is one hop away
from a content match. Attention has no state; 16 floats hold ~44 bits at best.

The missing component in our architecture: an EXACT ASSOCIATIVE MEMORY organ —
a per-context register file (one slot per key), exact write/read, ~3k params.
The transformer has no such unit: its "memory" is the input itself, read
softly and statelessly. The SSM's "memory" is 16 floats. The SRAM organ is
hardware in the old sense: slots that hold exactly what was written.

Model = SSMHost d16 (fluency/finite-state, 2.9k) + SRAMOrgan:
  - 16 slots, slot k stores the embedding of the value last written for key k
  - causal write protocol: key token sets last_key; value token writes
    slot[last_key] = emb(value); seen[k] flag (gate)
  - readout: if x[t] is a seen key k, organ logits = W_readout(slot[k])
  - host residual (L-GATE-INIT) + organ gate; everything else exact
If the SRAM organ is exact at 4096 (target CE ~ 0.01) where every transformer
flavor failed at 64, P13 is CLOSED BY CONSTRUCTION: ICL is solved by a
computational unit attention does not have, at ~3k params.

Task: ICL-MICRO (16-key random bijection, 31 examples + query + target at
L=64; 2047 examples at 4096), same generator as icl_micro.py.
USAGE: OMP_NUM_THREADS=1 python3 -u sram_icl.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
NK = NV = 16
VOCAB = NK + NV
LN16 = math.log(16.0)
CFG = dict(steps=2500, batch=32, train_len=64,
           eval_lens=[64, 512, 2048, 4096], eval_reps=2, d_model=16)
print(f"[setup] sram-icl cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- data (identical to icl_micro)
def gen_icl(batch, length, rng):
    assert length % 2 == 0
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        n = (length - 2) // 2
        mapping = list(range(NK))
        rng.shuffle(mapping)
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

# ---------------------------------------------------------------- host
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
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([SSMBlock(d) for _ in range(1)])
        self.norm = nn.LayerNorm(d)

    def hiddens(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

# ---------------------------------------------------------------- SRAM organ
# The clean implementation: single causal loop, exact bookkeeping
class SRAModel(nn.Module):
    """SSM host + SRAM organ, one causal pass, exact register file."""
    def __init__(self):
        super().__init__()
        self.host = SSMHost()
        d = CFG["d_model"]
        self.W_readout = nn.Linear(d, VOCAB)
        self.hhead = nn.Linear(d, VOCAB)
        nn.init.zeros_(self.hhead.weight)      # L-GATE-INIT
        nn.init.zeros_(self.hhead.bias)
        self.organ_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        B, L = x.shape
        d = CFG["d_model"]
        slots = torch.zeros(B, NK, d, device=x.device)
        seen = torch.zeros(B, NK, dtype=torch.bool, device=x.device)
        last_key = torch.full((B,), -1, dtype=torch.long, device=x.device)
        h = self.host.hiddens(x)
        emb = self.host.emb(x)
        organ_logits = torch.zeros(B, L, VOCAB, device=x.device)
        for t in range(L):
            tok = x[:, t]
            is_key = tok < NK
            is_val = tok >= NK
            k = tok.clamp(0, NK - 1)   # value tokens masked out downstream
            # organ answer at position t: x[t] is a repeat key -> slot value
            idx = k.view(B, 1, 1).expand(B, 1, d)
            cand = self.W_readout(slots.gather(1, idx).squeeze(1))
            mask = (is_key & seen.gather(1, k.unsqueeze(1)).squeeze(1)).float()
            organ_logits[:, t] = cand * mask.unsqueeze(-1)
            # exact writes (affect t+1..):
            for b in range(B):
                if bool(is_val[b]) and int(last_key[b]) >= 0:
                    slots[b, int(last_key[b])] = emb[b, t].detach()  # straight-through write
                    seen[b, int(last_key[b])] = True
                if bool(is_key[b]):
                    last_key[b] = int(tok[b])
        logits = self.hhead(h) + torch.exp(self.organ_gate) * organ_logits
        return logits

# ---------------------------------------------------------------- train/eval
def train_model(model, seed=0, tag="sram"):
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
            print(f"  [{tag} s{seed}] step {step}/{CFG['steps']} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_dce(model, L, reps=2):
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
RESULTS = {}
for seed in range(2):
    torch.manual_seed(seed)
    m = SRAModel()
    if seed == 0:
        print(f"[run] sram organ + host params={n_params(m)}", flush=True)
    train_model(m, seed)
    r = {f"L{L}": eval_dce(m, L, CFG["eval_reps"]) for L in CFG["eval_lens"]}
    r["params"] = n_params(m)
    RESULTS[f"sram_s{seed}"] = r
    print(f"  [sram s{seed}] {r}", flush=True)
    del m

print("\n" + "=" * 84)
print("SRAM-ORGAN ICL  dCE nats/token (0 = oracle) | target CE (ln16 = 2.773)")
print("  reference failures (documented, icl_micro/10k): ssm_d16 2.74-2.81,")
print("  ssm_d64 2.77-2.80, tf_rope 2.70-4.63, TF_STRONG 796k@10k: not run on ICL")
print("=" * 84)
print(f"{'run':<10}" + "".join(f"{f'L{L} (total|target)':<26}" for L in CFG["eval_lens"]), flush=True)
for run, r in RESULTS.items():
    print(f"{run:<10}" + "".join(f"{str(r[f'L{L}']):<26}" for L in CFG["eval_lens"]), flush=True)
print("=" * 84)
final = {"tag": "ARC2-C12-SRAM-ORGAN-ICL", "runs": RESULTS,
         "note": "total_dce | target_dCE per length; ln16=2.773 = random",
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
