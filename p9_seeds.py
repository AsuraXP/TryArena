"""
ARC-2 CYCLE 10 / P9 EXTENSION: crystallization lottery of the END-TO-END line
=============================================================================
P9 (crystallization lottery ~50%/seed) was CLOSED in Cycle 4 for DIRECT-
SUPERVISED table classes (L-DIRECT-GRADIENT, zero restarts). The Cycle-8/9
architecture adds END-TO-END CE-trained channels (linear SSM host, echo-organ
readout, MoA router) - indirectly supervised, the class where ssr_lab saw the
lottery. This sweep asks: does the new architecture have a hidden lottery?

Arms (all: 2500 steps, batch 32, L=64, AdamW 3e-3, clip 1.0, OMP=1):
  A) ssm_d16_1 (3.2k) on hard tasks (dyck10 + mod7, VOCAB 27) - 5 seeds
     pass = mod7@4096 dCE <= 0.05 AND dyck10@4096 dCE <= 0.30
  B) echo-organ (3.0k, exact K-stack) on Dyck-echo - 4 seeds
     pass = echo-dCE@4096 <= 0.10 (true-oracle echo metric)
  C) tf_rope (101k) on hard tasks - 3 seeds (baseline lottery comparison)
Theory context (validated, arxiv 2508.07395): non-negative input-dependent
SSMs provably cannot solve parity/modular counting in finite precision as a
BARE SSM layer - our d16 host has learned positive per-channel decay + MLP
residual + tied head, and empirically nails mod-7 (L-LINEAR-HOST). The seed
sweep tests whether that empirical win is seed-robust.
USAGE: OMP_NUM_THREADS=1 python3 -u p9_seeds.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
t_start = time.time()

# ---------------------------------------------------------------- reuse defs
strip_src = open("ssm_strip.py").read()
strip_defs = strip_src.split("# ---------------------------------------------------------------- experiment")[0]
gS = {"__name__": "p9s"}
exec(strip_defs, gS)

echo_src = open("dyck_echo.py").read()
echo_defs = echo_src.split("# ---------------------------------------------------------------- experiment")[0]
gE = {"__name__": "p9e"}
exec(echo_defs, gE)

VOCAB27 = gS["VOCAB"]
SSM16_1 = None
# rebuild the d16/1 variant (it's defined inside make_variants in ssm_strip)
SSMBlock = gS["SSMBlock"]

class S16_1(nn.Module):
    def __init__(self):
        super().__init__()
        d = 16
        self.emb = nn.Embedding(VOCAB27, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([SSMBlock(d) for _ in range(1)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB27)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

class TF27(nn.Module):
    """RoPE-free sinusoidal 2-layer TF on the 27-vocab hard tasks."""
    def __init__(self, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB27, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB27)
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

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- training
def train_hard(model, seed, tag):
    """same mixed hard-task recipe as ssm_strip.train"""
    CFG = gS["CFG"]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        xd, yd, _ = gS["gen_dyck10"](CFG["batch"] // 2, CFG["train_len"], rng)
        xm, ym, _ = gS["gen_mod7"](CFG["batch"] // 2, CFG["train_len"], rng)
        x, y = torch.cat([xm, xd]), torch.cat([ym, yd])
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB27), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 1000 == 0:
            print(f"  [{tag}] step {step}/{CFG['steps']} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_hard(model, task, L=4096, reps=2):
    return gS["eval_dce"](model, task, L, reps)

def train_echo(model, seed):
    CFG = gE["CFG"]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        x, y, o = gE["gen_echo"](CFG["batch"], CFG["train_len"], rng)
        loss = F.cross_entropy(model(x).reshape(-1, gE["VOCAB"]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 1000 == 0:
            print(f"  [echo s{seed}] step {step}/{CFG['steps']} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_echo_type(model, L=4096, reps=2):
    """echo-token dCE (true oracle = 0) on Dyck-echo"""
    model.eval()
    bs = max(1, min(8, 4096 // L))
    ce = n = 0
    per = {t: [0.0, 0] for t in range(gE["VOCAB"])}
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gE["gen_echo"](bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        for t in range(gE["VOCAB"]):
            m = (y == t)
            per[t][0] += nll[m].sum().item(); per[t][1] += int(m.sum())
        ce += nll.sum().item(); n += y.numel()
    echo_n = per[3][1] + per[4][1] + per[5][1]
    return (per[3][0] + per[4][0] + per[5][0]) / max(1, echo_n)

# ---------------------------------------------------------------- experiment
RESULTS = {}

print("=" * 78, flush=True)
print("ARM A: ssm_d16_1 (hard tasks) x 5 seeds  [pass: mod7<=0.05, dyck10<=0.30 @4096]", flush=True)
for seed in range(5):
    torch.manual_seed(seed)
    m = S16_1()
    if seed == 0:
        print(f"[arm A] params={n_params(m)}", flush=True)
    train_hard(m, seed, f"ssm s{seed}")
    r = {"mod7@4096": round(eval_hard(m, "mod7"), 4),
         "dyck10@4096": round(eval_hard(m, "dyck10"), 4)}
    r["pass"] = r["mod7@4096"] <= 0.05 and r["dyck10@4096"] <= 0.30
    RESULTS[f"A_ssm161_s{seed}"] = r
    print(f"  [A s{seed}] {r}", flush=True)
    del m

print("=" * 78, flush=True)
print("ARM B: echo-organ (Dyck-echo) x 4 seeds  [pass: echo-dCE@4096 <= 0.10]", flush=True)
for seed in range(4):
    torch.manual_seed(seed)
    m = gE["EchoModel"]()
    if seed == 0:
        print(f"[arm B] params={n_params(m)}", flush=True)
    train_echo(m, seed)
    e = eval_echo_type(m)
    r = {"echo_dce@4096": round(e, 4), "pass": e <= 0.10}
    RESULTS[f"B_echo_s{seed}"] = r
    print(f"  [B s{seed}] {r}", flush=True)
    del m

print("=" * 78, flush=True)
print("ARM C: tf_rope (hard tasks) x 3 seeds  [same pass bars as arm A]", flush=True)
for seed in range(3):
    torch.manual_seed(seed)
    m = TF27()
    if seed == 0:
        print(f"[arm C] params={n_params(m)}", flush=True)
    train_hard(m, seed, f"tf s{seed}")
    r = {"mod7@4096": round(eval_hard(m, "mod7"), 4),
         "dyck10@4096": round(eval_hard(m, "dyck10"), 4)}
    r["pass"] = r["mod7@4096"] <= 0.05 and r["dyck10@4096"] <= 0.30
    RESULTS[f"C_tf_s{seed}"] = r
    print(f"  [C s{seed}] {r}", flush=True)
    del m

print("\n" + "=" * 78)
print("P9 SEED SWEEP — pass rates (lottery = 1 - pass rate)")
print("=" * 78)
for arm in ("A", "B", "C"):
    rows = {k: v for k, v in RESULTS.items() if k.startswith(arm + "_")}
    n = len(rows); npass = sum(1 for v in rows.values() if v["pass"])
    print(f"arm {arm}: {npass}/{n} seeds pass -> lottery {100.0 * (n - npass) / n:.0f}%", flush=True)
    for k in sorted(rows):
        print(f"  {k}: {rows[k]}", flush=True)
print("=" * 78)
final = {"tag": "ARC2-C10-P9-SEEDS", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
