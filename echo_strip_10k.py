"""
ARC-2 CYCLE 10 / PHASE-4 STRIP-DOWN: is the host needed on Dyck-echo?
=====================================================================
The echo organ's win lives in the exact (top, empty) readout table (24 params),
which already knows the O/C distribution (it sees `empty`). The host's only
added value should be U-schedule exploitation (position-dependent p_open) —
worth ~ -0.29 total-dCE. Test: TABLE-ONLY organ (24 params, no host) x 2 seeds
vs the 2,974-param host+organ (dyke_s0.pt reference).
Expected: table-only echo-dCE ~ 0.01 (exact), total-dCE ~ 0.00 (no schedule
exploit); hero total ~ -0.29. -> the exactness is the ORGAN's, not the host's.
USAGE: OMP_NUM_THREADS=1 python3 -u echo_strip.py
"""
import json, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
t_start = time.time()

echo_src = open("dyck_echo.py").read()
echo_defs = echo_src.split("# ---------------------------------------------------------------- experiment")[0]
gE = {"__name__": "echo_strip"}
exec(echo_defs, gE)
CFG = dict(gE["CFG"]); CFG["steps"] = 10000
VOCAB = gE["VOCAB"]

class TableOnly(nn.Module):
    """24-param organ: exact K-stack features -> (top,empty) readout table."""
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.1 * torch.randn(4, VOCAB))

    def forward(self, x):
        f = gE["EchoModel"].features(self, x)   # unbound method, self unused
        combo = f[:, :, 0] + f[:, :, 1] * 2
        return self.table[combo]

@torch.no_grad()
def eval_full(model, L=4096, reps=2):
    model.eval()
    bs = max(1, min(8, 4096 // L))
    ce = orc = n = 0
    per = {t: [0.0, 0] for t in range(VOCAB)}
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gE["gen_echo"](bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        for t in range(VOCAB):
            m = (y == t)
            per[t][0] += nll[m].sum().item(); per[t][1] += int(m.sum())
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
    echo_n = per[3][1] + per[4][1] + per[5][1]
    open_n = per[0][1] + per[1][1]
    return {
        "total": round((ce - orc) / n, 4),
        "echo": round((per[3][0] + per[4][0] + per[5][0]) / max(1, echo_n), 4),
        "open": round((per[0][0] + per[1][0] - 0.0) / max(1, open_n), 4),
    }

def train_table(model, seed):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        x, y, o = gE["gen_echo"](CFG["batch"], CFG["train_len"], rng)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        opt.step(); opt.zero_grad()
        if step % 1000 == 0:
            print(f"  [tbl s{seed}] step {step}/{CFG['steps']} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

RESULTS = {}
for seed in range(1):
    torch.manual_seed(seed)
    m = TableOnly()
    if seed == 0:
        print(f"[run] table-only params={sum(p.numel() for p in m.parameters())}", flush=True)
    train_table(m, seed)
    r = eval_full(m, 4096, 2)
    r["params"] = 24
    RESULTS[f"table_only_s{seed}"] = r
    print(f"  [tbl s{seed}] @4096 {r}", flush=True)
    del m

# hero reference (dyke_s0.pt)
torch.manual_seed(0)
hero = gE["EchoModel"]()
hero.load_state_dict(torch.load("dyke_s0.pt", weights_only=True))
hr = eval_full(hero, 4096, 2)
hr["params"] = 2974
RESULTS["hero_ref (dyke_s0)"] = hr
print(f"  [hero]        @4096 {hr}", flush=True)

print("\n" + "=" * 72)
print("ECHO STRIP-DOWN @4096 (total dCE | echo-dCE | open-CE)")
print("=" * 72)
for k, v in RESULTS.items():
    print(f"{k:<24} params {v['params']:<6} total {v['total']:<9} echo {v['echo']:<8} open {v['open']}", flush=True)
print("=" * 72)
final = {"tag": "ARC2-C10-ECHO-STRIP-10K", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
