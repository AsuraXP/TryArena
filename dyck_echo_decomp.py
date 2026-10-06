"""
ARC-2 CYCLE 8 / DYCK-ECHO follow-up: per-token-type dCE decomposition.
The total-dCE table is distorted by the L4/L2 oracle convention (it overcounts
opens at depth>0 and close-decision coins). The ECHO tokens (E0/E1/Z) have a
TRUE oracle of exactly 0 (deterministic given the stack) and are where the
non-regular boundary lives. Recompute dCE per token type at 4096 (and 512):
  - hero (dyke_s0.pt loaded), ssm_d16_1 (retrained, seed 0), tf_rope (retrained, seed 0)
Win statement: hero echo-dCE ~ 0 while ssm echo-dCE ~ ln2/2-scale and tf worse.
USAGE: OMP_NUM_THREADS=1 python3 -u dyck_echo_decomp.py
"""
import json, math, resource, time
import torch
import torch.nn.functional as F

t_start = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location("de", "dyck_echo.py")
# dyck_echo.py runs its experiment on import - instead exec its definitions only
src = open("dyck_echo.py").read()
defs = src.split("# ---------------------------------------------------------------- experiment")[0]
g = {"__name__": "x"}
exec(defs, g)

DEVICE = "cpu"
torch.set_num_threads(1)
CFG = g["CFG"]
VOCAB = g["VOCAB"]

@torch.no_grad()
def eval_type_dce(model, L, reps=2):
    model.eval()
    bs = max(1, min(8, 4096 // L))
    ce = orc = n = 0
    per = {t: [0.0, 0.0, 0] for t in range(VOCAB)}  # ce, orc, n
    for i in range(reps):
        rng = __import__("random").Random(700_000 + L + i)
        x, y, o = g["gen_echo"](bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        for t in range(VOCAB):
            m = (y == t)
            per[t][0] += nll[m].sum().item()
            per[t][1] += o[m].sum().item()
            per[t][2] += int(m.sum())
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
    out = {"total_dce": round((ce - orc) / n, 4),
           "echo_dce (E0+E1+Z, oracle=0 exactly)": round(
               (per[3][0] + per[4][0] + per[5][0]) / max(1, per[3][2] + per[4][2] + per[5][2]), 4),
           "open_dce (O0+O1)": round(
               (per[0][0] + per[1][0] - per[0][1] - per[1][1]) / max(1, per[0][2] + per[1][2]), 4),
           "close_dce (C)": round((per[2][0] - per[2][1]) / max(1, per[2][2]), 4),
           "echo_frac": round((per[3][2] + per[4][2] + per[5][2]) / n, 3)}
    return out

ALL = {}
torch.manual_seed(0)
hero = g["EchoModel"]()
hero.load_state_dict(torch.load("dyke_s0.pt", weights_only=True))
for L in [512, 4096]:
    ALL[f"echo L{L}"] = eval_type_dce(hero, L)
del hero

torch.manual_seed(0)
ssm = g["PlainSSM"]()
g["train_model"](ssm, 0, "ssm")
for L in [512, 4096]:
    ALL[f"ssm L{L}"] = eval_type_dce(ssm, L)
del ssm

torch.manual_seed(0)
tf = g["TransformerLM"]()
g["train_model"](tf, 0, "tf")
for L in [512, 4096]:
    ALL[f"tf L{L}"] = eval_type_dce(tf, L)

print("=" * 92)
print("DYCK-ECHO per-token-type dCE  (echo oracle = 0 exactly; echo = the non-regular read)")
print("=" * 92)
for k, r in ALL.items():
    print(f"{k:<14} total {r['total_dce']:<8} echo {r['echo_dce (E0+E1+Z, oracle=0 exactly)']:<8} "
          f"open {r['open_dce (O0+O1)']:<8} close {r['close_dce (C)']:<8} echo_frac {r['echo_frac']}", flush=True)
print("=" * 92)
final = {"tag": "ARC2-C8-DYCKECHO-DECOMP", "runs": ALL,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
