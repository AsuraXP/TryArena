"""
ARC-2 CYCLE 14b / ISOLATED mixture, duty-cycle fix (TASK CYCLING)
=================================================================
C14 diagnosis chain (all controlled, logged):
  C13 shared host @20k: knee — echo -0.30 -> +1.13 (P15 = interference)
  C14 iso @20k:  knee GONE (1.14 -> 1.11) but organ branches stuck below
                 cert (echo 1.11 vs -0.30; icl tgt 0.28 vs 0.02)
  ablation_c14:  branch components + pure echo @batch32 = -0.298 (A/B/C/D)
  iso_echo_diag: full IsoModel, pure echo 32/32 = -0.2937; mixed 24/4/4 =
                 -0.3009 (both cert @2500) vs mixed 8/16/8 = 1.14 @20k
  => cause = per-branch DUTY CYCLE: 8 rows/step starves the organ branches
     of full-batch gradient signal (high-variance small-batch pathology);
     24+ rows/step converges to cert.
FIX: task cycling — round-robin pure-task FULL batches (32 rows each).
Each organ branch receives EXACTLY its standalone-certified protocol
(batch 32, L 63/64, pure-task stream); the router co-trains (sees each
family 1/3 of steps); zero parameter sharing (interference already gone).
Protocol: 10000 optimizer steps = ~3333 per task (1.33x the 2500-step
standalone certs); checkpoints at 3000 (1000/task) and 9000 (3000/task).
Baseline (protocol): micro TF 10k steps on the C13 mixed 8/16/8 stream.
Win = ALL three tasks at standalone-cert level inside ONE model, stable
across 1x -> 1.33x standalone budget (knee stays gone), routing 1.0.
USAGE: OMP_NUM_THREADS=1 python3 -u unified_iso2.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB = 45
CFG = dict(steps=10000, batch=32, train_len=63, ckpts=[3000, 9000],
           d_model=16, KSTACK=4096)
print(f"[setup] unified-iso2 (task cycling) cfg={CFG}", flush=True)
t_start = time.time()

g = {"__name__": "u2"}
exec(open("unified_iso.py").read().split("\nRESULTS = {}")[0], g)
IsoModel, n_params, train_step, eval_task = (
    g["IsoModel"], g["n_params"], g["train_step"], g["eval_task"])
gen_echo_t, gen_icl_t, gen_mod7_t, gen_mixed = (
    g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"], g["gen_mixed"])

TASKS = {0: gen_echo_t, 1: gen_icl_t, 2: gen_mod7_t}

def gen_pure(r, batch, length, rng):
    if r == 0:
        x, y, _ = gen_echo_t(batch, 64, rng)
        return x[:, :length], y[:, :length], torch.zeros(batch, dtype=torch.long)
    if r == 1:
        x, y, _ = gen_icl_t(batch, 64, rng)
        return x, y, torch.ones(batch, dtype=torch.long)
    x, y, _ = gen_mod7_t(batch, length, rng)
    return x, y, torch.full((batch,), 2, dtype=torch.long)

def eval_all(model, tag, results):
    r = {"echo": eval_task(model, gen_echo_t, 4096, 2, 0),
         "icl": eval_task(model, gen_icl_t, 4096, 2, 1, tgt=True),
         "mod7": eval_task(model, gen_mod7_t, 4096, 2, 2)}
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)

# ---------------------------------------------------------------- experiment
RESULTS = {}
torch.manual_seed(0)
iso = IsoModel()
print(f"[arm] isolated mixture, task cycling params={n_params(iso)}", flush=True)
iso.train()
opt = torch.optim.AdamW(iso.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
ckpts_done = set()
for step in range(1, CFG["steps"] + 1):
    r = (step - 1) % 3
    x, y, task = gen_pure(r, CFG["batch"], CFG["train_len"], rng)
    lm, rt = train_step(iso, opt, x, y, task)
    if step in CFG["ckpts"] and step not in ckpts_done:
        torch.save(iso.state_dict(), f"unified_iso2_{step}.pt")
        ckpts_done.add(step)
        print(f"    [iso2] checkpoint at step {step}", flush=True)
    if step % 2000 == 0:
        print(f"  [iso2 s0] step {step}/{CFG['steps']} lm {lm:.4f} rt {rt:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[iso2] trained in {time.time()-t0:.0f}s", flush=True)
torch.save(iso.state_dict(), "unified_iso2_final.pt")

print("[eval] checkpoints @4096:", flush=True)
for c in CFG["ckpts"]:
    iso.load_state_dict(torch.load(f"unified_iso2_{c}.pt"))
    eval_all(iso, f"iso2_{c // 1000}x", RESULTS)
iso.load_state_dict(torch.load("unified_iso2_final.pt"))
eval_all(iso, "iso2_final", RESULTS)
RESULTS["iso2_final"]["params"] = n_params(iso)
for c in CFG["ckpts"]:
    RESULTS[f"iso2_{c // 1000}x"]["params"] = n_params(iso)
del iso

# protocol baseline: micro TF, mixed 8/16/8 stream, 10k steps
class MixedTF(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4):
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
        mask = nn.Transformer.generate_square_subsequent_mask(L).to("cpu")
        return self.head(self.enc(h, mask))

torch.manual_seed(0)
tf = MixedTF()
print(f"[arm] micro TF (protocol baseline, 10k mixed) params={n_params(tf)}", flush=True)
tf.train()
opt = torch.optim.AdamW(tf.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
for step in range(1, CFG["steps"] + 1):
    x, y, task = gen_mixed(CFG["batch"], CFG["train_len"], rng)
    loss = F.cross_entropy(tf(x).reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 5000 == 0:
        print(f"  [tf] step {step}/{CFG['steps']} CE {loss.item():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

@torch.no_grad()
def eval_tf_task(model, gen, L, reps=2):
    model.eval()
    bs = max(1, min(4, 4096 // L))
    ce = orc = n = 0.0
    tgt_ce = tgt_n = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gen(bs, L, rng)
        nll = -F.log_softmax(model(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return (round((ce - orc) / n, 4), round(tgt_ce / tgt_n, 4))

r = {"echo": eval_tf_task(tf, gen_echo_t, 4096, 2),
     "icl": eval_tf_task(tf, gen_icl_t, 4096, 2),
     "mod7": eval_tf_task(tf, gen_mod7_t, 4096, 2),
     "params": n_params(tf)}
RESULTS["micro_tf_10k"] = r
print(f"  micro_tf_10k: {r}", flush=True)

print("\n" + "=" * 88)
print("ISOLATED mixture + TASK CYCLING @4096 (dCE; routing acc in parens)")
print("standalone certs: echo -0.2935 | icl 0.0217|0.0218 | mod7 0.0025-0.0071")
print("=" * 88)
for k in ["iso2_3x", "iso2_9x", "iso2_final", "micro_tf_10k"]:
    v = RESULTS[k]
    print(f"{k:<13} params {v['params']:<7} echo {v['echo']}  icl {v['icl']}  mod7 {v['mod7']}",
          flush=True)
print("=" * 88)
final = {"tag": "ARC2-C14-ISO2-CYCLING", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
