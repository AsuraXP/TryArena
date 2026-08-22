"""
C14 DIAGNOSTIC 2: locate the echo-branch multiplexing failure.
Ablation proved: branch0 components + pure echo @ batch32 = cert (-0.298).
Iso run proved: same components in the mixed model (8/32 duty) = 1.14 @20k.
Bisect with the REAL full IsoModel forward (grouped + router + scatter):
  E1: pure echo stream, all 32 rows task 0, L=63, 2500 steps
  E2: mixed 24/4/4 (echo duty 75%), L=63, 2500 steps
Report echo dCE @4096 + routing acc. Cert = -0.29; iso_20k = 1.1064.
"""
import json, random, time
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB = 45
g = {"__name__": "diag2"}
exec(open("unified_iso.py").read().split("\nRESULTS = {}")[0], g)
IsoModel, n_params, train_step, eval_task = g["IsoModel"], g["n_params"], g["train_step"], g["eval_task"]
gen_echo_t, gen_icl_t, gen_mod7_t = g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"]

def gen_pure_echo(batch, length, rng):
    x, y, o = gen_echo_t(batch, 64, rng)
    task = torch.zeros(batch, dtype=torch.long)
    return x[:, :length], y[:, :length], task

def gen_mixed24(batch, length, rng):
    n0, n1 = 24, 4
    x0, y0, _ = gen_echo_t(n0, 64, rng)
    x1, y1, _ = gen_icl_t(n1, 64, rng)
    x2, y2, _ = gen_mod7_t(batch - n0 - n1, length, rng)
    x = torch.cat([x0[:, :length], x1, x2])
    y = torch.cat([y0[:, :length], y1, y2])
    task = torch.cat([torch.zeros(n0), torch.ones(n1),
                      torch.full((batch - n0 - n1,), 2.0)]).long()
    return x, y, task

def run(tag, gen, steps=2500, seed=0):
    torch.manual_seed(seed)
    m = IsoModel()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y, task = gen(32, 63, rng)
        lm, rt = train_step(m, opt, x, y, task)
        if step % 1000 == 0:
            print(f"    [{tag}] step {step}/{steps} lm {lm:.4f} rt {rt:.4f}", flush=True)
    d = eval_task(m, gen_echo_t, 4096, 2, 0)
    print(f"  {tag:<28} echo dCE@4096 = {d}   ({time.time()-t0:.0f}s)", flush=True)
    return d

print("C14 DIAG-2 — full IsoModel forward, echo dCE@4096 (cert=-0.29, iso_20k=1.1064):", flush=True)
run("E1: pure echo 32/32", gen_pure_echo)
run("E2: mixed 24/4/4", gen_mixed24)
print("DONE", flush=True)
