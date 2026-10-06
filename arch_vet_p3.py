# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 3 — A3 (full base VETLM) re-run, isolated process,
torch.manual_seed(0) BEFORE model construction. NOTE: the Phase-1 base
was constructed on the FRESH default RNG state, which is entropy-seeded
(verified != seed-0 state), so a bit-identical cross-process repeat is
IMPOSSIBLE by design; the step-250 check below reports that honestly.
P3 therefore supplies a THIRD init sample of the base model (after the
Phase-1 fresh-init and the P2 post-A1/A2-init samples) — trajectory
variance of the full architecture. Tag ARCH-VET-LM-P3.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

_src = open("arch_vet_lm.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns)
V = _ns["V"]; VETLM = _ns["VETLM"]
make_pool = _ns["make_pool"]; make_batches = _ns["make_batches"]
train_arm = _ns["train_arm"]; val_ce = _ns["val_ce"]
task_acc = _ns["task_acc"]; n_params = _ns["n_params"]

pool = make_pool(512, 256, 12345)
torch.manual_seed(0)                    # seed BEFORE construction (Phase-1 parity)
m = VETLM(V, 16, k=5, K=4)
print(f"[A3] params={n_params(m)}", flush=True)
hist = train_arm("A3", m, pool, 2000, 8)

m.eval()
rng = random.Random(999);  vtr = make_batches(8, 256, rng)
rh  = random.Random(777);  vha = make_batches(8, 256, rh, hard=True)
r5  = random.Random(31337); v512 = make_batches(2, 512, r5)
r10 = random.Random(31415); v1024 = make_batches(2, 1024, r10)
ce = {"256_train": val_ce(m, vtr, 256),
      "256_hard": val_ce(m, vha, 256),
      "512": val_ce(m, v512, 512),
      "1024": val_ce(m, v1024, 1024)}
at = task_acc(m, 24, 256, random.Random(555), hard=False)
ae = task_acc(m, 24, 256, random.Random(666), hard=True)
bit = abs(hist[0][1] - 1.4488) < 1e-9
print(f"[A3] ce={ce}", flush=True)
print(f"[A3] acc(train)={ {k: round(v,3) for k,v in at.items()} }", flush=True)
print(f"[A3] acc(eval)={ {k: round(v,3) for k,v in ae.items()} }", flush=True)
print(f"[A3] bit_identical_to_phase1={bit} (step250 {hist[0][1]} vs 1.4488)", flush=True)
result = {"tag": "ARCH-VET-LM-P3",
          "note": "A3 base standalone, seed0-before-construction (Phase-1 default RNG state is entropy-seeded, bit-parity expected False); 3rd init sample of base",
          "params": n_params(m), "loss_curve": hist, "ce": ce,
          "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
          "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()},
          "bit_identical_to_phase1": bit,
          "wall_s": round(time.time() - T0, 1)}
print("RESULT " + json.dumps(result), flush=True)
with open("log.jsonl", "a") as fh:
    fh.write(json.dumps(result) + "\n")
print("DONE", flush=True)
