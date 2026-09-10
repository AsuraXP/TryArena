# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 5 (cycle 52) — structure at a larger budget:
VETbig (k=8, d=24, K=8, 20,697 p) on the FULL 4-task stream
(TRACK/MODK/DYCK/PAIR), 4000 steps (2x P1 budget), same pool /
probes as P1 + CE@2048 (no PE in VET -> valid). Question: does
the structure lift DYCK depth 3-4 and MODK beyond the micro
budget? Tag ARCH-VET-LM-P5.
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

if __name__ == "__main__":
    pool = make_pool(512, 256, 12345)
    torch.manual_seed(0)
    m = VETLM(V, 24, k=8, K=8)
    print(f"[P5] VETbig params={n_params(m)}", flush=True)
    hist = train_arm("P5-VETbig", m, pool, 4000, 8)
    m.eval()
    rng = random.Random(999);  vtr = make_batches(8, 256, rng)
    rh  = random.Random(777);  vha = make_batches(8, 256, rh, hard=True)
    r5  = random.Random(31337); v512 = make_batches(2, 512, r5)
    r10 = random.Random(31415); v1024 = make_batches(2, 1024, r10)
    r20 = random.Random(31416); v2048 = make_batches(2, 2048, r20)
    ce = {"256_train": val_ce(m, vtr, 256),
          "256_hard": val_ce(m, vha, 256),
          "512": val_ce(m, v512, 512),
          "1024": val_ce(m, v1024, 1024),
          "2048": val_ce(m, v2048, 2048)}
    at = task_acc(m, 24, 256, random.Random(555), hard=False)
    ae = task_acc(m, 24, 256, random.Random(666), hard=True)
    print(f"[P5] ce={ce}", flush=True)
    print(f"[P5] acc(train)={ {k: round(v,3) for k,v in at.items()} }", flush=True)
    print(f"[P5] acc(eval)={ {k: round(v,3) for k,v in ae.items()} }", flush=True)
    result = {"tag": "ARCH-VET-LM-P5",
              "protocol": "VETbig k=8 d=24 K=8 on full 4-task stream, "
                          "4000 steps, same pool/probes as P1 + CE@2048",
              "params": n_params(m), "loss_curve": hist, "ce": ce,
              "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
              "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()},
              "wall_s": round(time.time() - T0, 1)}
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
