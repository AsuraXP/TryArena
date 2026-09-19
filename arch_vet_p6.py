# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 6 (cycle 52) — basin rate of the full base
architecture's favorable basin (P3 found pair-ev .604 under one
init; P1 .057 / P2-A3 .094 under two others). 3 fresh init seeds
(111/222/333, torch.manual_seed BEFORE construction) x 2000 steps,
same pool/probes as P1. Reports the distribution of eval accs and
the fraction of seeds with pair-ev >= 0.5 (the P3 basin).
Tag ARCH-VET-LM-P6.
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
    result = {"tag": "ARCH-VET-LM-P6",
              "protocol": "base VETLM k=5 d=16 K=4, 3 init seeds "
                          "(111/222/333, seed before construction), "
                          "2000 steps each, same pool/probes as P1; "
                          "adds to P1(fresh default RNG)/P2-A3/P3(0) "
                          "init samples",
              "seeds": {}}
    for seed in (111, 222, 333):
        torch.manual_seed(seed)
        m = VETLM(V, 16, k=5, K=4)
        print(f"[P6 seed {seed}] params={n_params(m)}", flush=True)
        hist = train_arm(f"P6-{seed}", m, pool, 2000, 8)
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
        print(f"[P6 seed {seed}] ce={ce}", flush=True)
        print(f"[P6 seed {seed}] acc(eval)={ {k: round(v,3) for k,v in ae.items()} }", flush=True)
        result["seeds"][str(seed)] = {
            "ce": ce,
            "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
            "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()}}
    n_basin = sum(1 for s in result["seeds"].values()
                  if s["acc_eval_interval"]["pair"] >= 0.5)
    result["pair_ev_ge_05_rate_over_3"] = n_basin
    result["note"] = "combined 6-init sample (incl P1/P2-A3/P3) in log block"
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
