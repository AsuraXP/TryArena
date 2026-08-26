#!/usr/bin/env python3
"""ARCH-VET TF-only re-run (cycle 51): retrain the micro-Transformer arm
and run the full eval (CE @256/512/1024 + per-task acc train/hard).
PE buffer fixed to max_len=2048 (1025-token CE@1024 probe)."""
import json
import random
import time

import torch

import arch_vet_lm as A

T0 = time.time()


def main():
    torch.manual_seed(0)
    m = A.TFMicro(A.V, 16)
    print(f"[TF] params={A.n_params(m)}", flush=True)
    pool = A.make_pool(512, 256, 12345)
    hist = A.train_arm("TF", m, pool, 2000, 8)
    rng = random.Random(999)
    vtr = A.make_batches(8, 256, rng)
    rh = random.Random(777)
    vha = A.make_batches(8, 256, rh, hard=True)
    r512 = random.Random(31337)
    v512 = A.make_batches(2, 512, r512)
    r1024 = random.Random(31415)
    v1024 = A.make_batches(2, 1024, r1024)
    m.eval()
    ce = {"256_train": A.val_ce(m, vtr, 256),
          "256_hard": A.val_ce(m, vha, 256),
          "512": A.val_ce(m, v512, 512),
          "1024": A.val_ce(m, v1024, 1024)}
    acc_tr = A.task_acc(m, 24, 256, random.Random(555), hard=False)
    acc_ev = A.task_acc(m, 24, 256, random.Random(666), hard=True)
    result = {"tag": "ARCH-VET-LM-1-TF", "architectures": {
        "TF": {"params": A.n_params(m), "loss_curve": hist, "ce": ce,
               "acc_train_interval": acc_tr,
               "acc_eval_interval": acc_ev}},
        "note": "TF arm re-run after PE max_len fix (2048); identical "
                "pool/protocol to ARCH-VET-LM-1",
        "wall_s": round(time.time() - T0, 1)}
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
