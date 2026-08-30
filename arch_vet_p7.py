#!/usr/bin/env python3
"""C52 P7 DIVIDE frontier. d in {3,4}, quotients tokens 39-47. VETbase/VETbig/MAMBA. P4 budget.
Prior: long division as iterated subtraction (Knuth); neural arithmetic (Trask NALU 2018).
"""
import os, json
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
src = open("arch_vet_lm.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_vet_lm.py", "exec"), globals())
set_threads()
arms = {
    "VETbase": lambda: VETLM(k=5, d=16, K=4),
    "VETbig": lambda: VETLM(k=8, d=24, K=8),
    "MAMBA": lambda: MambaMicro(),
}
results = {}
for name, ctor in arms.items():
    m = ctor()
    print(f"P7 {name} p={m.nparams()}", flush=True)
    hist = train_arm(m, ["DIV"], steps=1500, L=96, B=4, seed=0, log_every=500, L_eval=(96, 192))
    acc = eval_acc(m, "DIV", n=40, span=(16, 40), L=96, seed=0)
    results[name] = {"params": m.nparams(), "acc": acc, "ce": hist[-1][2]}
    print("P7", name, results[name], flush=True)
rec = {"tag": "ARCH-VET-LM-P7", **results}
open("log.jsonl", "a").write(json.dumps(rec) + "\n")
print("RESULT", rec, flush=True)
