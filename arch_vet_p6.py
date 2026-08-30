#!/usr/bin/env python3
"""C52 P6: 3-seed basin rate of pair-ev .604 basin; seeds 111/222/333. VETbase, PAIR-focused 4-task 2000 steps."""
import os, json
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
src = open("arch_vet_lm.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_vet_lm.py", "exec"), globals())
set_threads()
out = []
for seed in (111, 222, 333):
    model = VETLM(k=5, d=16, K=4)
    print(f"P6 seed {seed} params {model.nparams()}", flush=True)
    hist = train_arm(model, ["TRACK", "MODK", "DYCK", "PAIR"], steps=2000, L=128, B=4,
                     seed=seed, log_every=500, L_eval=(128, 256))
    pair = eval_acc(model, "PAIR", n=48, span=(16, 48), L=128, seed=seed)
    track = eval_acc(model, "TRACK", n=32, span=(16, 48), L=128, seed=seed)
    rec = {"seed": seed, "pair_ev": pair, "track_ev": track, "ce": hist[-1][2]}
    print("P6 seed result", rec, flush=True)
    out.append(rec)
basin = sum(1 for r in out if r["pair_ev"] >= 0.5) / 3
rec = {"tag": "ARCH-VET-LM-P6", "seeds": out, "basin_rate": basin}
open("log.jsonl", "a").write(json.dumps(rec) + "\n")
print("RESULT", rec, flush=True)
