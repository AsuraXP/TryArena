#!/usr/bin/env python3
"""C52 P5: VETbig full 4-task @4000 steps + CE@2048. Seed 0."""
import os, json, time, sys
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
src = open("arch_vet_lm.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_vet_lm.py", "exec"), globals())
set_threads()
model = VETLM(k=8, d=24, K=8)
print("P5 VETbig params", model.nparams(), flush=True)
hist = train_arm(model, ["TRACK", "MODK", "DYCK", "PAIR"], steps=4000, L=256, B=4,
                 seed=0, log_every=250, L_eval=(256, 512, 1024, 2048))
accs = {t: {str(sp): eval_acc(model, t, n=32, span=sp, L=256, seed=0)
            for sp in [(32, 64), (64, 96)]} for t in ["TRACK", "MODK", "DYCK", "PAIR"]}
print("P5 ACC", accs, flush=True)
ce_last = {str(k): v for k, v in hist[-1][2].items()}
ce2048 = hist[-1][2].get(2048, None)
rec = {"tag": "ARCH-VET-LM-P5", "params": model.nparams(), "accs": accs,
       "ce_last": ce_last, "ce2048": ce2048}
open("log.jsonl", "a").write(json.dumps(rec) + "\n")
print("RESULT", rec, flush=True)
