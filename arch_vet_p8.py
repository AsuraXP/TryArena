#!/usr/bin/env python3
"""C52 P8 VETCAM — content-addressed LIFO READOUT.
Write path stays exact STE; read = learned-temperature softmax over cos-sim(xt, buf_j)
with top-of-stack fallback. Attack on L-LIFO-INIT-FRAGILE. Seeds 0,111.
Prior: Graves NTM 2014 content addressing; Bahdanau 2015; RAM (Mnih 2014).
"""
import os, json
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
src = open("arch_vet_lm.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_vet_lm.py", "exec"), globals())
set_threads()
out = []
for seed in (0, 111):
    m = VETLM(k=5, d=16, K=4, cam=True)
    print(f"P8 VETCAM seed {seed} p={m.nparams()}", flush=True)
    hist = train_arm(m, ["TRACK", "MODK", "DYCK", "PAIR"], steps=2000, L=128, B=4,
                     seed=seed, log_every=500, L_eval=(128, 256))
    pair = eval_acc(m, "PAIR", n=48, span=(16, 48), L=128, seed=seed)
    rec = {"seed": seed, "params": m.nparams(), "pair_ev": pair, "ce": hist[-1][2]}
    print("P8", rec, flush=True)
    out.append(rec)
rec = {"tag": "ARCH-VET-LM-P8", "seeds": out}
open("log.jsonl", "a").write(json.dumps(rec) + "\n")
print("RESULT", rec, flush=True)
