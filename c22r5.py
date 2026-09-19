"""C22-R round 5: long-window fine-tune of v9c (c22r4.pt).

Round-4 (ARC2-C22R4-REPAIR) fixed the length collapse (SSM decay clamp) and
passes D3/D4/D5/D6/D7; state4096=0.072, overwrite=0.9665 remain. Diagnosis:
at train length 63 the organ push (~+1.2 logit) only needs to beat the host's
short-range name memory; at L=4096 the host's spread flattens and the push is
insufficient (p(correct)~0.4). The organ learned exactly as much as L=63
demands (L-TRAIN-LENGTH-MISMATCH).

Fix: continue training from c22r4.pt on LONG windows (L=512, batch 8) so the
organ/host must solve answers at range. 3000 steps, lr 3e-4, organ wd=0,
decay clamp kept. Wall budget ~12 min. Tag ARC2-C22R5-REPAIR.
"""
import json, math, random, time, types

import torch
import torch.nn.functional as F

T0 = time.time()

# load c22r4 machinery (preamble up to smoke)
src = open("c22r4.py").read()
cut = src.index("# ============================================================== smoke")
mod = types.ModuleType("c4head")
exec(compile(src[:cut], "c22r4.py", "exec"), mod.__dict__)
g = mod.__dict__
V9 = g["DialogMachineV9b"]          # (class name retained from r3 lineage)
gen_w, train_step = g["gen_w"], g["train_step"]
stream_probe_w, overwrite_probe, dialogue_gen = (g["probe_w"],
                                                 g["overwrite_probe"],
                                                 g["dialogue_gen"])
routing_acc = g["routing_acc"]
PLUS, MINUS = g["PLUS"], g["MINUS"]
print(f"[c22r5] preamble loaded ({time.time()-T0:.0f}s)", flush=True)

torch.manual_seed(0)
m = V9()
m.load_state_dict(torch.load("c22r4.pt"))
m.train()
organ_ids = {id(p) for p in (m.st_m, m.st_add, m.math_table)}
organ_p = [p for p in m.parameters() if id(p) in organ_ids]
rest_p = [p for p in m.parameters() if id(p) not in organ_ids]
opt = torch.optim.AdamW([{"params": rest_p},
                         {"params": organ_p, "weight_decay": 0.0}], lr=3e-4)
rng = random.Random(29)
t0 = time.time()
LFT, B, STEPS = 512, 8, 3000
for step in range(1, STEPS + 1):
    x, y, o, task = gen_w(B, LFT, rng)
    lm, rt = train_step(m, opt, x, y, task)
    if step % 500 == 0:
        print(f"  [ft] step {step}/{STEPS} lm {lm:.4f} "
              f"st_m_abs {float(m.st_m.abs().sum()):.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
torch.save(m.state_dict(), "c22r5.pt")
print(f"[ft] done in {time.time()-t0:.0f}s", flush=True)

res = {"state4096": stream_probe_w(m, 0, 4096),
       "state16384": stream_probe_w(m, 0, 16384),
       "mathplus4096": stream_probe_w(m, 1, 4096, 1, op=PLUS),
       "mathminus4096": stream_probe_w(m, 1, 4096, 1, op=MINUS),
       "chat4096": stream_probe_w(m, 2, 4096),
       "overwrite4096": overwrite_probe(m, 4096),
       "routing": routing_acc(m),
       "head_gates": [round(float(v), 3) for v in torch.exp(m.head_gate)],
       "st_m_abs": round(float(m.st_m.abs().sum()), 1)}
print(f"[eval-r5] {res}", flush=True)
dlg = dialogue_gen(m)
print("[dialogue]", flush=True)
print(dlg, flush=True)
D7 = ("dave" in dlg.split("\n")[3] and "4 2" in dlg.split("\n")[9]
      and "1 2" in dlg.split("\n")[5] and " 6" in dlg.split("\n")[8])
bars = {"D1_state_le_0.01": res["state4096"] <= 0.01,
        "D2_overwrite_le_0.05": res["overwrite4096"] <= 0.05,
        "D3_16k_le_4k+0.05": res["state16384"] <= res["state4096"] + 0.05,
        "D4_mathplus_le_0.02": res["mathplus4096"] <= 0.02,
        "D4_mathminus_le_0.05": res["mathminus4096"] <= 0.05,
        "D5_chat_le_0.02": res["chat4096"] <= 0.02,
        "D6_routing_1.0": res["routing"] == 1.0,
        "D7_dialogue_exact": bool(D7)}
print(f"[bars] {bars}", flush=True)
final = {"tag": "ARC2-C22R5-REPAIR",
         "changes": ["long-window fine-tune L=512 b8 3k steps lr 3e-4 "
                     "from c22r4.pt"],
         "r5": res, "bars": bars, "wall_s": round(time.time() - T0, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
