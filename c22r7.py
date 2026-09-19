"""C22-R round 7: organ push amplification (st_m x2) + recalibration.

Round-6 margin analysis at the overwrite scored position (c22r6.pt):
  organ heidi +2.19 vs other names ~-0.7 (margin 2.9), host flat ~4.0 ->
  p(correct)=0.73, CE=0.3211. Bar CE<=0.05 needs margin ~4.9 (+2 nats).
Fix: double st_m (bilinear push), then recalibrate with a short long-window
fine-tune (1500 steps, L=1024, b4, lr 1e-4, 50% overwrite-distance streams).
Wall ~12 min. Tag ARC2-C22R7-REPAIR.
"""
import json, math, random, time, types

import torch
import torch.nn.functional as F

T0 = time.time()

src = open("c22r4.py").read()
cut = src.index("# ============================================================== smoke")
ns = {}
exec(compile(src[:cut], "c22r4.py", "exec"), ns)
g = ns["g"]
V9 = ns["DialogMachineV9b"]
gen_w, train_step = ns["gen_w"], ns["train_step"]
probe_w, overwrite_probe, dialogue_gen = ns["probe_w"], ns["overwrite_probe"], ns["dialogue_gen"]
routing_acc = ns["routing_acc"]
PLUS, MINUS = g["PLUS"], g["MINUS"]
H = ns["mod"].__dict__ if "mod" in ns else ns
print(f"[c22r7] preamble loaded ({time.time()-T0:.0f}s)", flush=True)

# --- overwrite-distance generator (same as round 6) ---
LN8, LN10, H8 = math.log(8.0), math.log(10.0), math.log(3.0)
(U, A, MY, NAME, IS, NOW, CODE, WHAT, OK, FINE, GOOD, N0) = (
    g[k] for k in ["U", "A", "MY", "NAME", "IS", "NOW", "CODE", "WHAT",
                    "OK", "FINE", "GOOD", "N0"])

def gen_ow(batch, length, rng):
    xs, ys, os_, tasks = [], [], [], []
    _fname_turn, _fcode_turn = H["_fname_turn"], H["_fcode_turn"]
    _qname_turn, _qcode_turn = H["_qname_turn"], H["_qcode_turn"]
    def emit(x, nll, u, a, ent_u, u_ent):
        x += [U] + u + [A] + a
        nll += [u_ent] + ent_u + [0.0] + [0.0] * len(a)
    P_my, P_what, P_fill = 0.19, 0.4576, 0.0953
    for _ in range(batch):
        name = N0 + rng.randrange(8)
        code = [rng.randrange(10), rng.randrange(10)]
        x, nll = [], []
        u, _ = _fname_turn(rng)
        emit(x, nll, u, [OK], [0.0] * len(u), -math.log(P_my))
        u, _ = _fcode_turn(rng)
        emit(x, nll, u, [OK], [0, 0, 0, LN10, LN10], -math.log(P_my))
        if rng.random() < 0.5:
            u, _ = _fname_turn(rng, now=True)
            name = u[-1]
            emit(x, nll, u, [OK], [0.0, 0.0, 0.0, 0.0, LN8], -math.log(P_my))
        else:
            u, _ = _fcode_turn(rng)
            code = [u[3], u[4]]
            emit(x, nll, u, [OK], [0, 0, 0, LN10, LN10], -math.log(P_my))
        while len(x) < length - 16:
            w = rng.choice([OK, FINE, GOOD])
            emit(x, nll, [w], [w], [H8], -math.log(P_fill))
        u, a = _qname_turn(name)
        emit(x, nll, u, a, [0.0] * len(u), -math.log(P_what))
        u, a = _qcode_turn(*code)
        emit(x, nll, u, a, [0.0] * len(u), -math.log(P_what))
        xs.append(torch.tensor(x[:length]))
        ys.append(torch.tensor(x[1:length + 1]))
        os_.append(torch.tensor(nll[1:length + 1]))
        tasks.append(0)
    return (torch.stack(xs), torch.stack(ys), torch.stack(os_),
            torch.tensor(tasks))

torch.manual_seed(0)
m = V9()
sd = torch.load("c22r6.pt")
sd["st_m"] = sd["st_m"] * 2.0
m.load_state_dict(sd)
print(f"[c22r7] st_m doubled, st_m_abs={float(m.st_m.abs().sum()):.1f}", flush=True)
print(f"[c22r7] overwrite right after scaling: {overwrite_probe(m, 4096)}", flush=True)
m.train()
organ_ids = {id(p) for p in (m.st_m, m.st_add, m.math_table)}
organ_p = [p for p in m.parameters() if id(p) in organ_ids]
rest_p = [p for p in m.parameters() if id(p) not in organ_ids]
opt = torch.optim.AdamW([{"params": rest_p},
                         {"params": organ_p, "weight_decay": 0.0}], lr=1e-4)
rng = random.Random(53)
t0 = time.time()
LFT, STEPS = 1024, 1500
for step in range(1, STEPS + 1):
    if step % 2 == 0:
        x, y, o, task = gen_w(4, LFT, rng)
    else:
        x, y, o, task = gen_ow(4, LFT, rng)
    lm, rt = train_step(m, opt, x, y, task)
    if step % 300 == 0:
        print(f"  [ft] step {step}/{STEPS} lm {lm:.4f} "
              f"st_m_abs {float(m.st_m.abs().sum()):.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
torch.save(m.state_dict(), "c22r7.pt")
print(f"[ft] done in {time.time()-t0:.0f}s", flush=True)

res = {"state4096": probe_w(m, 0, 4096),
       "state16384": probe_w(m, 0, 16384),
       "mathplus4096": probe_w(m, 1, 4096, 1, op=PLUS),
       "mathminus4096": probe_w(m, 1, 4096, 1, op=MINUS),
       "chat4096": probe_w(m, 2, 4096),
       "overwrite4096": overwrite_probe(m, 4096),
       "routing": routing_acc(m),
       "head_gates": [round(float(v), 3) for v in torch.exp(m.head_gate)],
       "st_m_abs": round(float(m.st_m.abs().sum()), 1)}
print(f"[eval-r7] {res}", flush=True)
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
final = {"tag": "ARC2-C22R7-REPAIR",
         "changes": ["st_m x2 from c22r6.pt", "1.5k steps L=1024 b4 lr 1e-4 "
                     "50% overwrite-distance streams"],
         "r7": res, "bars": bars, "wall_s": round(time.time() - T0, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
