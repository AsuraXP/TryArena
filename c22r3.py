"""C22-R round 3: chatbot repair — machine v9b (c22r3).

Round-2 (c22r2, ARC2-C22R2-REPAIR) fixed the two root causes and passed 6/7:
  D1 -0.031 PASS, D3 PASS, D4 PASS, D5 PASS, D6 PASS; FAIL: D2 overwrite 0.9445,
  D7 dialogue (math turns inside mixed conversation).

Round-2 residual diagnoses:
  D2: organ wiring is correct (f[name7]=1, qo=qname at A pos, M row pushes heidi
      +1.18 vs offdiag <-0.2) but push too weak: host0 spreads ~4.0+-0.2 over
      all 8 names at 4096, so CE=0.94 needs organ margin ~+5. Limiting factor:
      AdamW weight_decay=0.01 on st_m + finite steps (st_m_abs still rising).
  D7: router is per-STREAM from first 3 tokens; dialogue_gen embeds math turns
      in a state-started conversation -> routed to host0, which has no math
      organ. Also fam0 training streams contain no math turns.

Fixes here:
  F1 organ params (st_m, st_add, math_table) in a weight_decay=0 group.
  F2 fam0 ("state") streams gain math turns (MIX +math=18/118); training and
     probes share the builder; oracle gets the math-turn entropies.
  F3 V9b.forward: host0 (state) branch gets state organ AND math organ.
  F4 16k steps.

Wall budget: ~25 min. Tag ARC2-C22R3-REPAIR.
"""
import json, math, random, time, types

import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

src = open("dialog_chat.py").read()
cut = src.index("# ----------------------------------------------------------------- training")
cut2 = src.index("@torch.no_grad()", cut)
mod = types.ModuleType("dc_head")
exec(compile(src[:cut] + src[cut2:src.index("# ----------------------------------------------------------------- run")],
             "dialog_chat.py", "exec"), mod.__dict__)
g = mod.__dict__
_ts = src.index("def train_step")
exec(compile(src[_ts:cut2], "dialog_chat.py", "exec"), g)
(U, A, MY, NAME, IS, NOW, CODE, WHAT, THE, OK, FINE, GOOD, TELL, ME, IT, AND,
 PLUS, MINUS, N0, VOCAB) = (g[k] for k in
    ["U", "A", "MY", "NAME", "IS", "NOW", "CODE", "WHAT", "THE", "OK", "FINE",
     "GOOD", "TELL", "ME", "IT", "AND", "PLUS", "MINUS", "N0", "VOCAB"])
train_step, gen_dialogue_t = g["train_step"], g["gen_dialogue_t"]
stream_probe, overwrite_probe, dialogue_gen = (g["stream_probe"],
                                               g["overwrite_probe"],
                                               g["dialogue_gen"])
c22 = open("c22r2.py").read()
v9src = c22[c22.index("class DialogMachineV9"):c22.index("# =================================================== corrected probe oracle")]
exec(compile(v9src, "v9", "exec"), g)
DialogMachineV9 = g["DialogMachineV9"]
print(f"[c22r3] preamble loaded ({time.time()-T0:.0f}s)", flush=True)


class DialogMachineV9b(DialogMachineV9):
    """V9 + math organ available in the state-routed branch (D7 fix)."""

    def forward(self, x):
        B, L = x.shape
        rl = self.router(self.emb(x[:, :3]).reshape(B, -1))
        task = rl.argmax(-1)
        hg = torch.exp(self.head_gate)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for r in range(3):
            idx = (task == r).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            xr = x[idx]
            hr = self.norms[r](self.hosts[r](self.emb(xr)))
            lg = hg[r] * self.heads[r](hr)
            if r == 0:
                lg = lg + self._state_logits(xr) + self._math_logits(xr)
            elif r == 1:
                lg = lg + self._math_logits(xr)
            out.scatter_(0, idx.view(-1, 1, 1).expand_as(lg), lg)
        return out, rl


# --------------------- state streams WITH math turns (training + probes)
MIX_W = (45, 18, 18, 10, 9, 18)     # fill qname qcode fnow fcode math
TOT = sum(MIX_W)
P_FIRST = {WHAT: (MIX_W[1] + MIX_W[2] + MIX_W[5]) / TOT,
           MY: (MIX_W[3] + MIX_W[4]) / TOT}
for _w in (OK, FINE, GOOD, TELL):
    P_FIRST[_w] = MIX_W[0] / (4 * TOT)

def _emit_v(x, nll, u, a, ent_u, u_ent):
    assert len(ent_u) == len(u)
    x += [U] + u + [A] + a
    nll += [u_ent] + ent_u + [0.0] + [0.0] * len(a)

def _build_state_w(rng, length):
    _fname_turn, _fcode_turn = g["_fname_turn"], g["_fcode_turn"]
    _qname_turn, _qcode_turn = g["_qname_turn"], g["_qcode_turn"]
    _math_turn, _fill_turn = g["_math_turn"], g["_fill_turn"]
    LN2, LN8, LN10, H8 = g["LN2"], g["LN8"], g["LN10"], g["H8"]
    UM = -math.log(P_FIRST[MY])
    name = N0 + rng.randrange(8)
    code = [rng.randrange(10), rng.randrange(10)]
    x, nll = [], []
    order = [(_fname_turn, [0.0, LN2, 0.0, LN8]),
             (_fcode_turn, [0.0, LN2, 0.0, LN10, LN10])][rng.randrange(2)]
    u, _ = order[0](rng)
    _emit_v(x, nll, u, [OK], order[1], UM)
    if order[0] is _fname_turn:
        u, _ = _fcode_turn(rng)
        _emit_v(x, nll, u, [OK], [0, 0, 0, LN10, LN10], UM)
    else:
        u, _ = _fname_turn(rng)
        _emit_v(x, nll, u, [OK], [0, 0, 0, LN8], UM)
    kinds = (["fill"] * MIX_W[0] + ["qname"] * MIX_W[1] + ["qcode"] * MIX_W[2]
             + ["fnamenow"] * MIX_W[3] + ["fcode"] * MIX_W[4]
             + ["math"] * MIX_W[5])
    while len(x) < length - 14:
        kind = rng.choice(kinds)
        if kind == "fill":
            u, a = _fill_turn(rng)
            ent = [H8] + [0.0] * (len(u) - 1)
        elif kind == "qname":
            u, a = _qname_turn(name)
            ent = [H8] + [0.0] * (len(u) - 1)
        elif kind == "qcode":
            u, a = _qcode_turn(*code)
            ent = [H8] + [0.0] * (len(u) - 1)
        elif kind == "fnamenow":
            u, a = _fname_turn(rng, now=True)
            name = u[-1]
            ent = [H8, 0.0, 0.0, 0.0, LN8]
        elif kind == "fcode":
            u, a = _fcode_turn(rng)
            code = [u[3], u[4]]
            ent = [H8, 0.0, 0.0, LN10, LN10]
        else:
            u, a = _math_turn(rng)
            ent = [0.0, 0.0, LN10, LN2, LN10]
        _emit_v(x, nll, u, a, ent, -math.log(P_FIRST[u[0]]))
    u, a = _qname_turn(name)
    _emit_v(x, nll, u, a, [0.0] * len(u), 0.0)
    u, a = _qcode_turn(*code)
    _emit_v(x, nll, u, a, [0.0] * len(u), 0.0)
    assert len(x) >= length + 1
    return x, nll

def gen_w(batch, length, rng, fam=None, op=None):
    """training + probe generator: fam0 = state+math mix, fam1/2 as original."""
    xs, ys, os_, tasks = [], [], [], []
    for i in range(batch):
        f = fam if fam is not None else i % 3
        if f == 0:
            x, nll = _build_state_w(rng, length)
            xs.append(torch.tensor(x[:length]))
            ys.append(torch.tensor(x[1:length + 1]))
            os_.append(torch.tensor(nll[1:length + 1]))
            tasks.append(f)
        else:
            x, y, o, tk = gen_dialogue_t(1, length, rng, fam=f, op=op)
            xs.append(x[0]); ys.append(y[0]); os_.append(o[0]); tasks.append(f)
    return (torch.stack(xs), torch.stack(ys), torch.stack(os_),
            torch.tensor(tasks))

@torch.no_grad()
def probe_w(model, fam, L, reps=1, op=None):
    model.eval()
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(900_000 + L + fam * 100 + (7 if op else 0) + i)
        x, y, o, _ = gen_w(1, L, rng, fam=fam, op=op)
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
    return round((ce - orc) / n, 4)

@torch.no_grad()
def routing_acc(model, reps=8):
    model.eval()
    ok = tot = 0
    for f in range(3):
        for i in range(reps):
            rng = random.Random(800_000 + f * 100 + i)
            x, y, o, task = gen_w(1, 4096, rng, fam=f)
            _, rl = model(x)
            ok += int(rl.argmax(-1).item() == f); tot += 1
    return ok / tot


# ============================================================== smoke
m = DialogMachineV9b()
x = torch.tensor([[U, MY, NAME, IS, N0 + 3, A, OK,
                   U, WHAT, IS, 7, PLUS, 5, A, 1, 2, U, OK, A, OK]])
with torch.no_grad():
    out, rl = m(x)
    mo = m._math_logits(x)
assert float(mo[0, 13].abs().sum()) > 0, "math fires at A of plus"
assert float(mo[0, 14].abs().sum()) > 0, "math fires at d1 (ones)"
assert float(mo[0, 15].abs().sum()) == 0, "math silent after answer"
# fwd must add math organ on host0 branch:
m.router[2].weight.data[:] = 0.0     # force task 0 to check branch addition
m.router[2].bias.data[:] = torch.tensor([10.0, 0.0, 0.0])
with torch.no_grad():
    out0, _ = m(x)
    st = m._state_logits(x)
    ho = torch.exp(m.head_gate)[0] * m.heads[0](m.norms[0](m.hosts[0](m.emb(x))))
    diff = float((out0[0, 13] - (ho[0, 13] + st[0, 13] + mo[0, 13])).abs().max())
assert diff < 1e-4, f"host0 branch must include both organs (diff {diff})"
print("[smoke-v9b] forward host0 = host + state organ + math organ OK", flush=True)
print("[smoke-v9b] PASSED", flush=True)

# ================================================================ train
torch.manual_seed(0)
m = DialogMachineV9b()
m.train()
organ_ids = {id(p) for p in (m.st_m, m.st_add, m.math_table)}
organ_p = [p for p in m.parameters() if id(p) in organ_ids]
rest_p = [p for p in m.parameters() if id(p) not in organ_ids]
opt = torch.optim.AdamW([{"params": rest_p},
                         {"params": organ_p, "weight_decay": 0.0}], lr=3e-3)
rng = random.Random(17)
t0 = time.time()
STEPS = 16000
for step in range(1, STEPS + 1):
    x, y, o, task = gen_w(32, 63, rng)
    lm, rt = train_step(m, opt, x, y, task)
    if step % 2000 == 0:
        print(f"  [v9b] step {step}/{STEPS} lm {lm:.4f} rt {rt:.4f} "
              f"gates {[round(float(v), 2) for v in torch.exp(m.head_gate)]} "
              f"st_m_abs {float(m.st_m.abs().sum()):.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
torch.save(m.state_dict(), "c22r3.pt")
print(f"[v9b] trained in {time.time()-t0:.0f}s", flush=True)

# ================================================================== eval
res = {"state4096": probe_w(m, 0, 4096),
       "state16384": probe_w(m, 0, 16384),
       "mathplus4096": probe_w(m, 1, 4096, 1, op=PLUS),
       "mathminus4096": probe_w(m, 1, 4096, 1, op=MINUS),
       "chat4096": probe_w(m, 2, 4096),
       "overwrite4096": overwrite_probe(m, 4096),
       "routing": routing_acc(m),
       "head_gates": [round(float(v), 3) for v in torch.exp(m.head_gate)],
       "st_m_abs": round(float(m.st_m.abs().sum()), 1)}
print(f"[eval-v9b] {res}", flush=True)
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
final = {"tag": "ARC2-C22R3-REPAIR",
         "changes": ["organ wd=0", "math turns in state family (MIX +18/118)",
                     "host0 branch gets math organ too", "16k steps"],
         "v9b": res, "bars": bars, "D7_detail": D7,
         "wall_s": round(time.time() - T0, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
