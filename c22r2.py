"""C22-R round 2: chatbot state-organ repair (machine v9).

Root causes found in v8 (dialog_chat_final.pt):
  R1 MECHANISM: state-organ query/features are emitted PRE-update, so the
     bilinear organ fires at the ANSWER-TOKEN position while probes/training
     score the answer at the A-marker position (off-by-one). Math organ likewise
     fires one position late. The organ literally cannot help where CE is scored.
  R2 EVAL (L-EVAL-FIDELITY): stream_probe oracle o marks the U-turn-start
     position 0.0, but the next-turn kind is iid-drawn from MIX_W, so the
     first-token entropy (~1.312 nats) is irreducible and un-subtracted.
     Decomposition of v8 fam0@4096 dCE 0.2297: U-pos contrib +0.2710,
     everything else -0.0413. D1 bar <=0.01 was unreachable for ANY model.

Fixes (this script):
  - DialogMachineV9: staged query state machine (arm at NAME/CODE token,
    fire at A position; code second digit fires at d1 position); emit uses
    current token (causal). Math organ emits starting at the A position.
  - Corrected probe oracle: U-position entropy -ln P_mix(first token) is
    included in o for fam0 streams (gen_dialogue_t_v / stream_probe_v).
  - Full retrain 12k steps with identical hyperparams (AdamW 3e-3, rng 17).

Wall budget: ~12 min (train ~6 min + eval). Tag ARC2-C22R2-REPAIR.
"""
import json, math, os, random, time, types

import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()

# ---- exec dialog_chat.py head (constants, data, DialogMachine, train_step, probes)
src = open("dialog_chat.py").read()
cut = src.index("# ----------------------------------------------------------------- training")
cut2 = src.index("@torch.no_grad()", cut)          # stream_probe starts the probe block
mod = types.ModuleType("dc_head")
exec(compile(src[:cut] + src[cut2:src.index("# ----------------------------------------------------------------- run")],
             "dialog_chat.py", "exec"), mod.__dict__)
g = mod.__dict__
# train_step sits in the training section between the two cut points
_ts = src.index("def train_step")
exec(compile(src[_ts:cut2], "dialog_chat.py", "exec"), g)
(U, A, MY, NAME, IS, NOW, CODE, WHAT, THE, OK, FINE, GOOD, TELL, ME, IT, AND,
 PLUS, MINUS, N0, VOCAB, MIX_W) = (g[k] for k in
    ["U", "A", "MY", "NAME", "IS", "NOW", "CODE", "WHAT", "THE", "OK", "FINE",
     "GOOD", "TELL", "ME", "IT", "AND", "PLUS", "MINUS", "N0", "VOCAB", "MIX_W"])
DialogMachine = g["DialogMachine"]
train_step = g["train_step"]
gen_dialogue_t = g["gen_dialogue_t"]
_build_state = g["_build_state"]
_emit = g["_emit"]
stream_probe = g["stream_probe"]
overwrite_probe = g["overwrite_probe"]
dialogue_gen = g["dialogue_gen"]
TMAP = g["TMAP"]
print(f"[c22r2] preamble loaded ({time.time()-T0:.0f}s)", flush=True)


# ============================================================= machine v9
class DialogMachineV9(DialogMachine):
    """v8 + fixed organ timing: queries fire at the scored answer positions."""

    def _state_logits(self, x, dbg=False):
        B, L = x.shape
        f = torch.zeros(B, L, 41)
        qo = torch.zeros(B, L, 3)
        idx = torch.full((B, L, 6), 38, dtype=torch.long)
        for b in range(B):
            name, d1v, d2v = -1, -1, -1
            hist = []
            qn = 0   # qname: 0 idle | 1 armed (fires while ==1) | 2 spent
            qc = 0   # qcode: 0 idle | 1 armed | 2 tens-fired | 3 ones-fired
            for t in range(L):
                tok = int(x[b, t])
                h = hist
                # ---- emit (registers from past; tok visible causally) ----
                if name >= 0:
                    f[b, t, name] = 1.0
                    idx[b, t, 0] = name
                    idx[b, t, 3] = 39
                if d1v >= 0:
                    f[b, t, 8 + d1v] = 1.0
                    idx[b, t, 1] = 8 + d1v
                    idx[b, t, 4] = 40
                qcode_fire = (qc == 1 and tok == A) or qc == 2
                if qcode_fire and d2v >= 0:
                    pos = 0 if qc == 1 else 1          # tens @ A, ones @ d1
                    f[b, t, 18 + d2v * 2 + pos] = 1.0
                    idx[b, t, 2] = 18 + d2v * 2 + pos
                qi = 1 if qn == 1 else (2 if qcode_fire else 0)
                qo[b, t, qi] = 1.0
                idx[b, t, 5] = 41 + qi
                # ---- process tok ----
                if N0 <= tok <= N0 + 7 and len(h) >= 3 and h[-1] == IS \
                        and h[-2] == NAME and h[-3] in (MY, NOW):
                    name = tok - N0
                elif tok < 10 and len(h) >= 4 and h[-1] < 10 \
                        and h[-2] == IS and h[-3] == CODE and h[-4] == MY:
                    d1v, d2v = h[-1], tok
                # arm queries one token BEFORE the A marker
                if tok == NAME and len(h) >= 3 and h[-1] == MY \
                        and h[-2] == IS and h[-3] == WHAT:
                    qn = 1
                elif qn == 1 and tok == A:
                    qn = 2                              # fired at this pos
                elif qn == 2:
                    qn = 0
                if tok == CODE and len(h) >= 3 and h[-1] == MY \
                        and h[-2] == IS and h[-3] == WHAT:
                    qc = 1
                elif qc == 1 and tok == A:
                    qc = 2
                elif qc == 2 and tok < 10:
                    qc = 3
                elif qc == 3:
                    qc = 0
                hist = (hist + [tok])[-4:]
        Aadd = (self.st_add[idx[:, :, 0]] + self.st_add[idx[:, :, 1]]
                + self.st_add[idx[:, :, 2]] + self.st_add[idx[:, :, 3]]
                + self.st_add[idx[:, :, 4]] + self.st_add[idx[:, :, 5]])
        if dbg:
            return Aadd, f, qo, idx
        return Aadd + torch.einsum("blm,bln,mnv->blv", f, qo, self.st_m)

    def _math_logits(self, x):
        B, L = x.shape
        out = torch.zeros(B, L, VOCAB, device=x.device)
        T = self.math_table
        for b in range(B):
            hist = []
            ma, mop = -1, -1
            mact, mp, mpend = False, 0, []
            ma_s = mb_s = 0
            mplus = False
            for t in range(L):
                tok = int(x[b, t])
                # fire at the A marker (first answer digit) and onwards
                if mact and mp < len(mpend):
                    case = mp if mplus else 2
                    out[b, t, :10] = T[case, ma_s, mb_s]
                    mp += 1
                    if mp >= len(mpend):
                        mact = False
                if tok < 10 and len(hist) >= 2 and hist[-1] == IS \
                        and hist[-2] == WHAT and mop < 0:
                    ma = tok
                elif tok in (PLUS, MINUS) and ma >= 0:
                    mop = tok
                    mplus = (tok == PLUS)
                elif tok < 10 and ma >= 0 and mop >= 0:
                    s = ma + tok if mop == PLUS else (ma - tok) % 10
                    ma_s, mb_s = ma, tok
                    mpend = [s // 10, s % 10] if mop == PLUS else [s]
                    mact, mp = True, 0
                    ma, mop = -1, -1
                hist = (hist + [tok])[-4:]
        return out


# =================================================== corrected probe oracle
P_FIRST = {g["WHAT"]: 0.36, MY: 0.19}        # qname+qcode=0.36 | fnow+fcode=0.19
for _w in (OK, FINE, GOOD, TELL):
    P_FIRST[_w] = 0.1125          # MIX_W = (45,18,18,10,9) iid turn kinds

def _emit_v(x, nll, u, a, ent_u, u_ent):
    assert len(ent_u) == len(u)
    x += [U] + u + [A] + a
    nll += [u_ent] + ent_u + [0.0] + [0.0] * len(a)

def _build_state_v(rng, length):
    """= _build_state, but U-position oracle = -ln P_mix(first token)."""
    _fname_turn, _fcode_turn = g["_fname_turn"], g["_fcode_turn"]
    _qname_turn, _qcode_turn = g["_qname_turn"], g["_qcode_turn"]
    LN2, LN8, LN10, H8 = g["LN2"], g["LN8"], g["LN10"], g["H8"]
    UM = -math.log(0.19)                       # fact turns start with MY
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
             + ["fnamenow"] * MIX_W[3] + ["fcode"] * MIX_W[4])
    while len(x) < length - 14:
        kind = rng.choice(kinds)
        if kind == "fill":
            u, a = g["_fill_turn"](rng)
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
        else:
            u, a = _fcode_turn(rng)
            code = [u[3], u[4]]
            ent = [H8, 0.0, 0.0, LN10, LN10]
        _emit_v(x, nll, u, a, ent, -math.log(P_FIRST[u[0]]))
    u, a = _qname_turn(name)
    _emit_v(x, nll, u, a, [0.0] * len(u), 0.0)       # forced final turns
    u, a = _qcode_turn(*code)
    _emit_v(x, nll, u, a, [0.0] * len(u), 0.0)
    assert len(x) >= length + 1
    return x, nll

def gen_dialogue_t_v(batch, length, rng, fam=None, op=None):
    xs, ys, os_, tasks = [], [], [], []
    for i in range(batch):
        f = fam if fam is not None else i % 3
        if f == 0:
            x, nll = _build_state_v(rng, length)
            xs.append(torch.tensor(x[:length])); ys.append(torch.tensor(x[1:length + 1]))
            os_.append(torch.tensor(nll[1:length + 1])); tasks.append(f)
        else:
            x, y, o, tk = gen_dialogue_t(1, length, rng, fam=f, op=op)
            xs.append(x[0]); ys.append(y[0]); os_.append(o[0]); tasks.append(f)
    return (torch.stack(xs), torch.stack(ys), torch.stack(os_),
            torch.tensor(tasks))

@torch.no_grad()
def stream_probe_v(model, fam, L, reps=1, op=None):
    """dCE with corrected oracle (fam0 U-position entropy included)."""
    model.eval()
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(900_000 + L + fam * 100 + (7 if op else 0) + i)
        x, y, o, _ = gen_dialogue_t_v(1, L, rng, fam=fam, op=op)
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
            x, y, o, task = gen_dialogue_t_v(1, 4096, rng, fam=f)
            _, rl = model(x)
            ok += int(rl.argmax(-1).item() == f); tot += 1
    return ok / tot


# ============================================================== smoke
print("[smoke-v9] wiring checks ...", flush=True)
m = DialogMachineV9()
x = torch.tensor([[U, MY, NAME, IS, N0 + 3, A, OK,
                   U, MY, CODE, IS, 4, 7, A, OK,
                   U, WHAT, IS, MY, NAME, A, N0 + 3,
                   U, WHAT, IS, MY, CODE, A, 4, 7,
                   U, OK, A, OK]])
# positions: qname A @20 (answer dave @21) | qcode A @27, d1 @28, d2 @29
_, f, qo, idx = m._state_logits(x, dbg=True)
assert float(f[0, 20, 3]) == 1.0, "name-j active at A pos of q-name"
assert float(qo[0, 20, 1]) == 1.0, "qname query fires AT the A position"
assert float(qo[0, 21, 1]) == 0.0, "qname query silent at answer token pos"
assert float(f[0, 27, 12]) == 1.0 and float(qo[0, 27, 2]) == 1.0, \
    "code tens at A pos"
assert float(f[0, 27, 18 + 7 * 2 + 0]) == 1.0, "d2joint(k=7,pos0) @A"
assert float(qo[0, 28, 2]) == 1.0 and \
    float(f[0, 28, 18 + 7 * 2 + 1]) == 1.0, "code ones at d1 pos"
assert float(qo[0, 29, :].sum()) == 1.0 and float(qo[0, 29, 0]) == 1.0, \
    "query spent after d2"
# bilinear push check at the scored position
with torch.no_grad():
    m.st_m[3, 1, N0 + 3] = 5.0
    lg = m._state_logits(x)
    assert float(lg[0, 20, N0 + 3]) >= 4.5, "bilinear pushes name at A pos"
    m.st_m[3, 1, N0 + 3] = 0.0
print("[smoke-v9] state-organ timing OK", flush=True)

xm = torch.tensor([[U, WHAT, IS, 7, PLUS, 5, A, 1, 2,
                    U, WHAT, IS, 3, MINUS, 7, A, 6, U]])
lm_ = m._math_logits(xm)
assert float(lm_[0, 6].abs().sum()) > 0, "math fires at A (plus tens)"
assert float(lm_[0, 7].abs().sum()) > 0, "math fires at d1 (plus ones)"
assert float(lm_[0, 8].abs().sum()) == 0, "math silent after plus answer"
assert float(lm_[0, 15].abs().sum()) > 0, "math fires at A (minus)"
assert float(lm_[0, 16].abs().sum()) == 0, "math silent after minus answer"
for p in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 17]:
    assert float(lm_[0, p].abs().sum()) == 0.0, f"no math logits at {p}"
print("[smoke-v9] math-organ timing OK", flush=True)

# oracle sanity: mean U-pos oracle in a 4096 fam0 stream ~ 1.31 * frac
x_, y_, o_, _ = gen_dialogue_t_v(1, 4096, random.Random(900_000 + 4096), fam=0)
print(f"[smoke-v9] oracle mean {float(o_.mean()):.4f} (expect ~0.22)", flush=True)
assert float(o_.mean()) > 0.1, "corrected oracle must carry U-pos entropy"
print("[smoke-v9] PASSED", flush=True)

# ============================================ legacy v8 on corrected probe
m8 = DialogMachine()
m8.load_state_dict(torch.load("dialog_chat_final.pt"))
v8_corr = {
    "state4096_v": stream_probe_v(m8, 0, 4096),
    "state4096_raw": stream_probe(m8, 0, 4096),
    "overwrite4096": overwrite_probe(m8, 4096),
}
print(f"[baseline] v8-final on corrected probe: {v8_corr}", flush=True)
del m8

# ================================================================ train v9
torch.manual_seed(0)
m = DialogMachineV9()
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
STEPS = 12000
for step in range(1, STEPS + 1):
    x, y, o, task = gen_dialogue_t(32, 63, rng)
    lm, rt = train_step(m, opt, x, y, task)
    if step % 2000 == 0:
        print(f"  [v9] step {step}/{STEPS} lm {lm:.4f} rt {rt:.4f} "
              f"gates {torch.exp(m.head_gate).tolist()} "
              f"st_m_abs {float(m.st_m.abs().sum()):.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
torch.save(m.state_dict(), "c22r2.pt")
print(f"[v9] trained in {time.time()-t0:.0f}s", flush=True)

# ================================================================== eval
res = {"state4096_v": stream_probe_v(m, 0, 4096),
       "state4096_raw": stream_probe(m, 0, 4096),
       "state16384_v": stream_probe_v(m, 0, 16384),
       "mathplus4096": stream_probe(m, 1, 4096, 1, op=PLUS),
       "mathminus4096": stream_probe(m, 1, 4096, 1, op=MINUS),
       "chat4096": stream_probe(m, 2, 4096),
       "overwrite4096": overwrite_probe(m, 4096),
       "routing": routing_acc(m),
       "head_gates": [round(float(v), 3) for v in torch.exp(m.head_gate)],
       "st_m_abs": round(float(m.st_m.abs().sum()), 1)}
print(f"[eval-v9] {res}", flush=True)
print("[dialogue]", flush=True)
print(dialogue_gen(m), flush=True)

bars = {"D1_state_v_le_0.01": res["state4096_v"] <= 0.01,
        "D2_overwrite_le_0.05": res["overwrite4096"] <= 0.05,
        "D3_16k_le_4k+0.05": res["state16384_v"] <= res["state4096_v"] + 0.05,
        "D4_mathplus_le_0.02": res["mathplus4096"] <= 0.02,
        "D4_mathminus_le_0.05": res["mathminus4096"] <= 0.05,
        "D5_chat_le_0.02": res["chat4096"] <= 0.02,
        "D6_routing_1.0": res["routing"] == 1.0}
print(f"[bars] {bars}", flush=True)
final = {"tag": "ARC2-C22R2-REPAIR",
         "root_cause": ["organ query/feature emit off-by-one (fires at "
                        "answer-token pos, scored at A pos)",
                        "probe oracle missed iid turn-kind entropy at U pos "
                        "(~1.312 nats; v8 U-pos contrib 0.271 of 0.230 dCE)"],
         "v8_on_corrected_probe": v8_corr,
         "v9": res, "bars": bars,
         "wall_s": round(time.time() - T0, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
