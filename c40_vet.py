"""
ARC-2 CYCLE 40 / C26-R re-entry: VALUE-ENCODED TRANSPORT (VET) machine.
=======================================================================
Binding wall (L-PLATEAU-ATTRACTOR): every discrete-table attack lands on
326/500. Mechanism diagnosis (this cycle): the program must jointly encode
(mark-discipline flag, digit-flag, carried value) in <=24 control states;
the value carry alone needs 10 sub-states and its needle is a joint
10-row configuration search keeps missing.

NEW MACHINE CLASS: control Mealy (small h) + MECHANISM-OWNED VALUE
REGISTER r (exact, r in {0..9, bottom}). Value is written to r at the
consume trigger and read at the next cell; control never carries value.
This is the organ pattern (exact mechanism state + tiny control table)
applied to the tape machine. Satisfies the C26 re-entry gate:
value-encoded transport (also: prior art arxiv 2410.14067 — complex/
register parameterizations express copy with linear resources where
fixed real state needs exponential).

This run = HAND-DERIVED existence proof for the class (same order as
P4-DISC: construct, certify, then test discoverability in a later cycle).

Control states: A=(no mark eaten, no digit eaten) B=(mark eaten)
                C=(carrying value in r) D=(both eaten)  E=(digit-only)
Mechanism register rule (NOT searched):
  r := v        at (DIG_v, h in {A,B})          [consume]
  r := bottom   at (BLK,   h == C)              [after emit]
Output (E(a,h,r)): (DIG_v,{A,B}) -> BLK; (BLK,C) -> BDIG0+r;
                   (MARK,A) -> BLK; else identity.
Ph: A-MA RK->B, {A,B}-DIG->C, C-BLK->D, rest absorb/idle. p0h=A.

BARS (C26 acceptance, unchanged): S1 in-dist nd<=4 >= 498/500; S2 200/200
nd=16; S3 100/100 nd=32; S4 100/100 nd=64 joint; S5 passes=nd+1 + one-mark
trace. Wall: <1 min. Tag ARC2-C40-VET.
"""
import json, random, resource, time

import torch

torch.set_num_threads(1)
T0 = time.time()

MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
BDIG0 = 14
A_ALPH = 24
BOT = 10                      # register bottom
# control states
A_, B_, C_, D_, E_ = 0, 1, 2, 3, 4
NH = 5
p0h = A_

# ---- control transition Ph[a, h]
Ph = torch.zeros(A_ALPH, NH, dtype=torch.long)
Ph[:, A_] = A_; Ph[:, B_] = B_; Ph[:, C_] = C_; Ph[:, D_] = D_; Ph[:, E_] = E_
Ph[MARK, A_] = B_                       # eat first mark
Ph[DIG0:DIG0 + 10, A_] = C_             # first digit -> carry
Ph[DIG0:DIG0 + 10, B_] = C_
Ph[BLK, C_] = D_                        # emit done

# ---- register update R(a, h) -> new r (mechanism, exact)
def reg_next(a, h, r):
    if h in (A_, B_) and DIG0 <= a < DIG0 + 10:
        return a - DIG0
    if h == C_ and a == BLK:
        return BOT
    return r

# ---- output E(a, h, r)
def emit(a, h, r):
    if h in (A_, B_) and DIG0 <= a < DIG0 + 10:
        return BLK                      # consume digit
    if h == C_ and a == BLK:
        return BDIG0 + r                # write value at tgt
    if h == A_ and a == MARK:
        return BLK                      # consume first mark
    return a                            # identity


def make_tape(nd, digs):
    mid = []
    for d in digs:
        mid += [DIG0 + d, BLK]
    return [MARK] * nd + [SEP] + mid + [PAD]


def gen_digits(nd, r):
    return [r.randrange(10) for _ in range(nd - 1)] + [r.randrange(1, 9)]


def src_pos(nd, i):
    return nd + 1 + 2 * i


def tgt_pos(nd, i):
    return nd + 2 + 2 * i


def run_vet(tape, cap):
    """iterated passes of the VET machine until fixpoint."""
    tr = [int(sum(1 for x in tape if x == MARK))]
    for n in range(1, cap + 1):
        out, h, r = [], p0h, BOT
        for a in tape:
            out.append(emit(a, h, r))
            r = reg_next(a, h, r)
            h = int(Ph[a, h])
        if out == tape:
            return out, n, tr
        tape = out
        tr.append(int(sum(1 for x in tape if x == MARK)))
    return tape, cap, tr


def certify(nd, reps, g_seed):
    g = random.Random(g_seed)
    exact = 0
    pass_dev = 0.0
    trace_ok = True
    for _ in range(reps):
        digs = gen_digits(nd, g)
        tape = make_tape(nd, digs)
        fin, n, tr = run_vet(tape, nd + 9)
        ok = all(fin[src_pos(nd, i)] == BLK for i in range(nd)) and \
             all(fin[tgt_pos(nd, i)] == BDIG0 + digs[i] for i in range(nd))
        exact += ok
        pass_dev += abs(n - (nd + 1))
        # one-mark-per-pass trace: marks decrease by exactly 1 each pass
        for i in range(1, len(tr)):
            if tr[i - 1] > 0 and tr[i - 1] - tr[i] != 1:
                trace_ok = False
        if tr[-1] != 0:
            trace_ok = False
    return exact, reps, pass_dev / reps, trace_ok


print("[c40] VET machine: control states=5, register=mechanism (exact), "
      "searched params=0 (hand-derived existence proof)", flush=True)
res = {}
res["S1_indist"] = certify(2, 100, 401)[0] + certify(3, 150, 402)[0] + \
                   certify(4, 250, 403)[0]
s2 = certify(16, 200, 404)
s3 = certify(32, 100, 405)
s4 = certify(64, 100, 406)
res["S2_n16"], res["S3_n32"], res["S4_n64"] = s2[0], s3[0], s4[0]
res["S2_passdev"], res["S3_passdev"], res["S4_passdev"] = (s2[2], s3[2], s4[2])
res["trace_ok"] = s2[3] and s3[3] and s4[3]
# S5: passes = nd+1 exact on spot checks
spot = []
for nd in (1, 2, 4, 8, 16, 32, 64):
    g = random.Random(900 + nd)
    tape = make_tape(nd, gen_digits(nd, g))
    _, n, tr = run_vet(tape, nd + 9)
    spot.append((nd, n, n == nd + 1))
res["S5_passes_spot"] = spot
res["S5_all_exact"] = all(s[2] for s in spot)

bars = {"S1_ge_498of500": res["S1_indist"] >= 498,
        "S2_200of200": res["S2_n16"] == 200,
        "S3_100of100": res["S3_n32"] == 100,
        "S4_100of100": res["S4_n64"] == 100,
        "S5_passes_nd+1": res["S5_all_exact"],
        "S5_one_mark_trace": res["trace_ok"]}
res["bars"] = bars
res["ALL"] = all(bars.values())
print(f"[c40] S1 {res['S1_indist']}/500 | S2 {res['S2_n16']}/200 | "
      f"S3 {res['S3_n32']}/100 | S4 {res['S4_n64']}/100 | "
      f"S5 passes-exact {res['S5_all_exact']} trace {res['trace_ok']}",
      flush=True)
print(f"[c40] passes spot: {spot}", flush=True)
print(f"[bars] {bars}", flush=True)
final = {"tag": "ARC2-C40-VET", "machine": "value-encoded transport "
         "(control Mealy x mechanism value register)",
         "res": res, "wall_s": round(time.time() - T0, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
