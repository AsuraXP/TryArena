"""
ARC-2 CYCLE 41 / C26-R discoverability: can SEARCH find a VET program?
=======================================================================
Cycle 40 proved the VET class breaks the binding wall (hand-derived
existence proof). P4 standard requires DISCOVERABILITY: run search from a
blank genome inside the VET class and certify whatever it finds on the
full bars.

Search space (VET class definition):
  - Ph: 24 x 5 control transitions (values in 0..4)
  - Eh: 24 x 5 output ACTIONS from {IDENT=0, BLK=1, BDIG_R=2} where
    BDIG_R = "emit BDIG0+r" (the register readout symbol)
  - register rule is MECHANISM (fixed by class definition, not searched):
    r := value at consume trigger (h in {A,B} at DIG), r := bottom after
    (BLK at h=C), else unchanged.   [A=0,B=1,C=2,D=3,E=4, p0h=A]
Start genome: Ph[a,h]=h (hold state), Eh=IDENT everywhere (blank tape
machine). Staged hill-climb: random 1-3 entry mutations, accept on
fitness non-decrease; fitness = 0.6*exact + 0.2*slot + 0.1*consume +
0.1*mark-trace over 30 train cases (nd in {2,3,4}). Timebox ~8 min, then
certify best genome on the full C26 bars. Wall budget <= 12 min.
Tag ARC2-C41-VETSEARCH.
"""
import json, random, resource, time

import torch

torch.set_num_threads(1)
T0 = time.time()

MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0, BDIG0, ALPH, NH, BOT = 3, 14, 24, 5, 10
A_, B_, C_, D_, E_ = 0, 1, 2, 3, 4
IDENT, ACT_BLK, ACT_BDIG_R = 0, 1, 2


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


def run_vet(Ph, Eh, tape, cap):
    tr = [sum(1 for x in tape if x == MARK)]
    for n in range(1, cap + 1):
        out, h, r = [], A_, BOT
        for a in tape:
            act = Eh[a, h]
            if act == ACT_BDIG_R and r == BOT:
                act = IDENT            # empty-register read = no-op (mechanism)
            out.append(a if act == IDENT else
                       (BLK if act == ACT_BLK else BDIG0 + r))
            # mechanism register rule: consume trigger = DIG at h in {A,B}
            if h in (A_, B_) and DIG0 <= a < DIG0 + 10:
                r = a - DIG0
            elif h == C_ and a == BLK:
                r = BOT
            h = Ph[a, h]
        if out == tape:
            return out, n, tr
        tape = out
        tr.append(sum(1 for x in tape if x == MARK))
    return tape, cap, tr


TRAIN = []
g0 = random.Random(4141)
for nd in (2, 3, 4):
    for _ in range(10):
        digs = gen_digits(nd, g0)
        TRAIN.append((nd, digs, make_tape(nd, digs)))


def fitness(Ph, Eh):
    fx = fs = fc = ft = 0.0
    for nd, digs, tape in TRAIN:
        fin, n, tr = run_vet(Ph, Eh, tape, nd + 9)
        fx += all(fin[src_pos(nd, i)] == BLK for i in range(nd)) and \
              all(fin[tgt_pos(nd, i)] == BDIG0 + digs[i] for i in range(nd))
        fs += sum(fin[tgt_pos(nd, i)] == BDIG0 + digs[i]
                  for i in range(nd)) / nd
        fc += sum(fin[src_pos(nd, i)] == BLK for i in range(nd)) / nd
        ok = tr[-1] == 0 and all(tr[i - 1] - tr[i] in (0, 1)
                                 for i in range(1, len(tr)))
        one = all(tr[i - 1] - tr[i] == 1 for i in range(1, len(tr))
                  if tr[i - 1] > 0)
        ft += 0.5 * ok + 0.5 * one
    N = len(TRAIN)
    return (0.6 * fx + 0.2 * fs + 0.1 * fc + 0.1 * ft) / N


def certify(Ph, Eh, nd, reps, g_seed):
    g = random.Random(g_seed)
    exact = 0
    trace_ok = True
    pass_ok = True
    for _ in range(reps):
        digs = gen_digits(nd, g)
        tape = make_tape(nd, digs)
        fin, n, tr = run_vet(Ph, Eh, tape, nd + 9)
        exact += all(fin[src_pos(nd, i)] == BLK for i in range(nd)) and \
                 all(fin[tgt_pos(nd, i)] == BDIG0 + digs[i] for i in range(nd))
        pass_ok &= (n == nd + 1)
        trace_ok &= tr[-1] == 0 and all(
            tr[i - 1] - tr[i] == 1 for i in range(1, len(tr)) if tr[i - 1] > 0)
    return exact, reps, pass_ok, trace_ok


# ------------------------------------------------------------- search
Ph = torch.arange(NH).unsqueeze(0).expand(ALPH, NH).contiguous().clone()
Eh = torch.zeros(ALPH, NH, dtype=torch.long)
rng = random.Random(41)
best_f = fitness(Ph, Eh)
best_Ph, best_Eh = Ph.clone(), Eh.clone()
print(f"[c41] blank-genome fitness {best_f:.4f}", flush=True)

t0 = time.time()
evals = 0
plateau = 0
while time.time() - t0 < 450 and evals < 400_000:
    cand_Ph, cand_Eh = best_Ph.clone(), best_Eh.clone()
    for _ in range(rng.choice([1, 1, 2, 3])):
        a, h = rng.randrange(ALPH), rng.randrange(NH)
        if rng.random() < 0.5:
            cand_Ph[a, h] = rng.randrange(NH)
        else:
            cand_Eh[a, h] = rng.randrange(3)
    f = fitness(cand_Ph, cand_Eh)
    evals += 1
    if f >= best_f:
        if f > best_f:
            plateau = 0
        else:
            plateau += 1
        best_f, best_Ph, best_Eh = f, cand_Ph, cand_Eh
        if evals % 2000 < 2:
            print(f"  [c41] evals {evals} best {best_f:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if best_f >= 1.0:
            print(f"[c41] PERFECT train fitness at evals {evals}", flush=True)
            break
    else:
        plateau += 1
    if plateau > 3000:                     # restart perturbation
        for _ in range(4):
            a, h = rng.randrange(ALPH), rng.randrange(NH)
            best_Ph[a, h] = rng.randrange(NH)
            best_Eh[a, h] = rng.randrange(3)
        best_f = fitness(best_Ph, best_Eh)
        plateau = 0
print(f"[c41] search done: evals {evals} best {best_f:.4f} "
      f"({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------ certify
res = {}
res["S1_indist"] = certify(best_Ph, best_Eh, 2, 100, 501)[0] + \
                   certify(best_Ph, best_Eh, 3, 150, 502)[0] + \
                   certify(best_Ph, best_Eh, 4, 250, 503)[0]
s2 = certify(best_Ph, best_Eh, 16, 200, 504)
s3 = certify(best_Ph, best_Eh, 32, 100, 505)
s4 = certify(best_Ph, best_Eh, 64, 100, 506)
res["S2_n16"], res["S3_n32"], res["S4_n64"] = s2[0], s3[0], s4[0]
res["S5_passes_exact"] = bool(s2[2] and s3[2] and s4[2])
res["trace_ok"] = bool(s2[3] and s3[3] and s4[3])
bars = {"S1_ge_498of500": res["S1_indist"] >= 498,
        "S2_200of200": res["S2_n16"] == 200,
        "S3_100of100": res["S3_n32"] == 100,
        "S4_100of100": res["S4_n64"] == 100,
        "S5_passes_nd+1": res["S5_passes_exact"],
        "S5_one_mark_trace": res["trace_ok"]}
res["bars"] = bars
res["ALL"] = all(bars.values())
res["discovered"] = bool(res["ALL"])
torch.save({"Ph": best_Ph, "Eh": best_Eh}, "c41_vet_searched.pt")
print(f"[c41] S1 {res['S1_indist']}/500 S2 {res['S2_n16']}/200 "
      f"S3 {res['S3_n32']}/100 S4 {res['S4_n64']}/100 "
      f"S5 {res['S5_passes_exact']} trace {res['trace_ok']}", flush=True)
print(f"[bars] {bars}", flush=True)
final = {"tag": "ARC2-C41-VETSEARCH", "search": {
    "evals": evals, "train_fitness": round(best_f, 4),
    "wall_s": round(time.time() - t0, 1)}, "cert": res,
    "wall_s": round(time.time() - T0, 1),
    "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
