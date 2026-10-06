"""
ARC-2 CYCLE 43 / REASONING FRONTIER probe 2 — REVERSAL via VET+S (LIFO)
=======================================================================
C42 closed the single-head LTR tape class for reversal PROVABLY
(L-TRANSPORT-DIRECTION). This cycle: the minimal class extension that
restores leftward transport — a MECHANISM-OWNED PERSISTENT LIFO STACK
channel: the machine-v6 stack organ ported into the tape class (same
organ pattern as VET's register: exact mechanism state + tiny control
table). Class VET+S = VET + stack.

Class definition (mechanism fixed by class, control searched):
  control Mealy: Ph[a,h] (h in 0..4, p0h=0), Eh[a,h] in
    {IDENT=0, BLK=1, COND_R=2}
  COND_R = conditional readout (mechanism-owned): writes BDIG0+r only
    if a pop FIRED at this cell, else identity. The control cannot
    know whether the pop fired — the mechanism owns the condition.
  mechanism channels:
    S : PERSISTENT LIFO stack of digits (exact push/pop)
    P : persistent push count
    c : per-pass count of MARKs seen (reset each pass)
    s : per-pass cell index after SEP (reset, starting at 0 at the
        first post-SEP cell; slot cells 0,1,2,3... -> SOURCE cell of
        slot i = s even, TARGET cell = s odd)   [smoke-caught off-by-one]
    f : per-pass pop-fired flag (reset)
    r : per-pass register (reset to BOT)
  push at (DIG, h==2): S.push(d), P += 1
  pop  at (BLK, h==2): fires iff (s odd) and (not f) and
                       (c <= P - 1) and (S non-empty);
                       r := S.pop(); f := 1

DERIVATION (hand, BEFORE running — cycle-42 protocol):
  tape = [MARK]*nd + [SEP] + [DIG_i, BLK_i]*nd + [PAD]
  pass 1:   state0 eats mark 0 (->state1); SEP -> state2 (scan);
            slot i: push d_i + clear source (P = i+1); at BLK_i
            (s odd): pop BLOCKED because c = nd > i = P-1 for all
            i <= nd-1. => all pushed, NOTHING emitted, 1 mark eaten.
            (LIFO top after pass 1 = d_{nd-1}, which belongs to
            tgt_0 — the leftmost target — already passed in this
            pass: the first legal emit must wait one full pass.)
  pass k (2..nd+1): eat mark k-1 (c = nd-k+1 by the slot region);
            source residues (s even) never pop; first empty target
            BLK_{k-1} (s odd, c = nd-k+1 <= nd-1 = P-1, |S| =
            nd-k+2 > 0) -> pop top = d_{nd-k} = d_{nd-1-(k-1)}
            = EXACTLY the reversal pairing; f := 1 blocks further
            pops this pass.
  pass nd+2: sources BLK (s even), targets filled BDIG, no marks
            -> identity -> fixpoint. HALT.
  => tgt_i := d_{nd-1-i}, sources cleared, 1 mark eaten per pass
     while marks remain, n = nd+2 UNIFORMLY (nd >= 1), trace
     [nd, ..., 0, 0], stack empty at end.
  BAR DEVIATION vs C40/C41: passes = nd+2, not nd+1 — intrinsic
  LIFO overhead: LIFO output order (right-to-left) is the reverse of
  head target order (left-to-right), so the push pass cannot emit.
  Banked as finding L-LIFO-OVERHEAD (cost of escaping
  L-TRANSPORT-DIRECTION).

PRIOR ART (searched before implementing, directive 4):
  - Pushdown automata = finite control + unbounded LIFO stack; the
    push-all-then-pop-all sequence is the canonical pushdown
    transducer reversal construction (classical automata theory).
  - In-place reversal: two-pointer O(n/2) swap; in-place rotation via
    Gries-Mills block swap, avg 1.85 moves/elem (arxiv 2601.00979) —
    the tape-ROTATION geometry is the queued backup, not adopted.
  - VET prior art (C40): arxiv 2410.14067; C42 echo: arxiv 2402.01032.

BARS (C26/C40/C41 acceptance; S5 adapted to the class, see above):
S1 in-dist nd=2..4 >= 498/500; S2 200/200 nd=16; S3 100/100 nd=32;
S4 100/100 nd=64 joint; S5 passes=nd+2 + one-mark trace (spot
nd=1,2,4,8,16,32,64); stretch exact nd=128/256/512.
ARM A = hand-derived control (capability cert of the geometry).
ARM B = blank-genome hill-climb (C41 protocol: 450s / 400k evals,
        plateau-3000 restarts) -> certify whatever search finds
        (L-DISCOVERABILITY-BY-CLASS test on the new class).
Estimate (stated before launch): ARM A < 30s; ARM B <= 450s + cert;
total wall < 15 min, < 1 GB peak, 1 thread.
Tag ARC2-C43-REVBIND-VETS.
"""
import json, os, random, resource, time

import torch

torch.set_num_threads(1)
T0 = time.time()

MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0, BDIG0, ALPH, NH, BOT = 3, 14, 24, 5, 10
IDENT, ACT_BLK, ACT_COND_R = 0, 1, 2
SCAN = 2


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


def run_vs(Ph, Eh, tape, cap):
    """VET+S machine: control (Ph,Eh) x mechanism {S,P} persistent +
    {c,s,f,r} per-pass. Returns (final_tape, passes, mark_trace)."""
    tr = [int(sum(1 for x in tape if x == MARK))]
    S = []                      # persistent LIFO stack
    P = 0                       # persistent push count
    for n in range(1, cap + 1):
        out, h, r = [], 0, BOT
        f, c, s = 0, 0, 0       # per-pass: pop-fired, mark count, post-SEP index
        seen_sep = False
        for a in tape:
            act = int(Eh[a, h])
            fired = False
            if act == ACT_COND_R and a == BLK and h == SCAN and (s % 2 == 1) \
                    and not f and c <= P - 1 and len(S) > 0:
                r = S.pop()
                f = 1
                fired = True
            if act == ACT_BLK:
                out.append(BLK)
            elif fired:
                out.append(BDIG0 + r)
            else:
                out.append(a)
            # mechanism channels
            if a == SEP:
                seen_sep = True
            elif seen_sep:
                s += 1
            if a == MARK:
                c += 1
            if DIG0 <= a < DIG0 + 10 and h == SCAN:
                S.append(a - DIG0)
                P += 1
            h = int(Ph[a, h])
        if out == tape:
            return out, n, tr
        tape = out
        tr.append(int(sum(1 for x in tape if x == MARK)))
    return tape, cap, tr


TRAIN = []
g0 = random.Random(4141)
for nd in (2, 3, 4):
    for _ in range(10):
        digs = gen_digits(nd, g0)
        TRAIN.append((nd, digs, make_tape(nd, digs)))


def check(nd, fin, digs):
    want = digs[::-1]
    return all(fin[src_pos(nd, i)] == BLK for i in range(nd)) and \
        all(fin[tgt_pos(nd, i)] == BDIG0 + want[i] for i in range(nd))


def fitness(Ph, Eh):
    fx = fs = fc = ft = 0.0
    for nd, digs, tape in TRAIN:
        want = digs[::-1]
        fin, n, tr = run_vs(Ph, Eh, tape, nd + 9)
        fx += check(nd, fin, digs)
        fs += sum(fin[tgt_pos(nd, i)] == BDIG0 + want[i]
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
        fin, n, tr = run_vs(Ph, Eh, tape, nd + 9)
        exact += check(nd, fin, digs)
        pass_ok &= (n == nd + 2)
        trace_ok &= tr[-1] == 0 and all(
            tr[i - 1] - tr[i] == 1 for i in range(1, len(tr))
            if tr[i - 1] > 0)
    return exact, reps, pass_ok, trace_ok


def hand_genome():
    """The derivation as a control table:
    state 0 = fresh pass (eats first MARK), 1 = mark eaten pre-SEP,
    2 = slot scan (push DIG / conditional pop at BLK), 3,4 = spare
    (absorbing identity)."""
    Ph = torch.zeros(ALPH, NH, dtype=torch.long)
    for a in range(ALPH):
        for h in range(NH):
            Ph[a, h] = h
    Ph[MARK, 0] = 1
    Ph[SEP, 0] = 2
    Ph[SEP, 1] = 2
    Ph[SEP, 2] = 2
    Eh = torch.zeros(ALPH, NH, dtype=torch.long)
    Eh[MARK, 0] = ACT_BLK
    Eh[DIG0:DIG0 + 10, 2] = ACT_BLK
    Eh[BLK, 2] = ACT_COND_R
    return Ph, Eh


if os.environ.get("SMOKE") == "1":
    Ph, Eh = hand_genome()
    print("[c43-smoke] hand-genome wiring check", flush=True)
    for nd in (2, 3, 4):
        g = random.Random(7)
        for rep in range(3):
            digs = gen_digits(nd, g)
            tape = make_tape(nd, digs)
            fin, n, tr = run_vs(Ph, Eh, tape, nd + 9)
            ok = check(nd, fin, digs)
            print(f"  nd={nd} rep={rep} digs={digs} exact={ok} "
                  f"passes={n} (want {nd + 2}) trace={tr}", flush=True)
            if not ok:
                print(f"    tape={tape}\n    fin ={fin}", flush=True)
    print("SMOKE-DONE", flush=True)
    raise SystemExit(0)


def run_bars(Ph, Eh, s1_seeds, s2_seeds):
    s1 = (certify(Ph, Eh, 2, 100, s1_seeds[0])[0] +
          certify(Ph, Eh, 3, 150, s1_seeds[1])[0] +
          certify(Ph, Eh, 4, 250, s1_seeds[2])[0])
    s2 = certify(Ph, Eh, 16, 200, s2_seeds[0])
    s3 = certify(Ph, Eh, 32, 100, s2_seeds[1])
    s4 = certify(Ph, Eh, 64, 100, s2_seeds[2])
    spot = []
    for nd in (1, 2, 4, 8, 16, 32, 64):
        g = random.Random(900 + nd)
        tape = make_tape(nd, gen_digits(nd, g))
        _, n, _ = run_vs(Ph, Eh, tape, nd + 9)
        spot.append((nd, n, n == nd + 2))
    stretch = []
    for i, nd in enumerate((128, 256, 512)):
        exact = 0
        g = random.Random(700 + nd)
        for _ in range(5):
            digs = gen_digits(nd, g)
            fin, n, _ = run_vs(Ph, Eh, make_tape(nd, digs), nd + 9)
            exact += check(nd, fin, digs)
        stretch.append((nd, exact, n))
    res = {
        "S1_indist": s1,
        "S2_n16": s2[0], "S3_n32": s3[0], "S4_n64": s4[0],
        "S2_passdev": s2[2], "S3_passdev": s3[2], "S4_passdev": s4[2],
        "S5_passes_spot": spot,
        "S5_all_exact": all(x[2] for x in spot),
        "trace_ok": bool(s2[3] and s3[3] and s4[3]),
        "stretch": stretch,
        "stretch_ok": all(x[1] == 5 for x in stretch),
    }
    res["bars"] = {
        "S1_ge_498of500": res["S1_indist"] >= 498,
        "S2_200of200": res["S2_n16"] == 200,
        "S3_100of100": res["S3_n32"] == 100,
        "S4_100of100": res["S4_n64"] == 100,
        "S5_passes_nd+1": res["S5_all_exact"],
        "S5_one_mark_trace": res["trace_ok"],
        "STRETCH_128/256/512": res["stretch_ok"],
    }
    res["ALL"] = all(res["bars"].values())
    return res


# ------------------------------------------------------------ ARM A
PhA, EhA = hand_genome()
print("[c43] ARM A: hand-derived VET+S control (capability cert)",
      flush=True)
resA = run_bars(PhA, EhA, (601, 602, 603), (604, 605, 606))
print(f"[c43-A] S1 {resA['S1_indist']}/500 | S2 {resA['S2_n16']}/200 | "
      f"S3 {resA['S3_n32']}/100 | S4 {resA['S4_n64']}/100 | "
      f"S5 {resA['S5_all_exact']} trace {resA['trace_ok']} | "
      f"stretch {resA['stretch']}", flush=True)
print(f"[c43-A] bars {resA['bars']} ALL={resA['ALL']} "
      f"({time.time() - T0:.0f}s)", flush=True)

# ------------------------------------------------------------ ARM B
Ph = torch.arange(NH).unsqueeze(0).expand(ALPH, NH).contiguous().clone()
Eh = torch.zeros(ALPH, NH, dtype=torch.long)
rng = random.Random(43)
best_f = fitness(Ph, Eh)
best_Ph, best_Eh = Ph.clone(), Eh.clone()
print(f"[c43-B] blank-genome fitness {best_f:.4f}", flush=True)
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
        plateau = 0 if f > best_f else plateau + 1
        best_f, best_Ph, best_Eh = f, cand_Ph, cand_Eh
        if evals % 2000 < 2:
            print(f"  [c43-B] evals {evals} best {best_f:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if best_f >= 1.0:
            print(f"[c43-B] PERFECT train fitness at evals {evals}",
                  flush=True)
            break
    else:
        plateau += 1
    if plateau > 3000:
        for _ in range(4):
            a, h = rng.randrange(ALPH), rng.randrange(NH)
            best_Ph[a, h] = rng.randrange(NH)
            best_Eh[a, h] = rng.randrange(3)
        best_f = fitness(best_Ph, best_Eh)
        plateau = 0
print(f"[c43-B] search done: evals {evals} best {best_f:.4f} "
      f"({time.time() - t0:.0f}s)", flush=True)
resB = run_bars(best_Ph, best_Eh, (621, 622, 623), (624, 625, 626))
resB["discovered"] = bool(best_f >= 1.0 and resB["ALL"])
print(f"[c43-B] S1 {resB['S1_indist']}/500 | S2 {resB['S2_n16']}/200 | "
      f"S3 {resB['S3_n32']}/100 | S4 {resB['S4_n64']}/100 | "
      f"S5 {resB['S5_all_exact']} trace {resB['trace_ok']} | "
      f"discovered={resB['discovered']}", flush=True)
print(f"[c43-B] bars {resB['bars']}", flush=True)
torch.save({"Ph": best_Ph, "Eh": best_Eh}, "c43_vets_searched.pt")

final = {
    "tag": "ARC2-C43-REVBIND-VETS",
    "class": "VET+S (control Mealy x mechanism persistent LIFO stack; "
             "C42 L-TRANSPORT-DIRECTION escape via LIFO geometry)",
    "armA_hand": {k: resA[k] for k in
                  ("S1_indist", "S2_n16", "S3_n32", "S4_n64",
                   "S5_all_exact", "trace_ok", "stretch_ok", "bars",
                   "ALL")},
    "armB_search": {"evals": evals, "train_fitness": round(best_f, 4),
                    "wall_s": round(time.time() - t0, 1),
                    "cert": {k: resB[k] for k in
                             ("S1_indist", "S2_n16", "S3_n32", "S4_n64",
                              "S5_all_exact", "trace_ok", "stretch_ok",
                              "bars", "ALL", "discovered")}},
    "passes_bar": "n = nd+2 uniformly (nd>=1); L-LIFO-OVERHEAD vs "
                  "C40/C41 nd+1 — LIFO output order is the reverse of "
                  "head target order, push pass cannot emit",
    "wall_s": round(time.time() - T0, 1),
    "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
