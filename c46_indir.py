#!/usr/bin/env python3
"""C46 / reasoning-frontier probe 4 — INDIRECTION / NESTED BINDING on
the VET+S tape machine: capability classification by DERIVED state-
budget argument first (cycle-42 protocol), then hand construction
(existence), then staged discovery (C44/C45 method).

TASKS.
  1-HOP (array dereference):  T_i := V_{a_i}.
  2-HOP (nested binding):     T1_i := I_{a_i}, T2_i := V_{I_{a_i}}.
Value-agnostic: ONE control for EVERY digit assignment.

DERIVED THEORY (before any search — the C42 method).
  The control is a Mealy table over (symbol, state), 5 states; the
  value channels are value-OPaque to it:
    r (register): written by the mechanism, read only by the emit
      mechanism — the control never sees r.
    S (LIFO stack): values visible only as pop order.
    tape: the ONLY value-visible memory (DIG/BDIG/ADIG digits are
      distinct symbols the control CAN branch on).
  Mealy-on-original: the state transition at a cell uses the cell's
  PRE-WRITE symbol — a written value (BDIG) is invisible to the
  following transition.

  L-INDIRECTION-OPACITY: every address the control can branch on
  must be the symbol UNDER the head at transition time.
  L-INDIRECTION-REDUNDANCY: random access at the 5-state budget is
  paid in TAPE REDUNDANCY: the table is replicated once per reader
  (n^2 cells for 1-hop) so the per-digit state flow (branch at A_i
  -> address the V copy adjacent to A_i) works without data-
  dependent jumps.
  L-INDIRECTION-N4-LOCK (empirical, REVISED this cycle — the
  earlier derived mod-5 stride law is REFUTED): the hand control is
  certified realizable at n=4 and FAILS the full task at n=3,5,6,7,
  8,9 (fx=0.000 each, 40-tape sweep).  Derivation (refuted): the
  per-pass reader selection was believed to require the block
  offsets {0, L, ..., (n-1)L} distinct mod 5 (1-hop L = n+2 ->
  "excludes n == 3 mod 5").  Refutation: n=9 (L=11, L%5=1, the
  SAME class as working n=4's L=6) fails identically, so the
  stride is not the obstruction; and the L%5=0 cases (n=3,8) fail
  for ENTRY-ALIGNMENT reasons (all A-cells co-phase, but the
  mark-chain x SEP entry orbit never routes them to branch-state 3
  for those n) — co-phased blocks actually bind correctly WHEN the
  entry hits 3 (the per-digit ADIG rows disambiguate, no
  exclusivity needed).  The real lock is the coupling of the
  mark-clear chain duration, the SEP entry orbit, and the period-2
  selection cascade: it aligns at n=4 only.  General 1-hop
  realizability at n != 4 is OPEN (another entry family may unlock
  n=5..9).
  L-INDIRECTION-DEPTH-1: nesting depth >= 2 is UNREALIZABLE: the
  intermediate v = I_{a_i} must be re-exposed to the control to
  address the second table; at the T1->V boundary the V entry state
  is Ph[BLK, s] (data-independent) and the written BDIG_v is
  invisible (Mealy-on-original); the per-digit flow can carry only
  a (the index read at A_i), never v. Complete obstruction, no
  state-budget loophole.

HAND CONSTRUCTION (1-hop, n=4) — the C40-style existence proof.
  Layout: [MARK x4][SEP][A_i][V_0..V_3 (REPLICATED)][T_i] x i=0..3
  [PAD].  L = 6.  Mark orbit stride 2: (MARK,0)->2 clear, +2 cycle.
  Pass p entry e_p = 2(p-1)+1 (SEP row); reader i selected in the
  unique pass with e_p + i == 3 (selection = pass p: i = 2,0,3,1).
  Rows (state s in 0..4):
    ADIG_d: s==3 -> (3-d) mod 5   [per-digit branch, the ADDRESS]
            s !=3 -> 4            [absorb: the V chain starts at 4
                                    in non-selected passes -> V
                                    states {4,0,1,2} -> no RSET]
    DIG_d : s==3 -> 3, RSET(peek) [the V copy chain +1 mod 5; the
                                    RSET fires exactly at V_{a_i}]
            s !=3 -> (s+1) mod 5
    BLK   : Ph hold; Eh = REM at s in {0,1,2,4}, none at 3.
            r is NON-BOT at the cleared mark cells in most passes
            (the last branch block's peek survives) -> one spurious
            BDIG rewrite per mark cell (BDIG then inert) — the
            source of the 2n pass count (see below).
    BDIG_d: hold, no action (filled targets inert).
  Passes: 2n = 8 (measured): the selection is actually PERIOD-2
  (a branch block leaves its T at state 2-a, so the next block is
  re-selected two passes later, not one — the per-block flow
  branch -> peek -> emit is self-contained and idempotent, so all
  T_i := V_{a_i} land correctly by pass 2; REM fires only on BLK,
  so filled targets are inert).  The 2n count = n mark-clears +
  n-1 spurious REM cleanups: cleared mark cells stay BLK at state
  0 and r (from the last branch block's peek) is often non-BOT
  when they are visited -> one BDIG rewrite each (BDIG then inert).
  L-INDIRECTION-OVERHEAD (banked, cf. L-LIFO-OVERHEAD C43): the
  RSET-selectivity constraint h_a = 3-a forces T-state 2-a to hit
  0 for some a, so the mark cells share the REM-eligible state and
  the cleanup price is structural at the 5-state budget.
  A/V tables persist (PEEK, not consume: indirection reads leave
  the tables intact).

BARS (RESULT D):
  B_S1: C44 reversal genome under the EXTENDED mechanism
        (codes 3/4/5 unused by the genome -> fixed point): n=4..16.
  B_S2: 1-hop n=4 HAND control: 400 random (a, V) assignments,
        T exact + tables intact + halted + passes == 2n.
  B_S3: 1-hop n=4 DISCOVERED from blank (2-stage M1+Q pipeline,
        plateau walk, 3 seeds, ~24k evals = ARM-B class): the
        discoverability question on the new task.
  B_S4: 1-hop n=3 search (hand control fx=0.000 per the n-sweep):
        predicted plateau (the measured boundary).
  B_S5: 2-hop n=3 search (DERIVED UNREALIZABLE, depth-1 ceiling):
        predicted plateau (the measured theorem, C42-style shape).
  B_S6: 1-hop n=4 hand control, repeated indices a_i in {0,1}
        (peek robustness: the same V cell read by up to 4 readers).
PRIOR ART (searched 2026-08-26): NNPDA (arxiv 1711.05738 /
  UMD-CS-TR-3118) small-state controller + external memory;
  CS4820 Turing notes: recursion/pointer chains need an unbounded
  program-counter stack — finite-state sets cannot hold them
  (supports the depth-1 derived negative).  Gap: nobody ships a
  crisp small-state tape machine whose INDIRECTION capability is
  classified by state-budget counting (GAP-WE-ATTACK continues).
Tag ARC2-C46-INDIR. 1 thread.
"""
import itertools
import json
import os
import random
import sys
import time

T0 = time.time()
PEAK = 0.0

def _peak():
    global PEAK
    try:
        import resource
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        if mb > PEAK:
            PEAK = mb
    except Exception:
        pass

# ---------------- C46 encoding (34 symbols) ----------------
MARK, BLK, SEP = 0, 1, 2
DIG0 = 3            # V / I table values
PAD = 13
BDIG0 = 14          # filled targets
ADIG0 = 24          # index table A
ALPH = 44
NH = 5
BOT = 10
IDENT, ACT_BLK, ACT_COND_R, ACT_RSET, ACT_REM, ACT_CLR = 0, 1, 2, 3, 4, 5

def make_tape1(nd, a, V):
    """1-hop: [MARK^nd][SEP][A_i][V_0..V_{nd-1} replicated][T_i]...[PAD]
    L = nd+2 cells per block."""
    t = [MARK] * nd + [SEP]
    for i in range(nd):
        t.append(ADIG0 + a[i])
        t.extend(DIG0 + v for v in V)
        t.append(BLK)
    t.append(PAD)
    return t

def make_tape2(nd, a, I, V):
    """2-hop: [MARK^nd][SEP][A_i][I_0..I_{nd-1}][T1_i][V_0..V_{nd-1}][T2_i]...[PAD]
    L = 2*nd+2 cells per block."""
    t = [MARK] * nd + [SEP]
    for i in range(nd):
        t.append(ADIG0 + a[i])
        t.extend(DIG0 + v for v in I)
        t.append(BLK)   # T1_i
        t.extend(DIG0 + v for v in V)
        t.append(BLK)   # T2_i
    t.append(PAD)
    return t

def pos1(nd, i, what):
    """1-hop block [A_i][V_0..V_{nd-1}][T_i], L = nd+2."""
    base = nd + 1 + i * (nd + 2)
    if what == "A":
        return base
    if what == "T":
        return base + nd + 1
    return base + 1 + what  # V_j: offset 1+j

def t1_pos(nd, i):
    return nd + 1 + i * (nd + 2) + nd + 1

def tape2_pos(nd, i, what):
    L = 2 * nd + 2
    base = nd + 1 + i * L
    off = {"A": 0, "T1": nd + 1, "T2": 2 * nd + 1}
    return base + off[what]

def gen1(nd, r):
    a = [r.randrange(nd) for _ in range(nd)]
    V = [r.randrange(10) for _ in range(nd)]
    return a, V

def gen1rep(nd, r):
    """B_S6: repeated indices from {0,1}."""
    a = [r.randrange(2) for _ in range(nd)]
    V = [r.randrange(10) for _ in range(nd)]
    return a, V

def gen2(nd, r):
    a = [r.randrange(nd) for _ in range(nd)]
    I = [r.randrange(nd) for _ in range(nd)]
    V = [r.randrange(10) for _ in range(nd)]
    return a, I, V

# ---------------- mechanism: C45 + G3 (RSET peek) + G4 (REM) +
# ---------------- G5 (ACT_CLR) + register r ----------------
def step(t, h, S, P, r, Ph, Eh):
    """One pass. h = state at pass start (run passes 0). r = register.
    Returns (out, h_out, P_out, r_out, ident)."""
    out = list(t)
    f, c, s = 0, 0, 0
    seen_sep = False
    for i, a in enumerate(t):
        act = int(Eh[a][h])
        if act == ACT_COND_R and a == BLK and (s % 2 == 1) \
                and not f and c <= P - 1 and len(S) > 0:
            v = S.pop()
            f = 1
            out[i] = BDIG0 + v
        elif act == ACT_BLK:
            if DIG0 <= a < DIG0 + 10:
                S.append(a - DIG0)
                P += 1
            out[i] = BLK
        elif act == ACT_CLR:
            out[i] = BLK
        elif act == ACT_RSET:
            if DIG0 <= a < DIG0 + 10 or ADIG0 <= a < ADIG0 + 10:
                r = a - DIG0 if a < ADIG0 else a - ADIG0
        elif act == ACT_REM:
            if a == BLK and r != BOT:
                out[i] = BDIG0 + r
                r = BOT
        if a == SEP:
            seen_sep = True
        elif seen_sep:
            s += 1
        if a == MARK:
            c += 1
        h = int(Ph[a][h])
    return out, h, P, r, (out == t)

def run(Ph, Eh, tape, cap):
    tr = [int(tape.count(MARK))]
    t = list(tape)
    S, P, r = [], 0, BOT
    for m in range(1, cap + 1):
        t, h, P, r, ident = step(t, 0, S, P, r, Ph, Eh)
        tr.append(int(t.count(MARK)))
        if ident:
            return t, m, tr, True
    return t, cap, tr, False

def fx1(nd, fin, a, V):
    ok_t = all(fin[t1_pos(nd, i)] == BDIG0 + V[a[i]] for i in range(nd))
    ok_a = all(fin[pos1(nd, i, "A")] == ADIG0 + a[i] for i in range(nd))
    ok_v = True
    for i in range(nd):
        base = nd + 1 + i * (nd + 2) + 1
        for j in range(nd):
            if fin[base + j] != DIG0 + V[j]:
                ok_v = False
    return ok_t and ok_a and ok_v

def fx2(nd, fin, a, I, V):
    ok = True
    for i in range(nd):
        if fin[tape2_pos(nd, i, "T1")] != BDIG0 + I[a[i]]:
            ok = False
        if fin[tape2_pos(nd, i, "T2")] != BDIG0 + V[I[a[i]]]:
            ok = False
        if fin[tape2_pos(nd, i, "A")] != ADIG0 + a[i]:
            ok = False
    for reg in ("I", "V"):
        for i in range(nd):
            base = tape2_pos(nd, i, "A") + 1
            off = 0 if reg == "I" else nd + 2
            for j in range(nd):
                tbl = I if reg == "I" else V
                if fin[base + off + j] != DIG0 + tbl[j]:
                    ok = False
    return ok

def score_all(Ph, Eh, nd, taps, mode, cap=None):
    """Mean (fx, fs). taps = list of (tape, checker_args)."""
    cap = cap or 3 * nd + 8
    sx = sf = 0.0
    for tape, args in taps:
        fin, m, tr, halted = run(Ph, Eh, tape, cap)
        if mode == 1:
            a, V = args
            fx = 1.0 if (halted and fx1(nd, fin, a, V)) else 0.0
            fs = sum(1.0 for i in range(nd)
                     if fin[t1_pos(nd, i)] == BDIG0 + V[a[i]]) / nd
        else:
            a, I, V = args
            fx = 1.0 if (halted and fx2(nd, fin, a, I, V)) else 0.0
            fs = (sum(1.0 for i in range(nd)
                      if fin[tape2_pos(nd, i, "T1")] == BDIG0 + I[a[i]])
                  + sum(1.0 for i in range(nd)
                        if fin[tape2_pos(nd, i, "T2")]
                        == BDIG0 + V[I[a[i]]])) / (2 * nd)
        sx += fx
        sf += fs
    return sx / len(taps), sf / len(taps)

def trace_score(nd, tr):
    if not tr or tr[0] != nd:
        return 0.0
    c1 = nd - tr[1] if len(tr) > 1 else nd
    A = max(0.0, 1.0 - abs(c1 - 1))
    matches = 0
    for i, v in enumerate(tr[1:]):
        if v == max(nd - 1 - i, 0):
            matches += 1
        else:
            break
    B = min(matches, nd + 1) / (nd + 1)
    C = 1.0 if tr[-1] == 0 else 0.0
    return 0.4 * A + 0.4 * B + 0.2 * C

# ---------------- HAND control: 1-hop n=4 (derived) ----------------
def hand1():
    Ph = [[0] * NH for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    for a in range(ALPH):
        Ph[a] = list(range(NH))
    # MARK: +2 cycle, clear at 0
    Ph[MARK] = [2, 3, 4, 0, 1]
    Eh[MARK] = [ACT_CLR, 0, 0, 0, 0]
    # SEP: entries e1=1 (from 3), e2=3 (from 1), e3=0 (from 4),
    # e4=2 (from 2), e5=2 (from 0)
    Ph[SEP] = [2, 3, 2, 1, 0]
    # ADIG: branch at 3 -> (3-d) mod 5; absorb 4 otherwise
    for d in range(10):
        row = [4, 4, 4, (3 - d) % NH, 4]
        Ph[ADIG0 + d] = row
    # DIG (V copy): +1 cycle; RSET peek at 3 -> advance to 4
    # (a hold at 3 would re-peek every later V cell in the chain)
    for d in range(10):
        Ph[DIG0 + d] = [1, 2, 3, 4, 0]
        Eh[DIG0 + d] = [0, 0, 0, ACT_RSET, 0]
    # BLK: hold; REM at {0,1,2,4}
    Eh[BLK] = [ACT_REM, ACT_REM, ACT_REM, 0, ACT_REM]
    # BDIG / PAD: inert hold
    for d in range(10):
        Ph[BDIG0 + d] = list(range(NH))
    Ph[PAD] = list(range(NH))
    return Ph, Eh

# ---------------- search (C45 pipeline, full row space) ----------------
def blank_genome():
    Ph = [[h for h in range(NH)] for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    return Ph, Eh

def embed24(Ph24, Eh24):
    """24-symbol C43/C44 genome -> 44-symbol C46 (ADIG rows blank)."""
    Ph, Eh = [], []
    for a in range(24):
        Ph.append(list(Ph24[a]))
        Eh.append(list(Eh24[a]))
    for a in range(24, ALPH):
        Ph.append(list(range(NH)))
        Eh.append([0] * NH)
    return Ph, Eh

def mutate(Ph, Eh, a0s, rng, kmax=3):
    Ph2 = [row[:] for row in Ph]
    Eh2 = [row[:] for row in Eh]
    for _ in range(rng.randint(1, kmax + 1)):
        a0 = rng.choice(a0s)
        h = rng.randint(0, NH - 1)
        if rng.random() < 0.5:
            Ph2[a0][h] = (Ph2[a0][h] + rng.choice([-1, 1])) % NH
        else:
            Eh2[a0][h] = rng.randint(0, 5)
    if rng.random() < 0.4:
        a0 = rng.choice(a0s)
        h = rng.randint(0, NH - 1)
        Ph2[a0][h] = rng.randint(0, NH - 1)
        Eh2[a0][h] = rng.randint(0, 5)
    return Ph2, Eh2

def hill_climb(Ph, Eh, fit, a0s, budget_s, max_evals, rng, label,
               quiet=True, blank=None, stall_cap=400):
    """Hill-climb WITH PLATEAU WALK: equal-fitness candidates are
    accepted (p=0.5) so a zero-plateau search actually EXPLORES the
    space instead of degenerating to a star around the start
    (C46 forensics: the indirection needle has no partial-credit
    attractor; without plateau walking the 'negative' only covered
    the 1-3-entry star of the blank genome)."""
    best = fit(Ph, Eh)
    bestPh, bestEh = [r[:] for r in Ph], [r[:] for r in Eh]
    ev = 1
    t0 = time.time()
    stall = 0
    while time.time() - t0 < budget_s and ev < max_evals and stall < stall_cap:
        if blank is not None and rng.random() < 0.05:
            cPh, cEh = mutate(blank[0], blank[1], a0s, rng)
        else:
            cPh, cEh = mutate(bestPh, bestEh, a0s, rng)
        f = fit(cPh, cEh)
        ev += 1
        if f > best + 1e-12:
            best, bestPh, bestEh = f, cPh, cEh
            stall = 0
            if f >= 1.0 - 1e-9:
                break
        elif abs(f - best) <= 1e-12 and rng.random() < 0.5:
            bestPh, bestEh = cPh, cEh
            stall += 1
        else:
            stall += 1
    if not quiet:
        print(f"  [{label}] evals {ev} best {best:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return bestPh, bestEh, best, ev

ADIGS = list(range(ADIG0, ADIG0 + 10))
DIGS = list(range(DIG0, DIG0 + 10))
BDIGS = list(range(BDIG0, BDIG0 + 10))
ALLROWS = [SEP, BLK, PAD] + DIGS + BDIGS + ADIGS

def discover(fit1, fit2, T, seeds=(45, 46, 47), label="", budgets=None,
             q_stall=3000):
    """2-stage pipeline: M1 (MARK) -> Q (ALL non-MARK rows, the
    coupled indirection needle). C46 forensics: the RSET/REM/branch
    needle is INTRINSICALLY COUPLED (no fx/fs gradient exists until
    branch + RSET + REM coexist) -> the C44 staged decomposition is
    impossible here; the joint ~100-entry needle is attacked with
    the reset operator + 5% blank restarts x 3 seeds. T =
    (nd, taps, mode, cap)."""
    b = budgets or {"M1": (30, 2500), "Q": (240, 12000)}
    stages = [("M1", fit1, [MARK]), ("Q", fit2, ALLROWS[1:])]
    best = (-1.0, None, None, 0)
    for sd in seeds:
        rng = random.Random(sd)
        Ph, Eh = blank_genome()
        last = -1.0
        tot = 0
        for nm, fitf, a0s in stages:
            bs, me = b[nm]
            Ph, Eh, last, ev = hill_climb(Ph, Eh, lambda P, E, f=fitf: f(P, E),
                                          a0s, bs, me, rng,
                                          f"{label}-s{sd}-{nm}",
                                          blank=blank_genome(),
                                          stall_cap=400 if nm == "M1"
                                          else q_stall)
            tot += ev
        if last > best[0]:
            best = (last, Ph, Eh, tot)
    return best[1], best[2], best[0], best[3]

def taps1(nd, r, n=30, rep=False):
    gen = gen1rep if rep else gen1
    out = []
    for _ in range(n):
        a, V = gen(nd, r)
        out.append((make_tape1(nd, a, V), (a, V)))
    return out

def taps2(nd, r, n=30):
    out = []
    for _ in range(n):
        a, I, V = gen2(nd, r)
        out.append((make_tape2(nd, a, I, V), (a, I, V)))
    return out

def verify(Ph, Eh, nd, mode, r, n=60):
    taps = taps1(nd, r, n) if mode == 1 else taps2(nd, r, n)
    return score_all(Ph, Eh, nd, taps, mode)[0]

# ---------------- SMOKE ----------------
if os.environ.get("SMOKE"):
    import torch
    rng = random.Random(5)
    Ph, Eh = hand1()
    # 1) hand control 1-hop n=4: 200/200 exact + passes n+1
    ok, passes = 0, None
    for _ in range(200):
        a, V = gen1(4, rng)
        fin, m, tr, hal = run(Ph, Eh, make_tape1(4, a, V), 20)
        ok += hal and fx1(4, fin, a, V)
        passes = m
    assert ok == 200, f"hand1 n=4: {ok}/200"
    assert passes == 8, f"hand1 n=4 passes={passes}, want 2n=8"
    print(f"[c46-smoke] hand1 n=4: 200/200 exact, passes={passes} "
          f"(= 2n, L-INDIRECTION-OVERHEAD) OK", flush=True)
    # 2) repeated indices (peek robustness)
    ok = 0
    for _ in range(200):
        a, V = gen1rep(4, rng)
        fin, m, tr, hal = run(Ph, Eh, make_tape1(4, a, V), 20)
        ok += hal and fx1(4, fin, a, V)
    assert ok == 200, f"hand1 n=4 repeated: {ok}/200"
    print("[c46-smoke] hand1 n=4 repeated-index: 200/200 OK", flush=True)
    # 3) L-INDIRECTION-N4-LOCK (empirical, REVISED): the hand control
    #    is realizable IFF n == 4 over n = 3..9 (40-tape sweep each).
    #    The earlier derived mod-5 stride exclusion is REFUTED: n=9
    #    (L=11, L%5=1, same class as working n=4's L=6) still fails,
    #    while the L%5=0 cases (n=3,8) fail for entry-alignment
    #    reasons, not stride (co-phased blocks bind fine WHEN the
    #    entry routes them to branch-state 3 — ADIG disambiguates).
    sweep = {}
    for nd in (3, 4, 5, 6, 7, 8, 9):
        sweep[nd], _ = score_all(Ph, Eh, nd,
                                 taps1(nd, random.Random(777), 40), 1)
    assert sweep[4] == 1.0, f"hand1 n=4 fx={sweep[4]}"
    for nd in (3, 5, 6, 7, 8, 9):
        assert sweep[nd] < 1.0, f"hand1 n={nd} fx={sweep[nd]} (lock?)"
    sweepfmt = ", ".join(f"n{n}:{v:.3f}" for n, v in sweep.items())
    print(f"[c46-smoke] hand1 n-sweep 3..9: fx = {{{sweepfmt}}} "
          f"(realizable IFF n=4, N4-LOCK; mod-5 derivation REFUTED) "
          f"OK", flush=True)
    # 4) blank genome on 1-hop n=4: fx < 0.5
    fxn, _ = score_all(*blank_genome(), 4, taps1(4, random.Random(7), 20), 1)
    assert fxn < 0.5
    print(f"[c46-smoke] blank on 1-hop n=4: fx={fxn:.3f} OK", flush=True)
    # 5) C44 genome under the extended mechanism (24->44 embed):
    #    reversal regression n=4..16
    d = torch.load("c44_vets_discovered.pt", weights_only=False)
    PhE, EhE = embed24(d["Ph"].tolist(), d["Eh"].tolist())
    for nd in (4, 8, 16):
        ok = 0
        for _ in range(20):
            digs = [rng.randrange(10) for _ in range(nd - 1)] + \
                   [rng.randrange(1, 9)]
            mid = []
            for x in digs:
                mid += [DIG0 + x, BLK]
            tape = [MARK] * nd + [SEP] + mid + [PAD]
            fin, m, tr, hal = run(PhE, EhE, tape, 3 * nd + 8)
            ok += hal and all(fin[nd + 2 + 2 * i] == BDIG0 +
                              digs[nd - 1 - i] for i in range(nd)) \
                and all(fin[nd + 1 + 2 * i] == BLK for i in range(nd))
        assert ok == 20, f"C44 regression n={nd}: {ok}/20"
        print(f"[c46-smoke] C44 genome (ext. mechanism) n={nd}: "
              f"20/20 reversal OK", flush=True)
    print("SMOKE-DONE", flush=True)
    sys.exit(0)

# ---------------- main ----------------
def trace_mean(P, E, T):
    nd, taps, mode = T[0], T[1], T[2]
    acc = 0.0
    for tape, _ in taps:
        fin, m, tr, hal = run(P, E, tape, 3 * nd + 8)
        acc += trace_score(nd, tr)
    return acc / len(taps)

def main():
    import torch
    rng = random.Random(46)
    result = {"tag": "ARC2-C46-INDIR",
              "method": "derived state-budget theory (opacity / "
                        "redundancy / N4-lock / depth-1; the earlier "
                        "mod-5 stride derivation is REFUTED by the "
                        "n-sweep) + hand construction (1-hop n=4) + "
                        "2-stage discovery (M1 mark chain, Q joint "
                        "indirection needle, plateau walk) x 3 seeds "
                        "on C45 mechanism + RSET/REM/CLR + ADIG class",
              "D": {}}
    _peak()

    # ---- B_S1: C44 genome under the extended mechanism ----
    d = torch.load("c44_vets_discovered.pt", weights_only=False)
    PhE, EhE = embed24(d["Ph"].tolist(), d["Eh"].tolist())
    s1 = {}
    for nd in (4, 8, 16):
        fx = 0.0
        for _ in range(20):
            digs = [rng.randrange(10) for _ in range(nd - 1)] + \
                   [rng.randrange(1, 9)]
            mid = []
            for x in digs:
                mid += [DIG0 + x, BLK]
            tape = [MARK] * nd + [SEP] + mid + [PAD]
            fin, m, tr, hal = run(PhE, EhE, tape, 3 * nd + 8)
            ok = hal and all(fin[nd + 2 + 2 * i] == BDIG0 +
                             digs[nd - 1 - i] for i in range(nd)) \
                and all(fin[nd + 1 + 2 * i] == BLK for i in range(nd))
            fx += 1.0 if ok else 0.0
        s1[f"n{nd}"] = round(fx / 20, 3)
    result["D"]["B_S1_c44_regression"] = s1
    print(f"[c46-B_S1] C44 genome (extended mechanism): {s1}", flush=True)

    # ---- B_S2: hand control 1-hop n=4 ----
    PhH, EhH = hand1()
    fx_h, fs_h = score_all(PhH, EhH, 4, taps1(4, random.Random(202), 400), 1)
    fin, m, tr, hal = run(PhH, EhH, make_tape1(4, [2, 0, 3, 1],
                                              [5, 9, 2, 7]), 20)
    result["D"]["B_S2_hand1hop_n4"] = {
        "fx": round(fx_h, 4), "fs": round(fs_h, 4),
        "passes_witness": m, "formula": "2n = 8 (L-INDIRECTION-OVERHEAD)"}
    print(f"[c46-B_S2] hand 1-hop n=4: fx={fx_h:.4f} "
          f"fs={fs_h:.4f} passes={m}", flush=True)

    # ---- B_S6: hand control, repeated indices ----
    fx_r = score_all(PhH, EhH, 4, taps1(4, random.Random(203), 400,
                                        rep=True), 1)[0]
    result["D"]["B_S6_hand1hop_repeated"] = {"fx": round(fx_r, 4)}
    print(f"[c46-B_S6] hand 1-hop n=4 repeated-index: fx={fx_r:.4f}",
          flush=True)

    # ---- B_S3: 1-hop n=4 discovered ----
    T3 = (4, taps1(4, random.Random(460), 30), 1)
    def fit3a(P, E):
        c = score_all(P, E, 4, T3[1], 1)
        return 0.3 * c[1] + 0.1 * trace_mean(P, E, T3)
    def fit3b(P, E):
        c = score_all(P, E, 4, T3[1], 1)
        return 0.7 * c[1] + 0.3 * c[0]
    Ph3, Eh3, best3, ev3 = discover(fit3a, fit3b, T3, label="S3",
                                    budgets={"M1": (30, 2500),
                                             "Q": (400, 30000)},
                                    q_stall=8000)
    ver3 = verify(Ph3, Eh3, 4, 1, random.Random(703), 60)
    result["D"]["B_S3_1hop_n4_discovered"] = {
        "best": round(best3, 4), "evals": ev3,
        "verified": round(ver3, 3), "discovered": best3 >= 0.999}
    print(f"[c46-B_S3] 1-hop n=4 discovered: best={best3:.4f} "
          f"ev={ev3} ver={ver3:.3f}", flush=True)

    # ---- B_S4: 1-hop n=3 (constructed family fails per the n-sweep;
    #    no control found in budget; realizability formally OPEN —
    #    the mod-5 non-existence proof was REFUTED, so this is a
    #    budget+family negative, not a theorem) ----
    T4 = (3, taps1(3, random.Random(461), 30), 1)
    def fit4a(P, E):
        c = score_all(P, E, 3, T4[1], 1)
        return 0.3 * c[1] + 0.1 * trace_mean(P, E, T4)
    def fit4b(P, E):
        c = score_all(P, E, 3, T4[1], 1)
        return 0.7 * c[1] + 0.3 * c[0]
    Ph4, Eh4, best4, ev4 = discover(fit4a, fit4b, T4, label="S4",
                                    budgets={"M1": (30, 2500),
                                             "Q": (120, 8000)})
    ver4 = verify(Ph4, Eh4, 3, 1, random.Random(704), 60)
    result["D"]["B_S4_1hop_n3_nocontrol"] = {
        "best": round(best4, 4), "evals": ev4,
        "verified": round(ver4, 3),
        "prediction": "constructed family fails (hand fx=0.000, n-sweep); no control found in budget (plateau, ver=0.000); realizability at n=3 formally OPEN (mod-5 non-existence proof REFUTED)"}
    print(f"[c46-B_S4] 1-hop n=3 (no control: family fx=0.000, "
          f"open): best={best4:.4f} ev={ev4} ver={ver4:.3f}", flush=True)

    # ---- B_S5: 2-hop n=3 (DERIVED UNREALIZABLE) ----
    T5 = (3, taps2(3, random.Random(462), 30), 2)
    def fit5a(P, E):
        c = score_all(P, E, 3, T5[1], 2)
        return 0.3 * c[1] + 0.1 * trace_mean(P, E, T5)
    def fit5b(P, E):
        c = score_all(P, E, 3, T5[1], 2)
        return 0.7 * c[1] + 0.3 * c[0]
    Ph5, Eh5, best5, ev5 = discover(fit5a, fit5b, T5, label="S5",
                                    budgets={"M1": (30, 2500),
                                             "Q": (120, 8000)})
    ver5 = verify(Ph5, Eh5, 3, 2, random.Random(705), 60)
    result["D"]["B_S5_2hop_n3_derived_unrealizable"] = {
        "best": round(best5, 4), "evals": ev5,
        "verified": round(ver5, 3),
        "prediction": "plateau (depth-1 ceiling, L-INDIRECTION-DEPTH-1)"}
    print(f"[c46-B_S5] 2-hop n=3 (derived unrealizable): "
          f"best={best5:.4f} ev={ev5} ver={ver5:.3f}", flush=True)

    result["ckpt"] = "c46_indir_discovered.pt"
    result["wall_s"] = round(time.time() - T0, 1)
    _peak()
    result["peak_mb"] = round(PEAK, 1)
    print("RESULT " + json.dumps(result, default=str), flush=True)
    torch.save({"S3": (Ph3, Eh3), "S4": (Ph4, Eh4), "S5": (Ph5, Eh5),
                "hand1": (PhH, EhH)}, "c46_indir_discovered.pt")
    with open("log.jsonl", "a") as fp:
        fp.write(json.dumps({"ts": int(time.time()), **result}) + "\n")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
