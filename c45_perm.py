#!/usr/bin/env python3
"""C45 / reasoning-frontier probe 3 — ARBITRARY PERMUTATIONS on the
LIFO geometry: mechanism-level CLASSIFICATION (exact, exhaustive) +
DISCOVERY (staged contract search, C44 method) + length-generalization
law.

TASK. Tape [MARK x n, SEP, d0,BLK,d1,BLK,...,d_{n-1},BLK, PAD]
(C43 encoding: MARK=0 BLK=1 SEP=2 DIG0=3 PAD=13 BDIG0=14). Goal:
tgt_i := d_{pi[i]} (target->source index map), value-agnostic (works
for ANY digit assignment).

MACHINE = C43/C44 VET+S mechanism (h resets to 0 EACH pass; the tape
is the persistent memory; the (symbol,state) control table is the
per-pass state program) with two CONTROL-GENERALIZED channels:
  (G1) push: DIG cell with Eh[DIG,h]==ACT_BLK -> push value, clear.
      (C43 hardcoded this to h==SCAN; the C44 genome sets Eh[DIG,2]=1
      and no other DIG push state -> identical behavior.)
  (G2) pop: BLK cell with Eh[BLK,h]==COND_R, s odd, not-fired,
      c <= P-1, |S|>0 -> r:=S.pop(), write BDIG0+r.
      (C43 hardcoded h==SCAN; the C44 genome sets Eh[BLK,2]=2 only.)
  Everything else verbatim C43: per-pass f (1 pop/pass), c (marks on
  tape at pass start), s (0-based post-SEP index, source i at s=2i
  even / target i at s=2i+1 odd), ACT_BLK writes BLK (MARK consume,
  SEP destroy, erase), fixpoint = identity pass => halt.
The C44 discovered reversal genome is a fixed point of (G1,G2) ->
S1 regression is exact by construction (verified).

SEARCH: value-agnostic quotient — mutations hit a CLASS canonical row
(MARK / SEP / DIG{3..12} / BLK / BDIG{14..23} / PAD) and copy to all
rows of the class; the task is value-agnostic so this is lossless.
Staged cumulative contracts (C44 lessons: graded, cumulative,
precondition-bearing, 2-5 entry needles):
  M1 (MARK x5): one-mark-per-pass graded trace.
  M2 (SEP x5):  0.5 trace + 0.5 scan  (scan invariant: every post-SEP
                slot cell in a non-mark state {1,2,3}, every pass
                incl. fixpoint — L-CONTRACT-PURITY guards).
  P  (DIG x5):  0.4 trace + 0.3 scan + 0.3(0.5 fc + 0.5 Pn)  (fc =
                sources cleared, Pn = all n pushed by end).
  Q  (BLK x5):  full  = 0.6 fx + 0.2 fs + 0.1(0.5 fc+0.5 Pn) + 0.1 ft
  R  (BDIG x5): full (post-pop routing refinement).

PART A (combinatorics, exact upper bound): DFS over pass sequences
over {0..n-1}: each pass either waits (empty: c = n-m+1 decays; pops
iff the C43 gate c <= P-1 is open) or pushes a nonempty tape-ordered
subset (left = {x <= filled} pushed before the pop cell; pop iff the
gate is open at P + |left|; right after). Pop at pass m fills the
leftmost unfilled target with the stack top. C45 forensics: waiting
passes are first-class — the first (consecutive-blocks-only) model
missed [2,0,1,3] (discovered by search anyway, exposing the bug).
The control-reachable set is a subset of the PART-A reachable set
(state-trajectory realizability is the extra constraint the search
+ state-budget argument settle empirically).

PRIOR ART (searched 2026-08-26):
  - Presortedness via contiguous monotone runs / Shuffled Monotone
    Sequences (ScienceDirect S0304397513007962): run-based measures;
    O(n) class detection by down-step counting — the standard family
    for our partition/run structure.
  - k-pop-stack sortable permutations (Elder et al., EJC 28(1) #54,
    2021): multi-pass stack machines have NO pattern-avoidance
    characterization -> we classify by exhaustive partition
    enumeration (exact) instead of pattern classes.
  - Shuffle/queue-stack sorting (ScienceDirect S0304397524002962,
    2024): single queue = identity only; single stack = 231-avoiding.
  - C43/C44: PDA prior art (emergentmind PDA; NNPDA/NSPDA staged
    learning; DTIC ADA120123 discrete hill-climb discovery).

BARS (RESULT D):
  A_n4: exhaustive reachable set over all 24 permutations of n=4 +
        min passes.  A_n5: same for n=5 (subset of 120).
  B_S1: C44 discovered genome under (G1,G2): reversal n=4..32 exact +
        passes == n+2 (one control for ALL n).
  B_S2: n=4 BATTERY — all 24 pi, staged discovery each; the headline
        bar: B_discovered == A_reachable for every pi (search agrees
        with combinatorics; the rest are certified negatives).
  B_S3a: n=8 head-front pi=[0,7,6,5,4,3,2,1] (the n=8 generalization
        of the n=4 [0,3,2,1] pattern; A: B1=[1..7],B2=[0], 10
        passes): length-specific discovery at n=8.
  B_S3b: n=8 two-block reversed swap pi=[3,2,1,0,7,6,5,4] (A: 11
        passes): predicted control-unrealizable (the pass-1 push
        pattern park-prefix/push-suffix needs a state switch the
        5-state control cannot place) -> the three-layer boundary
        (schedule-reachable > control-realizable > discovered).
  B_S4: one A-unreachable n=4 pi: search budget exhausted, < 1.0.
  B_S5: n=5 sample (reachable pi; A is complete at n=5).
  B_S1b: length generalization: the discovered n=4 [0,3,2,1]
        control and the discovered n=8 S3a control must implement
        the head-front family pi_n = [0, n-1, ..., 1] at n=8/16
        (ONE control for all n); negative control: reversal control
        on head-front pi_8 (task-specificity check).
  LAW: L-LIFO-COMPLETENESS (PART A: with wait-passes the schedule
       level is ALL of S_n — 24/24 at n=4, 120/120 at n=5, min
       passes <= n+3 observed) + L-STATE-BUDGET (the control level
       is a strict subset: the per-pass state trajectory is a
       composition of the 5-state (symbol,state) rows over the tape
       symbol pattern; parked-then-pushed sources need the
       trajectory to re-enter a push state, which the 5-state table
       cannot express for the boundary pi set — search plateau +
       state-count argument) + L-LIFO-UNIQUENESS (REFINED: the
       length-generalizing nontrivial families are exactly the
       n-uniform schedules — reversal ("push all in pass 1") and
       head-front ("park src0, push the rest"; switch at the first
       slot cell, symbol-nameable); every other pi needs a switch
       at a named mid-tape position -> length-specific controls).
Tag ARC2-C45-PERM. 1 thread.
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

# ---------------- C43 encoding ----------------
MARK, BLK, SEP = 0, 1, 2
DIG0, BDIG0, PAD, ALPH, NH, BOT = 3, 14, 13, 24, 5, 10
IDENT, ACT_BLK, ACT_COND_R = 0, 1, 2

def make_tape(nd, digs):
    mid = []
    for d in digs:
        mid += [DIG0 + d, BLK]
    return [MARK] * nd + [SEP] + mid + [PAD]

def src_pos(nd, i):
    return nd + 1 + 2 * i

def tgt_pos(nd, i):
    return nd + 2 + 2 * i

def gen_digits(nd, r):
    return [r.randrange(10) for _ in range(nd - 1)] + [r.randrange(1, 9)]

# ---------------- mechanism (C43 + G1/G2) ----------------
def step(t, h, S, P, Ph, Eh, scan=False):
    """One pass. h = state at pass start (run() passes 0). Returns
    (out, h_out, P_out, ident, (post, insc))."""
    out = list(t)
    f, c, s = 0, 0, 0
    seen_sep = False
    post = insc = 0
    for i, a in enumerate(t):
        act = int(Eh[a][h])
        fired = False
        if act == ACT_COND_R and a == BLK and (s % 2 == 1) \
                and not f and c <= P - 1 and len(S) > 0:
            v = S.pop()
            f = 1
            fired = True
            out[i] = BDIG0 + v
        elif act == ACT_BLK:
            if DIG0 <= a < DIG0 + 10:
                S.append(a - DIG0)
                P += 1
            out[i] = BLK
        # (else: identity write)
        if a == SEP:
            seen_sep = True
        elif seen_sep:
            if a != PAD:
                post += 1
                if h in (1, 2, 3):
                    insc += 1
            s += 1
        if a == MARK:
            c += 1
        h = int(Ph[a][h])
    return out, h, P, (out == t), (post, insc)

def run(Ph, Eh, tape, cap):
    """Returns (final_tape, passes, mark_trace, halted)."""
    tr = [int(tape.count(MARK))]
    t = list(tape)
    S, P = [], 0
    scans = []
    for m in range(1, cap + 1):
        t, h, P, ident, sc = step(t, 0, S, P, Ph, Eh, scan=True)
        scans.append(sc)
        if ident:
            tr.append(int(t.count(MARK)))
            return t, m, tr, True
        t2 = t
        tr.append(int(t2.count(MARK)))
        t = t2
    return t, cap, tr, False

def run_scan(Ph, Eh, tape, cap):
    """Same as run but also returns per-pass (post, insc) list and
    per-pass good flag (all slot cells in states 1..3)."""
    tr = [int(tape.count(MARK))]
    t = list(tape)
    S, P = [], 0
    scans = []
    for m in range(1, cap + 1):
        t, h, P, ident, sc = step(t, 0, S, P, Ph, Eh, scan=True)
        scans.append(sc)
        tr.append(int(t.count(MARK)))
        if ident:
            return t, m, tr, True, scans
    return t, cap, tr, False, scans

def trace_score(nd, tr):
    """One-mark-per-pass discipline, DECOMPOSED-GRADED (C44
    zero-gradient law, 2nd instance — a single all-or-nothing prefix
    still gave zero partial credit):
      A (0.4): pass 1 consumes exactly one mark (c1 = n - tr[1]).
      B (0.4): tr[1:] prefix matching the descent n-1..0,0,...
      C (0.2): mark-zero fixpoint reached (tr[-1] == 0).
    The full contract scores exactly 1.0."""
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

def fx_of(nd, fin, digs, pi):
    ok_t = all(fin[tgt_pos(nd, i)] == BDIG0 + digs[pi[i]]
               for i in range(nd))
    ok_s = all(fin[src_pos(nd, i)] == BLK for i in range(nd))
    return ok_t and ok_s

# ---------------- PART A: exhaustive partition upper bound ----------------
def ordered_partitions(n):
    def rec(remaining):
        if not remaining:
            yield ()
            return
        sm = sorted(remaining)
        for r in range(1, len(sm) + 1):
            for sub in itertools.combinations(sm, r):
                for tail in rec(tuple(x for x in sm if x not in sub)):
                    yield (sub,) + tail
    yield from rec(tuple(range(n)))

def partA(n, capM=None):
    """Returns {pi_tuple: min_passes} over the PART-A (schedule-level)
    reachable set. DFS over pass sequences: each pass either (a) is
    EMPTY (waits: consumes a mark, c decreases, pops iff the gate
    c <= P-1 is open) or (b) pushes a nonempty tape-ordered subset
    (left = pushed before the pop cell = {x <= filled}, pop iff gate
    open at P + |left|, right = pushed after). C45 forensics: the
    first version (consecutive nonempty blocks only) MISSED schedules
    like [2,0,1,3] which need empty wait-passes — the gate c <= P-1
    opens as c decays, so waiting is a first-class move. Passes are
    capped at 2n+2 (sufficient: first pop by pass n+1, then
    consecutive)."""
    if capM is None:
        capM = 2 * n + 2
    res = {}
    seen = set()

    def dfs(remaining, m, S, pops):
        key = (remaining, m, S, pops)
        if key in seen:
            return
        seen.add(key)
        if m > capM:
            return
        if not remaining:
            # all pushed: pops fire on consecutive passes (gate
            # c = n-mm+1 <= n-1 stays open for mm >= 2); evaluate
            # from the CURRENT pass m (fixpoint pass = m + 1 if the
            # last pop lands on m).
            Ss = list(S)
            pp = list(pops)
            mm = m
            while len(pp) < n and mm <= capM:
                c = n - (mm - 1)
                if Ss and c <= n - 1:
                    pp.append(Ss.pop())
                mm += 1
            if len(pp) == n:
                pi = tuple(pp)
                passes = mm  # = last-pop pass + 1 = identity pass
                if pi not in res or passes < res[pi]:
                    res[pi] = passes
            return
        c = n - (m - 1)
        P = n - len(remaining)
        # (a) empty pass (wait)
        if m < capM:
            Ss = list(S)
            if Ss and c <= P - 1:
                v = Ss.pop()
                dfs(remaining, m + 1, tuple(Ss), pops + (v,))
            else:
                dfs(remaining, m + 1, S, pops)
        # (b) push a nonempty subset
        sm = sorted(remaining)
        for r in range(1, len(sm) + 1):
            for sub in itertools.combinations(sm, r):
                left = [x for x in sub if x <= len(pops)]
                right = [x for x in sub if x > len(pops)]
                Ss = list(S) + left
                P2 = P + len(left)
                pp = list(pops)
                if Ss and c <= P2 - 1:
                    pp.append(Ss.pop())
                dfs(frozenset(sm) - frozenset(sub), m + 1,
                    tuple(Ss + right), tuple(pp))

    dfs(frozenset(range(n)), 1, (), ())
    return res

def partA_partition(n, parts):
    """Simulate one specific partition; return (pi, passes) or None."""
    S, P = [], 0
    filled = 0
    pops = []
    m = 0
    while filled < n and m < 2 * n + 4:
        m += 1
        c = n - (m - 1)
        Bm = parts[m - 1] if m - 1 < len(parts) else ()
        left = [x for x in Bm if x <= filled]
        right = [x for x in Bm if x > filled]
        S.extend(left)
        P += len(left)
        if S and c <= P - 1:
            pops.append(S.pop())
            filled += 1
        S.extend(right)
        P += len(right)
    if filled < n:
        return None
    return tuple(pops), m + 1

# ---------------- search (C44 staged method, value-agnostic) ----------------
def blank_genome():
    Ph = [[h for h in range(NH)] for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    return Ph, Eh

# class canonical: symbol a representing the class; DIG class = DIG0,
# BDIG class = BDIG0
DIGS = list(range(DIG0, DIG0 + 10))
BDIGS = list(range(BDIG0, BDIG0 + 10))

def apply_row(Ph, Eh, a0, h, pv, ev):
    if a0 == DIG0:
        for a in DIGS:
            Ph[a][h], Eh[a][h] = pv, ev
    elif a0 == BDIG0:
        for a in BDIGS:
            Ph[a][h], Eh[a][h] = pv, ev
    else:
        Ph[a0][h], Eh[a0][h] = pv, ev

def mutate(Ph, Eh, a0s, rng, kmax=3):
    """1-3 single-entry changes, PLUS (p=0.4) a full (row,h) reset to
    a uniform (Ph, Eh) pair — the reset operator directly generates
    the coupled 2-entry needles (e.g. (MARK,0)=(k,1)) in one step;
    C45 forensics: the 1-entry walk alone needs ~150-400 lucky evals
    to assemble them (80-eval stalls starved every stage)."""
    Ph2 = [row[:] for row in Ph]
    Eh2 = [row[:] for row in Eh]
    for _ in range(rng.randint(1, kmax + 1)):
        a0 = rng.choice(a0s)
        h = rng.randint(0, NH - 1)
        if rng.random() < 0.5:
            pv = (int(Ph[a0][h]) + rng.choice([-1, 1])) % NH
            apply_row(Ph2, Eh2, a0, h, pv, int(Eh[a0][h]))
        else:
            ev = rng.randint(0, 3)
            apply_row(Ph2, Eh2, a0, h, int(Ph[a0][h]), ev)
    if rng.random() < 0.4:
        a0 = rng.choice(a0s)
        h = rng.randint(0, NH - 1)
        apply_row(Ph2, Eh2, a0, h, rng.randint(0, NH - 1),
                  rng.randint(0, 2))
    return Ph2, Eh2

def score_all(Ph, Eh, nd, pi, taps, cap=None):
    """Mean component scores (trace, scan, fc, Pn, fx, fs). cap
    defaults to nd+12 (reversal family, n+2 passes); pass a larger
    cap for 2n+1-pass schedule families (head-front, C45 forensic:
    the n=16 0.0 in run 2 was a cap artifact, not a mechanism
    failure)."""
    cap = cap or nd + 12
    st = ss = sf = sp = sx = sf_ = 0.0
    for digs, tape in taps:
        fin, m, tr, halted, scans = run_scan(Ph, Eh, tape, cap)
        st += trace_score(nd, tr)
        good = 0
        for post, insc in scans:
            if post > 0 and insc == post:
                good += 1
        ss += good / len(scans)
        sf += sum(fin[src_pos(nd, i)] == BLK for i in range(nd)) / nd
        sp += 1.0 if sf > 0 and all(fin[src_pos(nd, i)] == BLK
                                    for i in range(nd)) else 0.0
        fx = 1.0 if (halted and fx_of(nd, fin, digs, pi)) else 0.0
        sx += fx
        sf_ += sum(fin[tgt_pos(nd, i)] == BDIG0 + digs[pi[i]]
                   for i in range(nd)) / nd
    N = len(taps)
    return tuple(x / N for x in (st, ss, sf, sp, sx, sf_))

def f_M1(Ph, Eh, T):
    return score_all(Ph, Eh, T[0], T[1], T[2])[0]

def f_M2(Ph, Eh, T):
    c = score_all(Ph, Eh, T[0], T[1], T[2])
    return 0.5 * c[0] + 0.5 * c[1]

def f_P(Ph, Eh, T):
    c = score_all(Ph, Eh, T[0], T[1], T[2])
    return 0.4 * c[0] + 0.3 * c[1] + 0.3 * (0.5 * c[2] + 0.5 * c[3])

def f_full(Ph, Eh, T):
    c = score_all(Ph, Eh, T[0], T[1], T[2])
    return 0.6 * c[4] + 0.2 * c[5] + 0.1 * (0.5 * c[2] + 0.5 * c[3]) \
        + 0.1 * c[0]

def f_Q1(Ph, Eh, T):
    """CONTRACT-Q1 (BLK rows, first): the FIRST pop must land the
    correct value on tgt0 (q_0) — direct graded signal for the pop
    state before the full-fitness plateau (C44 Sa->Sb sub-
    contraction lesson)."""
    nd, pi, taps = T[0], T[1], T[2]
    acc = 0.0
    for digs, tape in taps:
        fin, m, tr, halted, _ = run_scan(Ph, Eh, tape, nd + 12)
        acc += 1.0 if fin[tgt_pos(nd, 0)] == BDIG0 + digs[pi[0]] else 0.0
    c = score_all(Ph, Eh, nd, pi, taps)
    return 0.5 * (acc / len(taps)) + 0.3 * (0.5 * c[2] + 0.5 * c[3]) \
        + 0.2 * c[0]

def hill_climb(Ph, Eh, fit, a0s, budget_s, max_evals, rng, label, T=None,
               quiet=False, blank=None):
    best = fit(Ph, Eh, T)
    bestPh, bestEh = [r[:] for r in Ph], [r[:] for r in Eh]
    ev = 1
    t0 = time.time()
    stall = 0
    while time.time() - t0 < budget_s and ev < max_evals and stall < 400:
        if blank is not None and rng.random() < 0.05:
            # injected restart: fresh candidate from blank (diversity
            # against local optima in the coupled-needle landscape)
            cPh, cEh = mutate(blank[0], blank[1], a0s, rng)
        else:
            cPh, cEh = mutate(bestPh, bestEh, a0s, rng)
        f = fit(cPh, cEh, T)
        ev += 1
        if f > best + 1e-12:
            best, bestPh, bestEh = f, cPh, cEh
            stall = 0
            if f >= 1.0 - 1e-9:
                if not quiet:
                    print(f"  [{label}] PERFECT at evals {ev} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                break
            if not quiet and ev % 300 == 0:
                print(f"  [{label}] evals {ev} best {best:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        else:
            stall += 1
    if not quiet:
        print(f"  [{label}] evals {ev} best {best:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    return bestPh, bestEh, best, ev

def discover(Ph, Eh, T, rng, label, budgets=None, quiet=True):
    """Run the 5-stage pipeline. T = (nd, pi, taps)."""
    b = budgets or {"M1": (45, 3000), "M2": (45, 3000), "P": (45, 3000),
                    "Q1": (45, 3000), "Q2": (90, 6000), "R": (90, 6000)}
    stages = [("M1", f_M1, [MARK]), ("M2", f_M2, [SEP]),
              ("P", f_P, [DIG0]), ("Q1", f_Q1, [BLK]),
              ("Q2", f_full, [BLK]), ("R", f_full, [BDIG0])]
    tot_ev = 0
    last = 0.0
    blank = blank_genome()
    for nm, fitf, a0s in stages:
        bs, me = b[nm]
        Ph, Eh, last, ev = hill_climb(Ph, Eh, lambda P, E, T=T, ff=fitf:
                                      ff(P, E, T), a0s, bs, me, rng,
                                       f"{label}-{nm}", T, quiet=quiet,
                                       blank=blank)
        tot_ev += ev
    return Ph, Eh, last, tot_ev

def discover_multi(T, seeds=(45, 46, 47), label="", budgets=None):
    """3 independent seed-starts of the 6-stage pipeline; keep the
    best (C45 forensics: single-stream search is fragile on the
    coupled-needle landscape; 3 starts x 400-eval stalls give high-
    confidence negatives). Returns (Ph, Eh, bestf, total_evals)."""
    best = (-1.0, None, None, 0)
    for sd in seeds:
        rng = random.Random(sd)
        Ph, Eh, f, evs = discover(*blank_genome(), T, rng,
                                  f"{label}-s{sd}", budgets=budgets,
                                  quiet=True)
        if f > best[0]:
            best = (f, Ph, Eh, evs)
    return best[1], best[2], best[0], best[3]

def make_taps(nd, r, n=30):
    taps = []
    for _ in range(n):
        digs = gen_digits(nd, r)
        taps.append((digs, make_tape(nd, digs)))
    return taps

def verify(Ph, Eh, nd, pi, r, n=60):
    taps = []
    for _ in range(n):
        digs = gen_digits(nd, r)
        taps.append((digs, make_tape(nd, digs)))
    c = score_all(Ph, Eh, nd, pi, taps)
    return c[4]  # fx

# ---------------- SMOKE ----------------
if os.environ.get("SMOKE"):
    import torch
    d = torch.load("c44_vets_discovered.pt", weights_only=False)
    Ph44, Eh44 = d["Ph"].tolist(), d["Eh"].tolist()
    rng = random.Random(5)
    # 1) C44 genome under (G1,G2): reversal regression
    for nd in (4, 8, 16, 32):
        c = score_all(Ph44, Eh44, nd, list(reversed(range(nd))),
                      make_taps(nd, rng, 20))
        fin, m, tr, halted = run(Ph44, Eh44,
                                 make_tape(nd, gen_digits(nd, rng)), nd + 12)
        assert c[4] == 1.0, f"reversal n={nd} fx={c[4]}"
        assert m == nd + 2, f"reversal n={nd} passes={m} want {nd+2}"
        print(f"[c45-smoke] reversal n={nd}: fx=1.0 passes={m} "
              f"(=n+2) OK", flush=True)
    # 2) partA sanity: reversal in A4 with n+2 passes; the
    #    gap-schedule [2,0,1,3] (hand-verified, 7 passes) must be
    #    in A4 (the consecutive-blocks-only version missed it)
    t0 = time.time()
    A4 = partA(4)
    A5t = partA(5)
    assert A4[(3, 2, 1, 0)] == 6, A4[(3, 2, 1, 0)]
    assert (0, 3, 2, 1) in A4
    assert (2, 0, 1, 3) in A4, "gap schedule [2,0,1,3] missing"
    print(f"[c45-smoke] partA(4): {len(A4)}/24 reachable, "
          f"partA(5): {len(A5t)}/120, reversal passes=6, "
          f"[2,0,1,3] in A4 OK ({time.time()-t0:.1f}s)", flush=True)
    # 3) blank genome: identity task (pi = identity) -> fx = 1.0
    #    (no writes needed; halt at pass 1)
    c = score_all(blank_genome()[0], blank_genome()[1], 4, [0, 1, 2, 3],
                  make_taps(4, rng, 5))
    print(f"[c45-smoke] blank on identity: fx={c[4]:.3f}", flush=True)
    # 4) mini M1 from blank reaches trace 1.0
    T = (4, [3, 2, 1, 0], make_taps(4, rng))
    Ph, Eh, p, e = hill_climb(*blank_genome(),
                              lambda P, E, T=T: f_M1(P, E, T), [MARK],
                              60, 4000, rng, "smoke-M1", T, quiet=True)
    print(f"[c45-smoke] mini M1: {p:.3f} after {e} evals (must be 1.0)",
          flush=True)
    assert p == 1.0
    print("SMOKE-DONE", flush=True)
    sys.exit(0)

# ---------------- main ----------------
def main():
    import torch
    rng = random.Random(45)
    result = {"tag": "ARC2-C45-PERM",
              "method": "PART A: exhaustive ordered-partition LIFO "
                        "simulation (schedule-level upper bound); "
                        "PART B: staged contract-decomposed hill-climb "
                        "over value-agnostic class-quotient control, "
                        "C43 mechanism + G1/G2 (C44 method)",
              "D": {}}
    _peak()

    # ---- PART A ----
    t0 = time.time()
    A4 = partA(4)
    A5 = partA(5)
    print(f"[c45-A] n=4: {len(A4)}/24 reachable; n=5: {len(A5)}/120 "
          f"({time.time()-t0:.1f}s)", flush=True)
    result["D"]["A_n4"] = {"reachable": len(A4), "of": 24,
                           "set": {str(list(k)): v
                                   for k, v in sorted(A4.items())}}
    result["D"]["A_n5"] = {"reachable": len(A5), "of": 120}
    result["D"]["A_n5_set"] = {str(list(k)): v for k, v in A5.items()}

    # ---- B_S1: reversal regression (C44 genome, one control for all n)
    d = torch.load("c44_vets_discovered.pt", weights_only=False)
    PhR, EhR = d["Ph"].tolist(), d["Eh"].tolist()
    s1 = {}
    for nd in (4, 8, 16, 32):
        c = score_all(PhR, EhR, nd, list(reversed(range(nd))),
                      make_taps(nd, rng, 40))
        s1[f"n{nd}"] = round(c[4], 3)
    print(f"[c45-B_S1] reversal regression (C44 genome): {s1}", flush=True)
    result["D"]["B_S1_reversal"] = s1

    # ---- B_S2: n=4 battery, all 24 pi ----
    allp4 = [list(p) for p in itertools.permutations(range(4))]
    battery = {}
    found = {}
    nmatch = nreach = 0
    t0 = time.time()
    for pi in allp4:
        reach = tuple(pi) in A4
        T = (4, pi, make_taps(4, random.Random(450 + sum(pi * 7)), 30))
        Ph, Eh, bestf, evs = discover_multi(T, label=f"S2-{pi}")
        disc = bestf >= 0.999
        ver = verify(Ph, Eh, 4, pi, random.Random(900 + sum(pi * 7)), 60)
        ver_ok = ver >= 0.999
        nreach += int(reach)
        nmatch += int(reach == disc)
        if ver_ok:
            found[str(pi)] = (Ph, Eh)
        battery[str(pi)] = {"A_reachable": reach, "B_discovered": disc,
                            "verified": ver_ok, "best_full": round(bestf, 4),
                            "evals": evs,
                            "A_passes": A4.get(tuple(pi))}
        print(f"[c45-B_S2] pi={pi} A={int(reach)} B={int(disc)} "
              f"ver={int(ver_ok)} best={bestf:.3f} ev={evs} "
              f"({time.time()-t0:.0f}s)", flush=True)
    result["D"]["B_S2_n4"] = battery
    result["D"]["B_S2_agree"] = f"{nmatch}/24"
    result["D"]["B_S2_A_reachable_count"] = nreach

    # ---- B_S3: n=8 — positive (length-specific discovery) +
    # boundary (A-reachable, predicted control-unrealizable) ----
    s3b = {"M1": (40, 3000), "M2": (40, 3000), "P": (60, 4000),
           "Q1": (60, 4000), "Q2": (120, 6000), "R": (120, 6000)}
    # S3a: "head to front, reverse the rest" q=[0,7,...,1]: the n=8
    # generalization of the n=4 pi=[0,3,2,1] pattern (B1=[1..7],
    # B2=[0]) — predicted control-realizable, 10 passes.
    pi8a = [0, 7, 6, 5, 4, 3, 2, 1]
    chka = partA_partition(8, ((1, 2, 3, 4, 5, 6, 7), (0,)))
    assert chka == (tuple(pi8a), 10), chka
    T8a = (8, pi8a, make_taps(8, random.Random(48), 30))
    Ph8, Eh8, best8, evs8 = discover_multi(T8a, label="S3a", budgets=s3b)
    ver8 = verify(Ph8, Eh8, 8, pi8a, random.Random(800), 80)
    fin8, m8, tr8, hal8 = run(Ph8, Eh8, make_tape(8, [3, 7, 1, 9,
                                                      2, 5, 8, 4]), 24)
    # S3b: two-block reversed swap q=[3,2,1,0,7,6,5,4] (B1=[4..7],
    # B2=[0..3]): A-reachable (11 passes) but the pass-1 push pattern
    # (park prefix, push suffix) needs a state switch the 5-state
    # (symbol,state) control cannot place after tgt0 — predicted
    # control-unrealizable: the three-layer boundary.
    pi8b = [3, 2, 1, 0, 7, 6, 5, 4]
    chkb = partA_partition(8, ((4, 5, 6, 7), (0, 1, 2, 3)))
    assert chkb == (tuple(pi8b), 11), chkb
    T8b = (8, pi8b, make_taps(8, random.Random(49), 30))
    Ph8b, Eh8b, best8b, evs8b = discover_multi(T8b, label="S3b",
                                               budgets=s3b)
    ver8b = verify(Ph8b, Eh8b, 8, pi8b, random.Random(801), 80)
    result["D"]["B_S3a_n8_positive"] = {
        "pi": pi8a, "A_reachable": True, "A_passes": 10,
        "discovered": best8 >= 0.999, "verified": ver8 >= 0.999,
        "best_full": round(best8, 4), "evals": evs8,
        "passes_observed": m8}
    result["D"]["B_S3b_n8_boundary"] = {
        "pi": pi8b, "A_reachable": True, "A_passes": 11,
        "discovered": best8b >= 0.999, "verified": ver8b >= 0.999,
        "best_full": round(best8b, 4), "evals": evs8b,
        "prediction": "control-unrealizable (state-trajectory limit)"}
    print(f"[c45-B_S3a] n=8 head-front: disc={best8>=0.999} "
          f"ver={ver8:.3f} passes={m8} (A says 10) ev={evs8}", flush=True)
    print(f"[c45-B_S3b] n=8 two-block: disc={best8b>=0.999} "
          f"ver={ver8b:.3f} best={best8b:.3f} ev={evs8b}", flush=True)

    # ---- B_S4: one A-unreachable n=4 pi ----
    unreach4 = [p for p in allp4 if tuple(p) not in A4]
    s4 = None
    if unreach4:
        pi = unreach4[0]
        T = (4, pi, make_taps(4, random.Random(450 + sum(pi * 7)), 30))
        Ph, Eh, bestf, evs = discover_multi(T, label=f"S4-{pi}")
        ver = verify(Ph, Eh, 4, pi, random.Random(900 + sum(pi * 7)), 60)
        s4 = {"pi": pi, "best_full": round(bestf, 4),
              "verified": ver >= 0.999, "evals": evs,
              "negative": ver < 0.999}
        print(f"[c45-B_S4] A-unreachable pi={pi}: best={bestf:.3f} "
              f"ver={ver:.3f} negative={ver<0.999} ev={evs}", flush=True)
    result["D"]["B_S4_unreachable"] = s4

    # ---- B_S5: n=5 sample (mixed) ----
    rng5 = random.Random(77)
    reach5 = [list(k) for k in A5.keys()]
    unreach5 = [list(p) for p in itertools.permutations(range(5))
                if tuple(p) not in A5]
    rng5.shuffle(reach5)
    rng5.shuffle(unreach5)
    sample5 = (reach5[:6] + unreach5[:2])[:8]
    s5 = {}
    for pi in sample5:
        T = (5, pi, make_taps(5, random.Random(500 + sum(pi * 7)), 30))
        Ph, Eh, bestf, evs = discover_multi(T, label=f"S5-{pi}")
        ver = verify(Ph, Eh, 5, pi, random.Random(950 + sum(pi * 7)), 60)
        s5[str(pi)] = {"A_reachable": tuple(pi) in A5,
                       "B_discovered": bestf >= 0.999,
                       "verified": ver >= 0.999,
                       "best_full": round(bestf, 4), "evals": evs}
        print(f"[c45-B_S5] pi={pi} A={tuple(pi) in A5} "
              f"B={bestf>=0.999} ver={ver>=0.999} best={bestf:.3f} "
              f"ev={evs}", flush=True)
    result["D"]["B_S5_n5_sample"] = s5

    # ---- B_S1b: length generalization of non-reversal families ----
    # head-front family: pi_n = [0, n-1, ..., 1] (park src0 in pass 1,
    # push the rest — an n-uniform switch at the FIRST slot cell,
    # which the (symbol,state) control CAN express length-
    # independently). Test: the discovered n=4 [0,3,2,1] control and
    # the discovered n=8 S3a control must implement pi_8 / pi_16
    # with ONE control each. Negative control: the reversal control
    # on head-front pi_8 (task-specificity).
    s1b = {}
    hh4 = found.get("[0, 3, 2, 1]")
    if hh4 is not None:
        for nd in (8, 16):
            pi_n = [0] + list(range(nd - 1, 0, -1))
            c = score_all(hh4[0], hh4[1], nd, pi_n, make_taps(nd, rng, 20),
                          cap=3 * nd + 4)
            s1b[f"n4-control@n{nd}"] = round(c[4], 3)
    pi16 = [0] + list(range(15, 0, -1))
    c = score_all(Ph8, Eh8, 16, pi16, make_taps(16, rng, 20), cap=3 * 16 + 4)
    s1b["n8-S3a-control@n16"] = round(c[4], 3)
    pi8hf = [0, 7, 6, 5, 4, 3, 2, 1]
    c = score_all(PhR, EhR, 8, pi8hf, make_taps(8, rng, 20))
    s1b["reversal-control@headfront-n8"] = round(c[4], 3)
    # necessity probe: [1,3,2,0] ("src1 front, rest reversed") is NOT
    # an n-invariant-gate schedule (middle pop needs c = P-1 with
    # P < n-1 -> trigger pass m = n-P+2 depends on n) -> its n=4
    # control must NOT generalize to the n=8 embedding
    # [1,7,6,5,4,3,2,0]. Predict 0.0.
    wf4 = found.get("[1, 3, 2, 0]")
    if wf4 is not None:
        pi_w = [1] + list(range(7, 1, -1)) + [0]
        c = score_all(wf4[0], wf4[1], 8, pi_w, make_taps(8, rng, 20),
                      cap=3 * 8 + 4)
        s1b["n4-1320-control@n8-embedding"] = round(c[4], 3)
    result["D"]["B_S1b_generalization"] = s1b
    print(f"[c45-B_S1b] head-front generalization: {s1b}", flush=True)

    result["ckpt"] = "c45_perm_discovered.pt"
    result["wall_s"] = round(time.time() - T0, 1)
    _peak()
    result["peak_mb"] = round(PEAK, 1)
    print("RESULT " + json.dumps(result, default=str), flush=True)
    torch.save({"S3a": (Ph8, Eh8),
                "S2_found": found,
                "A_n4": {str(k): v for k, v in A4.items()},
                "A_n5": {str(k): v for k, v in A5.items()}},
               "c45_perm_discovered.pt")
    with open("log.jsonl", "a") as fp:
        fp.write(json.dumps({"ts": int(time.time()), **result}) + "\n")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
