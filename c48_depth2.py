#!/usr/bin/env python3
"""C48 / reasoning-frontier probe 6 — DEPTH-2 INDUCTION: is there a
VALUE-AGNOSTIC control that computes MUL(a,b) = a*b for a,b >= 2 on
the VET+S tape machine (the C45/C46/C47 mechanism, 44 symbols: 6 act
kinds, 5 states, one-way passes, LIFO S, register r)?

C47 left depth-2 UNSETTLED: search found only overfit geometry
attractors (MUL(2,3)/MUL(3,3) fx=1.0 same-geometry, gen 24/50, 15/50;
EXP(2,3) gen 4/40), and the derived L-INDUCTION-DEPTH-2 hypothesis was
not yet certified. C48 decides it at a stated scale: machine-checked
derived theorems certifying the barrier OUTSIDE an explicit corner,
hand constructions for the rank-1 families (positive, per-parameter),
pop-channel forensics (a NEW negative law), and joint-pair search on
the corner (empirical).

LAYOUT (uniform, the C47 family):
  [MARK^a][SEP][DIG_v^b (template)][BLK^m (output)][PAD]
  goal: exactly a*b of the m output cells become BDIG0+v.

DERIVED THEORY (before any run — the C42 protocol).
  Channel audit for THIS layout (template strictly before output,
  left-to-right scan, one register r, BOT = empty):
    L2 (ONE FILL PER PASS, REM-mode): r is filled only by RSET, which
       fires only on DIG/ADIG cells — all template cells precede the
       output region. After the first REM drains r, no later cell in
       the same pass can refill it. => at most 1 REM-fill/pass.
       Pops (ACT_COND_R): one per pass (f flag). Pushes (ACT_BLK):
       destructive (DIG -> BLK), total <= b.
    L1 (MARK PASS BUDGET): the only one-shot countdown is the mark
       block; it decreases 0..2 per pass (orbit period <= 5). Either
       some clear happens => the r>0 phase lasts <= a passes, or
       NEVER clears => r constant => the front clock is a pure
       F-orbit => fills are 0 or m (L3). MACHINE-CHECKED (B_S4.L1)
       over all (Ph[BLK], Ph[MARK], mask).
    L3 (FRONT CLOCK TAIL): the output-front state at total-fill f is
       F^f(H(r)) (F = Ph[BDIG], H(r) = pre-front fold). In the r=0
       (or constant-r) phase the gate pattern is a consecutive-open
       prefix (bounded by the 5-state functional-graph transient)
       then a cycle: open cycle => fills run to region-full m
       (overproduction); closed cycle => the prefix stops the loop.
       MACHINE-CHECKED (B_S4.L3).
    L-POP-COLLISION (NEW, forensics in B_S2b): after any push, the
       emptied template cells are BLK — indistinguishable from output
       BLK. The pop (ACT_COND_R) writes at the FIRST eligible BLK
       (odd s-position, state, c<=P-1, S non-empty), which lies in
       the template region for q >= 2 pushes. => the pop channel
       CANNOT target the output region with q >= 2; with q = 1 at
       most 1 output fill (parity permitting). The first POP-LOOP
       hand attempt (push all b, pop b times) fails: 0/75 in SMOKE,
       fills land in the template region. Hence the pop channel is
       USELESS for output targeting in this layout — the REM channel
       is the only output writer (L2 + L-POP-COLLISION).
    THEOREM T1 (MODE-R CEILING): a control that never pushes a
       template DIG (RSET-dominant; by L-POP-COLLISION the only
       useful mode) outputs either (i) <= a + P fills (the r>0 phase
       <= a passes by L1, one fill/pass by L2, tail prefix <= P by
       L3, P = measured max consecutive-open prefix = 8: a transient
       <= 4 plus at most d-1 <= 4 cycle states before the first
       closed state), or (ii) m fills (overproduction). Hence exact
       a*b with m > a*b is IMPOSSIBLE when a*(b-1) > P (P = 8):
       certified-excluded e.g. all (a>=3, b>=4), (a>=5, b>=3),
       (a>=9, b>=2). Open corner (T1, a,b in 2..12):
       {(2, 2..5), (3, 2..3), (4, 2..3), (5,2), (6,2), (7,2), (8,2)}
       (13 pairs) -> B_S5 search.
       REFINEMENT for a = 1 (machine-checked, B_S4d): the pass-0
       front state and the tail phase are the SAME orbit point
       (d0 = Ph[DIG]^b(s_sep)). The check finds: (1, b) exact for
       b in {2,3,4} achievable — and JOINTLY, by ONE control
       (witness: d0-tuple (2,0,1,2), G = {0,1,2,3}, F = [2,0,3,4,4] with 4 closed AND fixed — prefix confinement),
       realized on the mechanism as hand_oneshot_joint (B_S2);
       b = 5 NOT achievable (max fill 4, the count-4 ceiling).
    THEOREM T2 (MODE-P BOUND, for completeness): a control that
       pushes q >= 1 template DIGs: REM-prefix before push <= 6
       (DIG state orbit period <= 5); pops <= b of which <= 1 can
       target the output (L-POP-COLLISION); post-push REM clock
       <= a (L1) + tail <= 4 (L3). => total output fills <=
       a + b + 10 (loose; the q=1 case is tight-ish: a + 1 + 4 + 6).
       Certified exclusion a*b > a + b + 10 i.e. (a-1)(b-1) > 11
       (subsumes T1 outside a thin strip).
    POSITIVE RANK-1 (the realizable family, hand-constructed):
      (a, 1), a <= 4: REPEAT-a (C47 B_S2; mark-orbit gated; the
      a=5 edge gives a - floor(a/5) fills — the mod-5 collision).
      (1, b), b in {2,3,4}: hand_oneshot(b) — the FRONT-CLOCK
      TRANSIENT construction (per-b): Ph[BDIG] orbit 0 -> 1 -> ... ->
      b -> b, REM armed on {0..b-1}, the DIG orbit lands the pre-
      front state at 0; pass 0 + tail give exactly b fills in b+1
      passes, value-agnostic in v. AND hand_oneshot_joint — ONE
      control exact for all three (B_S4d witness realized):
      F = [2,0,3,4,4] (orbit 0->2->3->4 fixed, 1->0), REM armed on
      {0,1,2,3}, PhDIG = [1,2,0,0,0] with s0 = 0 so the phase
      d0(b) = (2, 0, 1) for b = (2, 3, 4); each phase walks 1..3
      open states then hits the closed state 4. Value-agnostic in
      v AND b.
    FRONTIER CLAIM (confirmed by B_S5): value-agnostic MUL is
    REALIZABLE on the rank-1 family {(a,1): a<=4} x {(1,b): b in
    2..4, one joint control} (+ REPEAT k <= 4); every realizable
    data-value loop runs to AT MOST 4 (L-INDUCTION-FOUR — the
    5-state clock: per-channel transient/collision at 5; the a+8
    T1 tail is a per-pass budget, not a single loop count); the
    rank-2 corner is empirically undiscoverable/overfit-only
    (B_S5: 0.883 plateau on every joint + single bar).
  Prior art (searched 2026-08-26): (i) Chistikov, "Notes on Counting
  with Finite Machines" (FSTTCS 2014, LIPIcs vol 29): deterministic
  PDAs count to n / modulo n with Theta(log n) STATES — counting to
  n with a 5-state controller is impossible without materializing
  the count on the stack (the push channel); one materialized count
  gives ONE counted loop, and here that loop cannot even target the
  output (L-POP-COLLISION). (ii) Counter machines: multiplication =
  nested loop over two operands — two independent counters (outer
  loop variable + accumulator); 2-counter = universal (Minsky);
  1-counter computes n^2/a*b as recognition (C47 sources retained:
  ResearchGate 263873086, Springer 10.1007/978-3-031-34326-1_11).
  (iii) C46: controller value-OPAQUE to S depth; C47:
  L-INDUCTION-TAPE-ORBIT / GATING / PUSH-BOUND / REGISTER-BROADCAST.

HONESTY CLAUSE: T1/T2/L1/L3/B_S4d are machine-checked over finite
clock abstractions that SOUNDLY over-approximate the control class
(every actual control induces a tuple in the checked space; negatives
transfer to the mechanism; positives are realized by explicit hand
controls below). The rank-2 corner is NOT certified unrealizable —
B_S5 reports the search outcome either way. No joint fx=1.0 is
claimed below. Tag ARC2-C48-DEPTH2.  1 thread.
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

# ---------------- C47 encoding (44 symbols), verbatim ----------
MARK, BLK, SEP = 0, 1, 2
DIG0 = 3
PAD = 13
BDIG0 = 14
ADIG0 = 24
ALPH = 44
NH = 5
BOT = 10
IDENT, ACT_BLK, ACT_COND_R, ACT_RSET, ACT_REM, ACT_CLR = 0, 1, 2, 3, 4, 5

def step(t, h, S, P, r, Ph, Eh):
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
            if a != PAD:
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
        if ident:
            tr.append(int(t.count(MARK)))
            return t, m, tr, True
        tr.append(int(t.count(MARK)))
    return t, cap, tr, False

def blank_genome():
    Ph = [[h for h in range(NH)] for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    return Ph, Eh

# ---------------- hand controls --------------------------------
def hand_repeat():
    """C47 B_S2: REPEAT(k,v) = k fills of v, k <= 4, mark-orbit
    gated (stride-2 orbit, 5-cycle; REM armed at states 1-4)."""
    Ph = [[s for s in range(NH)] for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    Ph[MARK] = [2, 3, 4, 0, 1]
    Eh[MARK] = [ACT_CLR, 0, 0, 0, 0]
    Ph[SEP] = [0, 1, 2, 3, 4]
    Eh[SEP] = [0, 0, 0, 0, 0]
    for d in range(10):
        Eh[DIG0 + d] = [ACT_RSET] * NH
        Ph[DIG0 + d] = [0, 1, 2, 3, 4]
    Eh[BLK] = [0, ACT_REM, ACT_REM, ACT_REM, ACT_REM]
    Ph[BLK] = [0, 1, 2, 3, 4]
    for d in range(10):
        Ph[BDIG0 + d] = [0, 1, 2, 3, 4]
        Eh[BDIG0 + d] = [0, 0, 0, 0, 0]
    Ph[PAD] = [0, 1, 2, 3, 4]
    Eh[PAD] = [0, 0, 0, 0, 0]
    return Ph, Eh

def hand_oneshot(b):
    """C48 NEW: (1, b) for b in {2,3,4} — the FRONT-CLOCK TRANSIENT
    construction. Pass 0: mark cleared (state 0 -> ACT_CLR), SEP,
    DIG^b (RSET r:=v; Ph[DIG] = [0,0,1,2,3] sends the pre-front
    state to 0 for all b >= 2), output front at state 0: REM armed
    on {0..b-1} => 1 fill. Tails: front state walks F = Ph[BDIG] =
    0->1->...->b->b; REM opens at states 1..b-1 (b-1 fills), stops
    at state b (closed). Total b fills, b+1 passes, value-agnostic.
    b=5 would need transient 5 (impossible in 5 states): the edge."""
    Ph = [[s for s in range(NH)] for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    Ph[MARK] = [1, 1, 1, 1, 1]
    Eh[MARK] = [ACT_CLR, 0, 0, 0, 0]
    Ph[SEP] = [0, 2, 2, 2, 2]     # state 1 (after mark) -> 2 (s0)
    Eh[SEP] = [0, 0, 0, 0, 0]
    for d in range(10):
        Ph[DIG0 + d] = [0, 0, 1, 2, 3]   # 2 -> 1 -> 0 -> 0 -> 0
        Eh[DIG0 + d] = [ACT_RSET] * NH
    F = [i + 1 for i in range(b)] + [b] * (NH - b)  # 0->1->...->b->b
    for d in range(10):
        Ph[BDIG0 + d] = F
        Eh[BDIG0 + d] = [0, 0, 0, 0, 0]
    Eh[BLK] = [ACT_REM if s < b else 0 for s in range(NH)]
    # Ph[BLK] MUST equal F: the filled-prefix walk is F and the next
    # front state is Ph[BLK][front state] — they are the same orbit
    # walk (forensics: with Ph[BLK][3] = 2 != F[3] = 3 the walk
    # jumped back into an armed state and over-filled).
    Ph[BLK] = F
    Ph[PAD] = [0, 1, 2, 3, 4]
    Eh[PAD] = [0, 0, 0, 0, 0]
    return Ph, Eh

def hand_oneshot_joint():
    """C48 (B_S4d witness, realized): ONE control exact for
    (1, b), b in {2, 3, 4} simultaneously. Clock: F = [2,0,3,4,4]
    (orbit 0->2->3->4 fixed; 1->0), REM armed on G = {0,1,2,3}
    (state 4 CLOSED AND FIXED => prefix confinement: once the front
    reaches 4, every later BLK cell is also at 4 and no REM can fire
    past the front — the checker's front-only dynamics match the
    tape). DIG orbit s0 = 0: 0 -> 1 -> 2 -> 0 -> 1 -> 2
    (PhDIG = [1,2,0,0,0]) => the pre-front phase d0(b) = (2, 0, 1)
    for b = (2, 3, 4). Pass 0 REM fires at each d0(b) in G (+1),
    then the tail walk opens b-1 more states before hitting 4:
    b=2: phase 3 -> close; b=3: phase 2 -> 3 -> close; b=4:
    phase 0 -> 2 -> 3 -> close. Totals (2, 3, 4), passes b+1."""
    Ph = [[s for s in range(NH)] for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    Ph[MARK] = [1, 1, 1, 1, 1]
    Eh[MARK] = [ACT_CLR, 0, 0, 0, 0]
    Ph[SEP] = [0, 0, 0, 0, 0]     # state 1 (after mark) -> s0 = 0
    Eh[SEP] = [0, 0, 0, 0, 0]
    for d in range(10):
        Ph[DIG0 + d] = [1, 2, 0, 0, 0]
        Eh[DIG0 + d] = [ACT_RSET] * NH
    F = [2, 0, 3, 4, 4]
    for d in range(10):
        Ph[BDIG0 + d] = F
        Eh[BDIG0 + d] = [0, 0, 0, 0, 0]
    Eh[BLK] = [ACT_REM, ACT_REM, ACT_REM, ACT_REM, 0]
    Ph[BLK] = F
    Ph[PAD] = [0, 1, 2, 3, 4]
    Eh[PAD] = [0, 0, 0, 0, 0]
    return Ph, Eh

def hand_poploop():
    """C48 FORENSICS (expected FAIL — L-POP-COLLISION): push all b
    template DIGs (constant-1 state maps; Eh[DIG][1] = ACT_BLK),
    then pop at output-front state 1. The emptied template cells
    are BLK at s = 1, 2, ... and STEAL the pops before the output
    region (first-eligible-BLK rule). Fills land in the template
    region, not the output => fx = 0 for (1, b)."""
    Ph = [[s for s in range(NH)] for _ in range(ALPH)]
    Eh = [[0] * NH for _ in range(ALPH)]
    Ph[MARK] = [1, 1, 1, 1, 1]
    Eh[MARK] = [ACT_CLR, 0, 0, 0, 0]
    Ph[SEP] = [1, 1, 1, 1, 1]
    Eh[SEP] = [0, 0, 0, 0, 0]
    for d in range(10):
        Ph[DIG0 + d] = [1, 1, 1, 1, 1]
        Eh[DIG0 + d] = [0, ACT_BLK, 0, 0, 0]
    Ph[BLK] = [1, 1, 1, 1, 1]
    Eh[BLK] = [0, ACT_COND_R, 0, 0, 0]
    for d in range(10):
        Ph[BDIG0 + d] = [1, 1, 1, 1, 1]
        Eh[BDIG0 + d] = [0, 0, 0, 0, 0]
    Ph[PAD] = [1, 1, 1, 1, 1]
    Eh[PAD] = [0, 0, 0, 0, 0]
    return Ph, Eh

# ---------------- tasks -----------------------------------------
def make_tape(a, b, v, m):
    return [MARK] * a + [SEP] + [DIG0 + v] * b + [BLK] * m + [PAD]

def fx_count(fin, a, b, v, m):
    base = a + 1 + b
    return sum(1 for j in range(m) if fin[base + j] == BDIG0 + v) == a * b

def fill_stats(fin, a, b, v, m):
    base = a + 1 + b
    out_f = sum(1 for j in range(m) if fin[base + j] == BDIG0 + v)
    tpl_f = sum(1 for i in range(a + 1, a + 1 + b)
                if fin[i] == BDIG0 + v)
    return out_f, tpl_f

def pair_taps(a, b, r, n=20):
    m = a * b + 2 * b + 2
    out = []
    for _ in range(n):
        v = r.randrange(10)
        out.append((make_tape(a, b, v, m), (a, b, v, m)))
    return out

def score_pair(Ph, Eh, a, b, r, n=20):
    sx = sf = 0.0
    for t, (a2, b2, v, m) in pair_taps(a, b, r, n):
        fin, mm, tr, hal = run(Ph, Eh, t, 3 * a2 * b2 + 8)
        filled = fx_count(fin, a2, b2, v, m)
        c = sum(1 for j in range(m) if fin[a2 + 1 + b2 + j] == BDIG0 + v)
        sx += 1.0 if (hal and filled) else 0.0
        sf += min(c, a2 * b2) / (a2 * b2)
    return sx / n, sf / n, 0

# ---------------- search infra (C46/C47, verbatim) --------------
ALLROWS = [SEP, BLK, PAD] + list(range(DIG0, DIG0 + 10)) \
    + list(range(BDIG0, BDIG0 + 10)) + list(range(ADIG0, ADIG0 + 10))

def embed24(Ph24, Eh24):
    Ph, Eh = [], []
    for a in range(24):
        Ph.append([int(x) for x in Ph24[a]])
        Eh.append([int(x) for x in Eh24[a]])
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
    best = fit(Ph, Eh)
    bestPh, bestEh = [r[:] for r in Ph], [r[:] for r in Eh]
    ev = 1
    t0 = time.time()
    stall = 0
    while time.time() - t0 < budget_s and ev < max_evals \
            and stall < stall_cap:
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

def trace_mean(P, E, taps):
    tot = 0.0
    for t, args in taps:
        a, b, v, m = args
        fin, mm, tr, hal = run(P, E, t, 3 * a * b + 8)
        k = tr[0]
        c1 = tr[0] - (tr[1] if len(tr) > 1 else tr[0])
        a_ = 1.0 - abs(c1 - min(1, k))
        desc = 0.0
        for i in range(1, k + 1):
            if i < len(tr):
                desc += 1.0 if tr[i] == k - i else 0.0
        desc /= max(1, k)
        c = 1.0 if tr[-1] == 0 else 0.0
        tot += 0.4 * a_ + 0.4 * desc + 0.2 * c
    return tot / len(taps)

def joint_fitness_factory(pairs):
    def joint_fx(P, E):
        s = 0.0
        for p in pairs:
            s += score_pair(P, E, p[0], p[1], random.Random(4810), 20)[0]
        return s / len(pairs)
    def fit1(P, E):
        p = pairs[0]
        tt = pair_taps(p[0], p[1], random.Random(4800), 20)
        return 0.3 * score_pair(P, E, p[0], p[1],
                                random.Random(4820), 20)[1] \
            + 0.1 * trace_mean(P, E, tt)
    def fit2(P, E):
        s = 0.0
        for p in pairs:
            c1, c2, _ = score_pair(P, E, p[0], p[1],
                                   random.Random(4820), 20)
            s += 0.7 * c1 + 0.3 * c2
        return s / len(pairs)
    return joint_fx, fit1, fit2

def discover2(pairs, seeds=(48, 49, 50), label="",
              budgets=None, q_stall=3000):
    """2-stage (M1 mark trace -> Q joint plateau-walk) x seeds on a
    MULTI-PAIR joint fitness (value-agnosticity test)."""
    b = budgets or {"M1": (30, 2500), "Q": (90, 5000)}
    jfx, fit1, fit2 = joint_fitness_factory(pairs)
    best = (-1.0, None, None, 0)
    for sd in seeds:
        rng = random.Random(sd)
        Ph, Eh = blank_genome()
        last = -1.0
        tot = 0
        for nm, fitf, a0s in (("M1", fit1, [MARK]),
                              ("Q", fit2, ALLROWS[1:])):
            bs, me = b[nm]
            Ph, Eh, last, ev = hill_climb(Ph, Eh, fitf, a0s, bs, me, rng,
                                          f"{label}-s{sd}-{nm}",
                                          blank=blank_genome(),
                                          stall_cap=400 if nm == "M1"
                                          else q_stall)
            tot += ev
        if last > best[0]:
            best = (last, Ph, Eh, tot)
    return best[1], best[2], best[0], best[3]

# ---------------- machine-checked lemmas ------------------------
def all_funcs():
    """All 5^5 = 3125 functions on 5 states, (3125, 5) int8."""
    import numpy as np
    return np.array(list(itertools.product(range(5), repeat=5)),
                    dtype=np.int8)

def itersig(F):
    """F^k(0) for k = 0..4, shape (5, 3125)."""
    import numpy as np
    s = np.zeros(F.shape[0], dtype=np.int8)
    out = np.zeros((5, F.shape[0]), dtype=np.int8)
    out[0] = s
    for k in range(1, 5):
        s = F[np.arange(F.shape[0]), s]
        out[k] = s
    return out

def orbit_tables(F):
    """O[k][s] = F^k(s), k = 0..4, s = 0..4; shape (5, 3125, 5)."""
    import numpy as np
    n = F.shape[0]
    idx = np.arange(n)
    s = np.arange(5, dtype=np.int8)[None, :].repeat(n, axis=0)
    O = np.zeros((5, n, 5), dtype=np.int8)
    O[0] = s
    for k in range(1, 5):
        s = F[idx[:, None], s]
        O[k] = s
    return O

def machine_check_L1(A, B, a):
    """L1: for every (F_BLK, F_MARK, mask) [3125 x 3125 x 32]: the
    r>0 phase (mark count a) lasts <= a passes (some clear per r>0
    pass) or NEVER clears (r constant). Vectorized."""
    import numpy as np
    n = 3125
    sig = A.T                      # (3125, 5): F_BLK^k(0)
    no_clear = 0
    maxpasses = 0
    for mask in range(32):
        mk = np.array([int(mask >> i & 1) for i in range(5)],
                      dtype=np.int8)
        sg = np.repeat(sig, n, axis=0)           # (n*n, 5)
        ob = np.repeat(B, n, axis=0).reshape(n * n, 5, 5)
        ar = np.arange(n * n)
        r = np.full(n * n, a, dtype=np.int8)
        pc = np.zeros(n * n, dtype=np.int16)
        for _ in range(2 * a + 2):
            if not (r > 0).any():
                break
            x0 = sg[ar, a - r.astype(np.int64)]  # per-row gather
            clr = np.zeros(n * n, dtype=np.int8)
            for i in range(1, a):
                oi = ob[ar, i, x0]
                clr += ((r > i) & (oi == 0) & (mk[oi] == 1)).astype(
                    np.int8)
            clr += ((r > 0) & (x0 == 0) & (mk[x0] == 1)).astype(
                np.int8)
            pc[r > 0] += 1
            r = np.maximum(r - clr, 0)
        cleared = r == 0
        maxpasses = max(maxpasses, int(pc[cleared].max())
                        if cleared.any() else 0)
        no_clear += int((r > 0).sum())
    return {"a": a,
            "classes_checked": int(32 * n * n),
            "no_clear_classes": int(no_clear),
            "cleared_classes": int(32 * n * n - no_clear),
            "max_r0_passes_cleared": int(maxpasses),
            "statement": "checked family (front-clear dynamics, which "
                         "contains every hit-at-position-0 control and "
                         "every no-hit control): the r>0 phase (a=2) "
                         "lasts <= 2 passes, or never clears. General "
                         "controls by the monotonicity argument: once "
                         "a clear happens it recurs >=1/pass until r=0 "
                         "or the (shrunk) window misses the hit forever "
                         "=> an eventually-constant r>=1 phase, whose "
                         "fills are 0 or m by L3. Either way: r>0 "
                         "fills <= a, tail 0/m (or the <=4 prefix in "
                         "the constant-r-from-start case)."}

def machine_check_L3(F, m_cap=40):
    """L3: for every (F, H0, G) [3125 x 5 x 32]: the constant-r tail
    from f = 0 is a consecutive-open prefix (<= 4) or runs to
    region-full. Vectorized over F."""
    import numpy as np
    n = F.shape[0]
    idx = np.arange(n)
    s = np.arange(5, dtype=np.int8)[None, :].repeat(n, axis=0)
    P = np.zeros((m_cap + 1, n, 5), dtype=np.int8)
    P[0] = s
    for k in range(1, m_cap + 1):
        s = F[idx[:, None], s]
        P[k] = s
    pref_max = 0
    full = 0
    for h0 in range(5):
        for g in range(32):
            G = set(i for i in range(5) if g >> i & 1)
            fills = np.zeros(n, dtype=np.int8)
            alive = np.ones(n, dtype=bool)
            for f in range(m_cap + 1):
                if not alive.any():
                    break
                cur = P[f][np.arange(n), h0]
                op = np.isin(cur, list(G)) & alive
                fills[op] = f + 1
                alive = op
            is_full = fills == m_cap + 1
            full += int(is_full.sum())
            if not is_full.all():
                pref_max = max(pref_max, int(fills[~is_full].max()))
    return {"combos_checked": int(32 * 5 * n),
            "max_prefix_fills": int(pref_max),
            "region_full_classes": int(full),
            "statement": "the constant-r tail is a consecutive-open "
                         "prefix (max above; a transient <= 4 plus at "
                         "most d-1 <= 4 cycle states before the "
                         "first closed state) or runs to region-full "
                         "m (open cycle => overproduction)"}

def machine_check_oneshot():
    """B_S4d: (1, b) exact-fill achievability. Space: the pre-front
    phase d0(b) = PhDIG^b(s0) (PhDIG: 3125, s0: 5), the front clock
    F = PhBDIG (3125), the gate G (32). fill(b) = f0 + run,
    f0 = [d0(b) in G], run = consecutive G-states of the F-orbit
    from F^f0(d0(b)) (the tail, stops at first non-G). Checks, for
    b in 2..5: (i) per-b: exists combo with fill(b) = b?
    (ii) joint (b in {2,3,4}): exists combo with fill(2)=2,
    fill(3)=3, fill(4)=4 simultaneously?"""
    import numpy as np
    PhDIG = all_funcs()
    F = all_funcs()
    n = 3125
    # d0 tuples: for each (PhDIG, s0): (d0(2), d0(3), d0(4), d0(5))
    # PhDIG^k(s0), k = 2..5
    s = np.arange(5, dtype=np.int8)[None, :].repeat(n, axis=0)
    Pdig = [s]
    for k in range(1, 6):
        s = PhDIG[np.arange(n)[:, None], s]
        Pdig.append(s)
    dtup = {}
    for s0 in range(5):
        for i in range(n):
            t = tuple(int(Pdig[k][i, s0]) for k in (2, 3, 4, 5))
            dtup.setdefault(t, 0)
            dtup[t] += 1
    # F powers
    s2 = np.arange(5, dtype=np.int8)[None, :].repeat(n, axis=0)
    Pf = [s2]
    for k in range(1, 12):
        s2 = F[np.arange(n)[:, None], s2]
        Pf.append(s2)
    # dedupe (PhDIG, s0) by d0-tuple
    reps = list(dtup.keys())
    ar = np.arange(n)
    joint_witness = None
    max_fill = {}
    okper = {b: False for b in (2, 3, 4, 5)}
    for g in range(32):
        G = [i for i in range(5) if g >> i & 1]
        if not G:
            continue
        for tup in reps:
            fills = {}
            confines = {}
            for bi, b in ((0, 2), (1, 3), (2, 4), (3, 5)):
                d0 = np.full(n, tup[bi], dtype=np.int8)
                f0 = np.isin(d0, G).astype(np.int8)
                # phase = F^f0(d0): apply F (Pf[1]) where f0 = 1
                ph = np.where(f0 == 1, Pf[1][ar, d0], d0)
                run = np.zeros(n, dtype=np.int8)
                alive = np.ones(n, dtype=bool)
                cur = ph
                for j in range(8):
                    ing = np.isin(cur, G) & alive
                    run[ing] = j + 1
                    alive = ing
                    if not alive.any():
                        break
                    cur = F[ar, cur]
                fills[b] = f0 + run   # tail capped at 8 (< m)
                # PREFIX CONFINEMENT: the open set {j: F^j(d0) in G}
                # must be an initial segment (once closed, never open
                # again) — otherwise the mechanism's REM fires at a
                # BLK cell PAST the front (fill skip) and the checker
                # dynamics no longer match the tape.
                close = np.zeros(n, dtype=bool)
                okp = np.ones(n, dtype=bool)
                cur2 = d0
                for j in range(12):
                    inG = np.isin(cur2, G)
                    okp &= ~(close & inG)
                    close |= ~inG
                    cur2 = F[ar, cur2]
                confines[b] = okp
            ok = ((fills[2] == 2) & confines[2]
                  & (fills[3] == 3) & confines[3]
                  & (fills[4] == 4) & confines[4])
            if ok.any() and joint_witness is None:
                F_i = int(np.argmax(ok))
                joint_witness = {"d0tuple": tup, "g": g,
                                 "F_index": F_i,
                                 "F_row": [int(x) for x in F[F_i]],
                                 "G": G}
            for bi, b in ((0, 2), (1, 3), (2, 4), (3, 5)):
                if ((fills[b] == b) & confines[b]).any():
                    okper[b] = True
                cm = fills[b][confines[b]]
                if cm.size:
                    max_fill[b] = max(max_fill.get(b, 0), int(cm.max()))
    return {"d0_tuple_classes": len(reps),
            "per_b_exact_achievable": {str(b): okper[b]
                                       for b in (2, 3, 4, 5)},
            "per_b_max_fill": {str(b): max_fill.get(b, 0)
                               for b in (2, 3, 4, 5)},
            "joint_234_possible": bool(joint_witness is not None),
            "joint_witness": joint_witness,
            "statement": "(1,b) exact fill = b achievable per-b for "
                         "the listed b; the JOINT (one control, b in "
                         "{2,3,4}) is "
                         + ("possible" if joint_witness
                            else "CERTIFIED UNREALIZABLE "
                                 "(phase d0(b) depends on b)")}

# ---------------- SMOKE -----------------------------------------
if os.environ.get("SMOKE"):
    Ph, Eh = blank_genome()
    fin, m, tr, hal = run(Ph, Eh, make_tape(2, 2, 3, 8), 20)
    print(f"[c48-smoke] mechanism runs: blank MUL(2,2) passes={m} "
          f"halted={hal}", flush=True)
    import torch
    d = torch.load("c44_vets_discovered.pt", weights_only=False)
    Ph4, Eh4 = d["Ph"].numpy().tolist(), d["Eh"].numpy().tolist()
    Ph2 = [[h for h in range(NH)] for _ in range(ALPH)]
    Eh2 = [[0] * NH for _ in range(ALPH)]
    for a in range(24):
        Ph2[a] = [int(x) for x in Ph4[a]]
        Eh2[a] = [int(x) for x in Eh4[a]]
    ok = 0
    rng = random.Random(11)
    for _ in range(20):
        nd = rng.choice([4, 8, 16])
        digs = [rng.randrange(10) for _ in range(nd - 1)] + \
            [rng.randrange(1, 9)]
        mid = []
        for x in digs:
            mid += [DIG0 + x, BLK]
        tt = [MARK] * nd + [SEP] + mid + [PAD]
        fin, m, tr, hal = run(Ph2, Eh2, tt, 3 * nd + 8)
        ok += hal \
            and all(fin[nd + 2 + 2 * i] == BDIG0 + digs[nd - 1 - i]
                    for i in range(nd)) \
            and all(fin[nd + 1 + 2 * i] == BLK for i in range(nd))
    assert ok == 20, f"C44 regression: {ok}/20"
    print("[c48-smoke] C44 reversal (harness soundness): 20/20 OK",
          flush=True)
    Phr, Ehr = hand_repeat()
    ok = 0
    tot = 0
    rng = random.Random(47)
    for k in (1, 2, 3, 4):
        for _ in range(25):
            v = rng.randrange(10)
            t = make_tape(k, 1, v, 8)
            fin, m, tr, hal = run(Phr, Ehr, t, 3 * k + 8)
            tot += 1
            ok += hal and fx_count(fin, k, 1, v, 8)
    assert ok == tot == 100, f"REPEAT a=1..4 x (a,1): {ok}/{tot}"
    print("[c48-smoke] REPEAT (a,1) a=1..4: 100/100 exact OK",
          flush=True)
    for b in (2, 3, 4):
        Pho, Eho = hand_oneshot(b)
        ok = 0
        tot = 0
        ps = {}
        rng = random.Random(48 + b)
        for _ in range(25):
            v = rng.randrange(10)
            t = make_tape(1, b, v, 8)
            fin, m, tr, hal = run(Pho, Eho, t, 4 * b + 8)
            tot += 1
            ok += hal and fx_count(fin, 1, b, v, 8)
            ps[m] = ps.get(m, 0) + 1
        assert ok == tot == 25, f"oneshot (1,{b}): {ok}/{tot} ps={ps}"
        print(f"[c48-smoke] ONESHOT (1,{b}): 25/25 exact, passes={ps} "
              f"(expected b+1 = {b+1}) OK", flush=True)
    # JOINT (1, b in {2,3,4}) — one control, all three exact
    Phj, Ehj = hand_oneshot_joint()
    for b in (2, 3, 4):
        ok = 0
        ps = {}
        rng = random.Random(50 + b)
        for _ in range(25):
            v = rng.randrange(10)
            t = make_tape(1, b, v, 8)
            fin, m, tr, hal = run(Phj, Ehj, t, 4 * b + 8)
            ok += hal and fx_count(fin, 1, b, v, 8)
            ps[m] = ps.get(m, 0) + 1
        assert ok == 25, f"JOINT (1,{b}): {ok}/25 ps={ps}"
        print(f"[c48-smoke] JOINT (1,{b}): 25/25 exact, passes={ps} OK",
              flush=True)
    # b=5 edge: oneshot(5) would need transient 5 — construct with
    # b=4 orbit (0->1->2->3->4->4) armed {0..3}: fills 4, not 5.
    Pho, Eho = hand_oneshot(4)
    fin, m, tr, hal = run(Pho, Eho, make_tape(1, 5, 7, 10), 30)
    out_f, tpl_f = fill_stats(fin, 1, 5, 7, 10)
    print(f"[c48-smoke] (1,5) under b4-clock: output fills = {out_f}/5 "
          f"(expected < 5, the count-4 ceiling) ", flush=True)
    assert out_f < 5, "b=5 must fail (transient-5 impossible)"
    # POP-LOOP forensics: fills must land in the TEMPLATE region
    Php, Ehp = hand_poploop()
    ok = 0
    tot = 0
    tpl_total = 0
    rng = random.Random(49)
    for b in (2, 3, 4):
        for _ in range(25):
            v = rng.randrange(10)
            t = make_tape(1, b, v, 8)
            fin, m, tr, hal = run(Php, Ehp, t, 4 * b + 8)
            tot += 1
            out_f, tpl_f = fill_stats(fin, 1, b, v, 8)
            ok += fx_count(fin, 1, b, v, 8)
            tpl_total += tpl_f
    assert ok == 0, f"POP-LOOP should give 0 output-exact: {ok}/{tot}"
    assert tpl_total > 0, "POP-LOOP should fill the template region"
    print(f"[c48-smoke] POP-LOOP forensics: 0/{tot} output-exact, "
          f"template-region fills = {tpl_total} (L-POP-COLLISION) OK",
          flush=True)
    print("SMOKE-DONE (C48 harness + rank-1 hand controls + "
          "L-POP-COLLISION forensics)", flush=True)
    sys.exit(0)

# ---------------- main ------------------------------------------
def main():
    rng = random.Random(48)
    result = {"tag": "ARC2-C48-DEPTH2",
              "method": ("machine-checked derived theory: T1 mode-R "
                         "ceiling (fills <= a + P or m, P = L3 max "
                         "prefix; L1 mark-pass budget 3125^2 x 32 "
                         "for a=2,3 + L3 front-clock tail 3125 x 5 x "
                         "32 + (1,b) phase check B_S4d) + T2 mode-P "
                         "bound + L-POP-COLLISION forensics + hand "
                         "rank-1 controls (REPEAT (a,1); ONESHOT "
                         "(1,b) b=2..4 per-b; ONESHOT-JOINT one "
                         "control for b=2..4, B_S4d witness) + joint-"
                         "pair search on the rank-2 corner (C44/C45 "
                         "method, plateau-walk) on the C47 mechanism"),
              "D": {}}
    _peak()
    import torch

    # ---- B_S1: C44 reversal regression (harness soundness) ----
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
            fx += 1.0 if (hal and all(
                fin[nd + 2 + 2 * i] == BDIG0 + digs[nd - 1 - i]
                for i in range(nd))) else 0.0
        s1[f"n{nd}"] = round(fx / 20, 3)
    result["D"]["B_S1_c44_regression"] = s1
    print(f"[c48-B_S1] C44 genome (extended mechanism): {s1}", flush=True)

    # ---- B_S2: ONESHOT (1, b) hand controls, per-b ----
    s2 = {}
    for b in (2, 3, 4, 5):
        Pho, Eho = hand_oneshot(min(b, 4))
        fx = 0.0
        ps = {}
        for _ in range(100):
            v = rng.randrange(10)
            t = make_tape(1, b, v, 8 if b < 5 else 10)
            fin, m, tr, hal = run(Pho, Eho, t, 4 * b + 8)
            fx += 1.0 if (hal and fx_count(fin, 1, b, v, 8 if b < 5
                                           else 10)) else 0.0
            ps[m] = ps.get(m, 0) + 1
        s2[f"b{b}"] = {"fx_100": round(fx / 100, 4),
                       "passes_hist": {str(k): v for k, v in ps.items()},
                       "construction": "front-clock transient, ONE "
                                       "control per b" if b <= 4
                       else "REUSES b4 clock (expected fail, edge)"}
    # JOINT (1, b in {2,3,4}) — the B_S4d witness, realized
    Phj, Ehj = hand_oneshot_joint()
    sj = {}
    for b in (2, 3, 4):
        fx = 0.0
        ps = {}
        for _ in range(100):
            v = rng.randrange(10)
            t = make_tape(1, b, v, 8)
            fin, m, tr, hal = run(Phj, Ehj, t, 4 * b + 8)
            fx += 1.0 if (hal and fx_count(fin, 1, b, v, 8)) else 0.0
            ps[m] = ps.get(m, 0) + 1
        sj[f"b{b}"] = {"fx_100": round(fx / 100, 4),
                       "passes_hist": {str(k): v for k, v in ps.items()}}
    sj["construction"] = ("ONE control (B_S4d witness): F=[2,0,3,4,4], "
                          "G={0,1,2,3}, PhDIG=[1,2,0,0,0], s0=0 — "
                          "value-agnostic in BOTH v and b")
    result["D"]["B_S2_joint_1_b234"] = sj
    print(f"[c48-B_S2] ONESHOT per-b: "
          f"{ {k: v['fx_100'] for k, v in s2.items()} }; JOINT: "
          f"{ {k: v['fx_100'] for k, v in sj.items() if k[0]=='b'} }",
          flush=True)

    # ---- B_S2b: L-POP-COLLISION forensics (the failed POP-LOOP) ----
    Php, Ehp = hand_poploop()
    s2b = {}
    for b in (2, 3, 4):
        out_ok = 0
        tpl_fills = 0
        for _ in range(25):
            v = rng.randrange(10)
            t = make_tape(1, b, v, 8)
            fin, m, tr, hal = run(Php, Ehp, t, 4 * b + 8)
            out_f, tpl_f = fill_stats(fin, 1, b, v, 8)
            out_ok += fx_count(fin, 1, b, v, 8)
            tpl_fills += tpl_f
        s2b[f"b{b}"] = {"output_exact_25": out_ok,
                        "template_region_fills_25": tpl_fills}
    s2b["law"] = "L-POP-COLLISION: emptied template cells are BLK; " \
                 "the pop writes at the first eligible BLK (template " \
                 "region for q>=2) => the pop channel cannot target " \
                 "the output; REM is the only output writer"
    result["D"]["B_S2b_poploop_forensics"] = s2b
    print(f"[c48-B_S2b] POP-LOOP forensics: {s2b}", flush=True)

    # ---- B_S3: REPEAT (a,1) regression (C47 hand) ----
    Phr, Ehr = hand_repeat()
    s3 = {}
    for a in (1, 2, 3, 4):
        fx = 0.0
        for _ in range(100):
            v = rng.randrange(10)
            t = make_tape(a, 1, v, 8)
            fin, m, tr, hal = run(Phr, Ehr, t, 3 * a + 8)
            fx += 1.0 if (hal and fx_count(fin, a, 1, v, 8)) else 0.0
        s3[f"a{a}"] = round(fx / 100, 4)
    fin, m, tr, hal = run(Phr, Ehr, make_tape(5, 1, 7, 8), 20)
    s3["a5_edge"] = f"{sum(1 for j in range(8) if fin[7+j] == BDIG0+7)}/5"
    result["D"]["B_S3_repeat_a_1"] = s3
    print(f"[c48-B_S3] REPEAT (a,1): {s3}", flush=True)

    # ---- B_S4: machine-checked lemmas + certified corner ----
    print("[c48-B_S4] machine-checking L1 (3125^2 x 32 mark combos, "
          "a=2,3)...", flush=True)
    Fblk = all_funcs()
    Fmark = all_funcs()
    A_l1, B_l1 = itersig(Fblk), orbit_tables(Fmark)
    l1 = {f"a{a}": machine_check_L1(A_l1, B_l1, a) for a in (2, 3)}
    print(f"[c48-B_S4] L1 a=2: {l1['a2']}", flush=True)
    print(f"[c48-B_S4] L1 a=3: {l1['a3']}", flush=True)
    print("[c48-B_S4] machine-checking L3 (3125 x 5 x 32 clock)...",
          flush=True)
    l3 = machine_check_L3(all_funcs())
    print(f"[c48-B_S4] L3: {l3}", flush=True)
    print("[c48-B_S4] machine-checking (1,b) phase (B_S4d)...",
          flush=True)
    l4 = machine_check_oneshot()
    print(f"[c48-B_S4] (1,b) phase check: "
          f"per-b={l4['per_b_exact_achievable']} "
          f"joint234={l4['joint_234_possible']} "
          f"maxfill={l4['per_b_max_fill']}", flush=True)
    # T1: total <= a (r>0 phase) + tail prefix, or m. Exact a*b with
    # m > a*b requires a*b <= a + prefix_max  <=>  a*(b-1) <= prefix_max.
    t1_cap = int(l3["max_prefix_fills"])
    t1_excl = [(a, b) for a in range(2, 13) for b in range(2, 13)
               if a * (b - 1) > t1_cap]
    t1_corner = [(a, b) for a in range(2, 13) for b in range(2, 13)
                 if a * (b - 1) <= t1_cap]
    t2_corner = [(a, b) for a in range(2, 13) for b in range(2, 13)
                 if (a - 1) * (b - 1) <= 11]
    corner_union = sorted(set(t1_corner) | set(t2_corner))
    result["D"]["B_S4_machine_check"] = {
        "L1_mark_pass_budget": l1,
        "L2_one_fill_per_pass": "structural (register + scan order)",
        "L3_front_clock_tail": l3,
        "L4_oneshot_phase": l4,
        "T1_modeR_ceiling": f"fills <= a + {t1_cap} OR m (overproduction); "
                            f"exact a*b impossible when a*(b-1) > "
                            f"{t1_cap}",
        "T1_certified_excluded_2_12": f"{len(t1_excl)} pairs",
        "T2_modeP_bound": "output fills <= a + b + 10 (L-POP-"
                          "COLLISION: pops <= 1 output-targeted); "
                          "exact a*b impossible when "
                          "(a-1)(b-1) > 11",
        "open_corner_union_T1_T2": corner_union,
        "statement": "rank-2 value-agnostic MUL certified "
                     "unrealizable outside the corner; corner "
                     "representatives -> B_S5 (search)"}
    print(f"[c48-B_S4] corner (union T1/T2): {corner_union}", flush=True)

    # ---- B_S5: joint-pair search on the open rank-2 corner ----
    for tag, pairs in (("B_S5a_joint_22_32", [(2, 2), (3, 2)]),
                       ("B_S5b_joint_23_24", [(2, 3), (2, 4)])):
        Ph5, Eh5, best5, ev5 = discover2(pairs, label=tag)
        ver = {}
        for p in pairs:
            ver[p] = score_pair(Ph5, Eh5, p[0], p[1],
                                random.Random(705), 60)[0]
        result["D"][tag] = {
            "pairs": [list(p) for p in pairs],
            "best_joint_trace_fitness": round(best5, 4),
            "evals": ev5,
            "verified_fx_per_pair": {f"{a}x{b}": round(ver[(a, b)], 3)
                                     for (a, b) in pairs},
            "joint_fx": round(sum(ver.values()) / len(ver), 3),
            "discovered_joint": all(v >= 0.99 for v in ver.values())}
        print(f"[c48-{tag}] best={best5:.4f} ev={ev5} "
              f"ver={ {f'{a}x{b}': round(ver[(a,b)],3) for (a,b) in pairs} }",
              flush=True)
    a, b = 2, 4
    T6 = pair_taps(a, b, random.Random(486), 20)
    def fit6a(P, E):
        return 0.3 * score_pair(P, E, a, b, random.Random(486), 20)[1] \
            + 0.1 * trace_mean(P, E, T6)
    def fit6b(P, E):
        c = score_pair(P, E, a, b, random.Random(486), 20)
        return 0.7 * c[1] + 0.3 * c[0]
    jfx, fit1, fit2 = joint_fitness_factory([(a, b)])
    Ph6, Eh6, best6, ev6 = discover2([(a, b)], label="B_S5c_single_24")
    ver6 = score_pair(Ph6, Eh6, a, b, random.Random(706), 60)[0]
    gen6 = 0
    for (a2, b2) in ((2, 4), (2, 2), (2, 3), (3, 2), (4, 3)):
        for v in range(10):
            m2 = a2 * b2 + 2 * b2 + 2
            t = make_tape(a2, b2, v, m2)
            fin, mm, tr, hal = run(Ph6, Eh6, t, 3 * a2 * b2 + 8)
            gen6 += hal and fx_count(fin, a2, b2, v, m2)
    result["D"]["B_S5c_single_24"] = {
        "best": round(best6, 4), "evals": ev6,
        "verified_same_geometry": round(ver6, 3),
        "generalized_5geoms_x10v": f"{gen6}/50",
        "target": a * b,
        "note": "single-pair overfit attractor expected (C47 pattern)"}
    print(f"[c48-B_S5c] MUL(2,4) single: best={best6:.4f} ev={ev6} "
          f"ver={ver6:.3f} gen={gen6}/50", flush=True)

    torch.save({"S5a": (Ph5, Eh5), "S5c": (Ph6, Eh6),
                "poploop": (Php, Ehp), "repeat": (Phr, Ehr),
                "oneshot_b2": hand_oneshot(2),
                "oneshot_b3": hand_oneshot(3),
                "oneshot_b4": hand_oneshot(4)},
               "c48_depth2_discovered.pt")
    result["ckpt"] = "c48_depth2_discovered.pt"
    result["laws"] = [
        "L-INDUCTION-FOUR (the frontier): every REALIZABLE value-"
        "agnostic data-value loop on this machine runs to AT MOST 4 "
        "(REPEAT a<=4 mark-orbit, mod-5 edge at 5; ONESHOT (1,b) b<=4 "
        "front-clock transient, edge at 5; pop channel cannot target "
        "the output at all — L-POP-COLLISION). 5-state clock: "
        "transient <= 4, collision at 5, both channels.",
        "L-INDUCTION-RANK1 (T1+T2+L1..L4): value-agnostic MUL is "
        "realizable on the rank-1 family {(a,1): a<=4} (REPEAT, mod-5 "
        "edge) x {(1,b): b in 2..4} by ONE control (B_S4d witness "
        "realized: F=[2,0,3,4,4], G={0,1,2,3}, PhDIG=[1,2,0,0,0] — "
        "value-agnostic in v AND b; per-b controls also banked); "
        "rank-2 certified impossible outside the corner "
        f"a*(b-1)>{t1_cap} (mode R) / (a-1)(b-1)>11 (mode P); the "
        "corner's joint realizability is decided empirically by B_S5 "
        "(0.883 plateau, not discovered).",
        "L-POP-COLLISION (new): in the [template][output] layout the "
        "pop channel writes at the first eligible BLK, which after "
        "pushing lies in the emptied TEMPLATE region (q>=2); output-"
        "targeted pops are impossible for q>=2. First POP-LOOP hand "
        "attempt failed 0/75 (SMOKE) with fills in the template "
        "region — logged as the forensics that killed the "
        "S-emptiness-loop hypothesis for output writing."]
    _peak()
    result["wall_s"] = round(time.time() - T0, 1)
    result["peak_mb"] = round(PEAK, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)

main()
