"""
ARC-2 CYCLE 44 / C43 ARM-B re-entry: VET+S DISCOVERABILITY — staged
contract-decomposed search.
=======================================================================
C43: reversal CERTIFIED under VET+S (hand-derived control, all bars),
but ARM-B blank-genome hill-climb (C41 protocol, 450s / 27,555 evals)
stalled at train fitness 0.8350 (best genome S1 115/500, S2-S4 0, trace
discipline True). Diagnosis: JOINT NEEDLE — one-mark-per-pass discipline
+ SEP->scan routing + push/pop scan discipline must be found
simultaneously; 1-3 entry mutation width cannot cross the plateau.

ATTACK (c24c P4-DISC precedent: staged discovery curriculum + generic
repair search over snapped tables): partition the control genome into
two CONTRACTS and search each under a stage fitness with prior
contracts FROZEN:
  CONTRACT-M (mark discipline + routing): rows a in {MARK, SEP, PAD}
    x h 0..4 plus BLK x h {0,1}. Stage-1 fitness = fraction of train
    cases with (perfect one-mark trace) AND (all digits pushed, P==nd).
  CONTRACT-S (scan discipline): the remaining rows (DIG/BLK/BDIG/PAD
    x all states, BLK x {0,1} frozen). Stage-2 fitness = full reversal
    fitness (C41 formula, reversal objective).
Then STAGE 3 = BASIN PROFILE: k-entry perturbations of the discovered
genome (k in 1,2,4; 8 perturbations each; 30s re-climb timebox) ->
success rate + evals-to-1.0 (the discoverability basin; C41 analog =
877 evals from blank for VET binding).

PRIOR ART (directive 4, searched before implementing):
  - NNPDA (RNN controller + external stack learns stack control + FSM
    transitions from data; gradient-soft stack, no crisp discrete
    crystallization, small DCF grammars only).
  - NSPDA (arXiv 1909.05233): neural-state pushdown automata, two-stage
    incremental learning, noise regularization; prior knowledge cuts
    convergence ~10x. Validates staged learning of stack machines.
  - DVPA active learning (MFCS 2022, LIPIcs 241): visibly-pushdown
    automata learned via control words + stack-content queries.
  - Hill-climbing construction of finite automata from examples
    (DTIC ADA120123, 8 states, thousands of steps, 14/14) — direct
    prior art for discrete-control hill-climb discovery.
  GAP WE ATTACK: nobody ships a CRISP snapped discrete LIFO tape
  machine whose control is discovered by STAGED CONTRACT search from
  terminal contracts, with length-certified exactness (stretch nd=512).

BARS (C44 acceptance):
  D1 stage-1 fitness_M == 1.0 within 300s
  D2 stage-2 (sub-contracted S.a fc-ramp + S.b pop-cliff + full-S
     rescue) train fitness == 1.0 within 900s
  D3 discovered genome passes the FULL C43 bars (S1 500/500 in-dist,
     S2 200/200 nd=16, S3 100/100 nd=32, S4 100/100 nd=64, S5
     passes=nd+2 + one-mark trace, stretch 128/256/512 exact)
  D4 basin profile at k=1,2,4 (log; not a hard bar)
Estimate (stated before launch): 300s + 900s + 720s + cert < 60s
= ~30 min wall, < 1 GB peak, 1 thread. Hard wall 45 min.
Tag ARC2-C44-VETS-DISC.
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


def run_vs(Ph, Eh, tape, cap, diag=False):
    """VET+S machine (C43 class). diag: also return (P, scan_min) where
    scan_min = min over passes of the fraction of post-SEP cells
    processed in the SCAN state (the per-pass routing invariant the
    pop mechanism depends on — L-CONTRACT-PURITY)."""
    tr = [int(sum(1 for x in tape if x == MARK))]
    S = []
    P = 0
    scan_min = 1.0
    good_passes = 0
    for n in range(1, cap + 1):
        out, h, r = [], 0, BOT
        f, c, s = 0, 0, 0
        seen_sep = False
        post = insc = 0
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
            if a == SEP:
                seen_sep = True
            elif seen_sep:
                s += 1
                post += 1
                if h == SCAN:
                    insc += 1
            if a == MARK:
                c += 1
            if DIG0 <= a < DIG0 + 10 and h == SCAN:
                S.append(a - DIG0)
                P += 1
            h = int(Ph[a, h])
        if post > 0:
            scan_min = min(scan_min, insc / post)
        if post > 0 and insc == post:
            good_passes += 1
        if out == tape:
            if diag:
                return out, n, tr, P, scan_min, good_passes / n
            return out, n, tr
        tape = out
        tr.append(int(sum(1 for x in tape if x == MARK)))
    if diag:
        return tape, cap, tr, P, scan_min, good_passes / cap
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


def trace_ok(tr):
    return tr[-1] == 0 and all(
        tr[i - 1] - tr[i] == 1 for i in range(1, len(tr)) if tr[i - 1] > 0)


def score_all(Ph, Eh):
    """One pass over TRAIN; returns mean component scores
    (trace, m2, fc, fx, fs, ft). Stage fitnesses = weighted sums of
    upstream components (CUMULATIVE — each stage re-scores every
    upstream invariant so downstream stages cannot silently break it;
    L-CONTRACT-PURITY, 3rd instance: the Sa stage drifted
    Ph[DIG,2] != 2, breaking the M2 scan invariant that was frozen)."""
    st = sm = sf = sx = ss = sf_ = 0.0
    for nd, digs, tape in TRAIN:
        want = digs[::-1]
        fin, n, tr, P, scan_min, m2frac = run_vs(Ph, Eh, tape, nd + 9,
                                                 diag=True)
        st += 0.5 * (tr[-1] == 0) + 0.5 * trace_ok(tr)
        sm += m2frac
        sf += sum(fin[src_pos(nd, i)] == BLK for i in range(nd)) / nd
        sx += check(nd, fin, digs)
        ss += sum(fin[tgt_pos(nd, i)] == BDIG0 + want[i]
                  for i in range(nd)) / nd
        sf_ += 0.5 * (tr[-1] == 0 and all(tr[i - 1] - tr[i] in (0, 1)
                                          for i in range(1, len(tr)))) + \
             0.5 * trace_ok(tr)
    N = len(TRAIN)
    return tuple(x / N for x in (st, sm, sf, sx, ss, sf_))


def fitness_full(Ph, Eh):
    st, sm, sf, sx, ss, sf_ = score_all(Ph, Eh)
    return 0.6 * sx + 0.2 * ss + 0.1 * sf + 0.1 * sf_


def fitness_M1(Ph, Eh):
    """CONTRACT-M1 (MARK rows): one-mark-per-pass discipline (graded).
    Phase-4 patch #5: the 17-row M contract (4-entry needle over a
    136-entry space) was still too joint for 1-3 mutation hill-climb
    (27k evals, 0.7000) -> decompose M into M1 (MARK rows: 2-entry
    needle, graded trace feedback) and M2 (SEP rows: 2-entry needle,
    per-pass scan feedback)."""
    return score_all(Ph, Eh)[0]


def fitness_M2(Ph, Eh):
    """CONTRACT-M2 (SEP rows): per-pass scan routing invariant — SEP
    must EXIST and every post-SEP cell be processed in SCAN state in
    EVERY pass (including the fixpoint pass). Cumulative: trace
    re-scored. Phase-4 patch #6: the first M2 fitness (min over
    counted passes) was VACUOUSLY satisfied by a SEP-destroying
    contract (post=0 after pass 1 -> uncounted) — parasitic-solution
    class, caught by the per-pass requirement (L-CONTRACT-PURITY,
    2nd instance)."""
    comps = score_all(Ph, Eh)
    return 0.5 * comps[0] + 0.5 * comps[1]


# ----------------------------- contract partition
M_A = {MARK, SEP, PAD}            # a in M_A: all h in CONTRACT-M
S_A = set(range(ALPH)) - M_A      # DIG/BLK/BDIG: h in 2..4 CONTRACT-S,
                                  # BLK h in {0,1} CONTRACT-M


def is_M_entry(a, h):
    return a in M_A or (a == BLK and h in (0, 1))


rows_M1 = [(MARK, h) for h in range(NH)]
rows_M2 = [(SEP, h) for h in range(NH)]
rows_Sa = [(DIG0 + i, 2) for i in range(10)]
rows_Sb = [(BLK, 2)]
rows_ALL = [(a, h) for a in range(ALPH) for h in range(NH)]


def blank_genome():
    Ph = torch.arange(NH).unsqueeze(0).expand(ALPH, NH).contiguous().clone()
    Eh = torch.zeros(ALPH, NH, dtype=torch.long)
    return Ph, Eh


def mutate(Ph, Eh, rng, n_mut, rows):
    for _ in range(n_mut):
        a, h = rows[rng.randrange(len(rows))]
        if rng.random() < 0.5:
            Ph[a, h] = rng.randrange(NH)
        else:
            Eh[a, h] = rng.randrange(3)


def hill_climb(Ph, Eh, fit, rows, timebox, eval_cap, rng, seed_tag):
    best_f = fit(Ph, Eh)
    best_Ph, best_Eh = Ph.clone(), Eh.clone()
    t0 = time.time()
    evals = 0
    plateau = 0
    while time.time() - t0 < timebox and evals < eval_cap:
        cPh, cEh = best_Ph.clone(), best_Eh.clone()
        mutate(cPh, cEh, rng, rng.choice([1, 1, 2, 3]), rows)
        f = fit(cPh, cEh)
        evals += 1
        if f >= best_f:
            plateau = 0 if f > best_f else plateau + 1
            best_f, best_Ph, best_Eh = f, cPh, cEh
            if evals % 1000 < 2:
                print(f"  [{seed_tag}] evals {evals} best {best_f:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if best_f >= 1.0:
                print(f"[{seed_tag}] PERFECT at evals {evals}", flush=True)
                break
        else:
            plateau += 1
        if plateau > 3000:
            mutate(best_Ph, best_Eh, rng, 4, rows)
            best_f = fit(best_Ph, best_Eh)
            plateau = 0
    return best_Ph, best_Eh, best_f, evals


def certify(Ph, Eh, nd, reps, g_seed):
    g = random.Random(g_seed)
    exact = 0
    trace_ok_ = True
    pass_ok = True
    for _ in range(reps):
        digs = gen_digits(nd, g)
        tape = make_tape(nd, digs)
        fin, n, tr = run_vs(Ph, Eh, tape, nd + 9)
        exact += check(nd, fin, digs)
        pass_ok &= (n == nd + 2)
        trace_ok_ &= trace_ok(tr)
    return exact, reps, pass_ok, trace_ok_


def run_bars(Ph, Eh, s1, s2):
    s1x = (certify(Ph, Eh, 2, 100, s1[0])[0] +
           certify(Ph, Eh, 3, 150, s1[1])[0] +
           certify(Ph, Eh, 4, 250, s1[2])[0])
    s2x = certify(Ph, Eh, 16, 200, s2[0])
    s3x = certify(Ph, Eh, 32, 100, s2[1])
    s4x = certify(Ph, Eh, 64, 100, s2[2])
    spot = []
    for nd in (1, 2, 4, 8, 16, 32, 64):
        g = random.Random(900 + nd)
        _, n, _ = run_vs(Ph, Eh, make_tape(nd, gen_digits(nd, g)), nd + 9)
        spot.append((nd, n, n == nd + 2))
    stretch = []
    for nd in (128, 256, 512):
        exact = 0
        g = random.Random(700 + nd)
        for _ in range(5):
            digs = gen_digits(nd, g)
            fin, n, _ = run_vs(Ph, Eh, make_tape(nd, digs), nd + 9)
            exact += check(nd, fin, digs)
        stretch.append((nd, exact, n))
    res = {"S1_indist": s1x, "S2_n16": s2x[0], "S3_n32": s3x[0],
           "S4_n64": s4x[0], "S5_passes_spot": spot,
           "S5_all_exact": all(x[2] for x in spot),
           "trace_ok": bool(s2x[3] and s3x[3] and s4x[3]),
           "stretch": stretch, "stretch_ok": all(x[1] == 5 for x in stretch)}
    res["bars"] = {"S1_ge_498of500": res["S1_indist"] >= 498,
                   "S2_200of200": res["S2_n16"] == 200,
                   "S3_100of100": res["S3_n32"] == 100,
                   "S4_100of100": res["S4_n64"] == 100,
                   "S5_passes_nd+2": res["S5_all_exact"],
                   "S5_one_mark_trace": res["trace_ok"],
                   "STRETCH_128/256/512": res["stretch_ok"]}
    res["ALL"] = all(res["bars"].values())
    return res


def fitness_Sa(Ph, Eh):
    """CONTRACT-S.a (DIG x state-2 rows): source-clearing ramp with
    ALL upstream invariants re-scored (cumulative). Phase-4 patch #7:
    the fc-only fitness let the ramp drift Ph[DIG,2] != 2 (fc is a
    terminal property — clearing eventually still worked), silently
    breaking the frozen M2 scan invariant -> case-dependent partial
    reversals (best-pop 0.5933). Cumulative scoring rejects any
    candidate that breaks an upstream invariant (L-CONTRACT-PURITY,
    3rd instance)."""
    comps = score_all(Ph, Eh)
    return 0.5 * comps[0] + 0.25 * comps[1] + 0.25 * comps[2]


# ----------------------------- SMOKE wiring
if os.environ.get("SMOKE") == "1":
    # reference genome (C43 hand-derived) must score 1.0 on all stages
    PhA = torch.zeros(ALPH, NH, dtype=torch.long)
    for a in range(ALPH):
        for h in range(NH):
            PhA[a, h] = h
    PhA[MARK, 0] = 1
    PhA[SEP, 0] = 2
    PhA[SEP, 1] = 2
    PhA[SEP, 2] = 2
    EhA = torch.zeros(ALPH, NH, dtype=torch.long)
    EhA[MARK, 0] = ACT_BLK
    EhA[DIG0:DIG0 + 10, 2] = ACT_BLK
    EhA[BLK, 2] = ACT_COND_R
    print(f"[c44-smoke] hand genome: M1={fitness_M1(PhA, EhA):.3f} "
          f"M2={fitness_M2(PhA, EhA):.3f} Sa={fitness_Sa(PhA, EhA):.3f} "
          f"full={fitness_full(PhA, EhA):.3f} (all must be 1.0)",
          flush=True)
    total = ALPH * NH
    print(f"[c44-smoke] partition: M1 {len(rows_M1)} rows, M2 "
          f"{len(rows_M2)}, Sa {len(rows_Sa)}, Sb {len(rows_Sb)}, "
          f"total entries {total}", flush=True)
    # mini stage-1a from blank: 200 evals must reach 1.0
    rng = random.Random(0)
    Ph, Eh = blank_genome()
    Ph, Eh, f, ev = hill_climb(Ph, Eh, fitness_M1, rows_M1, 30, 200, rng,
                               "smoke1")
    print(f"[c44-smoke] mini stage-1a: fitness_M1={f:.3f} after {ev} "
          f"evals (must be 1.0)", flush=True)
    print("SMOKE-DONE", flush=True)
    raise SystemExit(0)


# ----------------------------- STAGE 1: CONTRACT-M1 (MARK rows)
rng = random.Random(44)
Ph, Eh = blank_genome()
print(f"[c44] STAGE 1a (CONTRACT-M1, {len(rows_M1)} rows): "
      f"blank fitness_M1={fitness_M1(Ph, Eh):.3f}", flush=True)
Ph, Eh, fM1, evM1 = hill_climb(Ph, Eh, fitness_M1, rows_M1, 180, 100_000,
                               rng, "c44-s1a")
print(f"[c44] STAGE 1a done: fitness_M1={fM1:.4f} evals={evM1} "
      f"({time.time() - T0:.0f}s)", flush=True)
print(f"[c44] STAGE 1b (CONTRACT-M2, {len(rows_M2)} rows): "
      f"fitness_M2={fitness_M2(Ph, Eh):.4f}", flush=True)
Ph, Eh, fM2, evM2 = hill_climb(Ph, Eh, fitness_M2, rows_M2, 180, 100_000,
                               rng, "c44-s1b")
print(f"[c44] STAGE 1b done: fitness_M2={fM2:.4f} evals={evM2} "
      f"({time.time() - T0:.0f}s)", flush=True)

# ----------------------------- STAGE 2 (SUB-CONTRACTED; Phase-4 patch).
# First stage-2 attempt (joint 103-row search, full fitness) stalled on
# the 10-entry DIG-clearing ramp (0.10 -> 0.2389 @ 353s, restarts
# degrading). Patch = decompose S into S.a (DIG x state-2 rows, fc
# fitness — pure monotone ramp) and S.b (BLK x state-2 row, full
# fitness — 1-entry pop cliff), with a full-S rescue search if S.b
# fails (c24c E2 general-repair precedent).
print(f"[c44] STAGE 2a (CONTRACT-S.a, {len(rows_Sa)} rows, M+S.b frozen): "
      f"fitness_Sa={fitness_Sa(Ph, Eh):.4f} full={fitness_full(Ph, Eh):.4f}",
      flush=True)
Ph, Eh, fSa, evSa = hill_climb(Ph, Eh, fitness_Sa, rows_Sa, 300, 200_000,
                               rng, "c44-s2a")
print(f"[c44] STAGE 2a done: fitness_Sa={fSa:.4f} evals={evSa} "
      f"({time.time() - T0:.0f}s)", flush=True)
print(f"[c44] STAGE 2b (CONTRACT-S.b, {len(rows_Sb)} rows): "
      f"full={fitness_full(Ph, Eh):.4f}", flush=True)
Ph, Eh, fSb, evSb = hill_climb(Ph, Eh, fitness_full, rows_Sb, 120, 50_000,
                               rng, "c44-s2b")
print(f"[c44] STAGE 2b done: full={fSb:.4f} evals={evSb} "
      f"({time.time() - T0:.0f}s)", flush=True)
fS = fSb
evS = evSa + evSb
if fS < 1.0:
    print(f"[c44] STAGE 2 RESCUE (full genome, full fitness, 300s)",
          flush=True)
    Ph, Eh, fS, evR = hill_climb(Ph, Eh, fitness_full, rows_ALL, 300,
                                 200_000, rng, "c44-s2resc")
    evS += evR
    print(f"[c44] STAGE 2 rescue done: full={fS:.4f} evals={evR} "
          f"({time.time() - T0:.0f}s)", flush=True)
torch.save({"Ph": Ph, "Eh": Eh}, "c44_vets_discovered.pt")

# ----------------------------- D3: full bars on discovered genome
resD = run_bars(Ph, Eh, (641, 642, 643), (644, 645, 646))
print(f"[c44] D3 discovered-genome bars: {resD['bars']} "
      f"S1 {resD['S1_indist']}/500 S2 {resD['S2_n16']}/200 "
      f"S3 {resD['S3_n32']}/100 S4 {resD['S4_n64']}/100 "
      f"stretch {resD['stretch']}", flush=True)

# ----------------------------- STAGE 3: basin profile
basin = {}
for k in (1, 2, 4):
    succ, evs_list = 0, []
    for p in range(8):
        cPh, cEh = Ph.clone(), Eh.clone()
        mutate(cPh, cEh, rng, k, rows_ALL)
        _, _, f_back, ev_back = hill_climb(cPh, cEh, fitness_full,
                                           rows_ALL, 30, 3_000, rng,
                                           f"c44-basin-k{k}-p{p}")
        if f_back >= 1.0:
            succ += 1
            evs_list.append(ev_back)
    basin[f"k{k}"] = {"succ": succ, "of": 8,
                      "evals_to_1": evs_list}
    print(f"[c44] basin k={k}: {succ}/8 back to 1.0, "
          f"evals {evs_list} ({time.time() - T0:.0f}s)", flush=True)

final = {
    "tag": "ARC2-C44-VETS-DISC",
    "method": "staged contract-decomposed hill-climb (c24c P4-DISC "
              "precedent) over VET+S control tables",
    "D1_stage1": {"M1_trace": round(fM1, 4), "M1_evals": evM1,
                  "M2_scan": round(fM2, 4), "M2_evals": evM2,
                  "pass": fM1 >= 1.0 and fM2 >= 1.0},
    "D2_stage2": {"train_fitness": round(fS, 4), "evals": evS,
                  "stage2a_sa": round(fSa, 4), "stage2a_evals": evSa,
                  "pass": fS >= 1.0},
    "D3_bars": {k: resD[k] for k in ("S1_indist", "S2_n16", "S3_n32",
                                     "S4_n64", "S5_all_exact", "trace_ok",
                                     "stretch_ok", "bars", "ALL")},
    "D4_basin_profile": basin,
    "ckpt": "c44_vets_discovered.pt",
    "wall_s": round(time.time() - T0, 1),
    "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
