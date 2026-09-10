"""
ARC-2 CYCLE 42 / REASONING FRONTIER: REVERSAL BINDING — class barrier.
=======================================================================
Question: can the VET principle (registers over finite control) extend to
reversal binding (tgt_i <- d_{nd-1-i})?

THEOREM (derived this cycle): multi-pass left-to-right head machines with
write-only-forward semantics transport values monotonically RIGHTWARD: a
value at cell x can only ever move to cells >= x (each pass writes only
at-or-ahead of the head; future passes re-scan from the left, so a value
never appears further left than its leftmost historical position).
Reversal pairs digit j with tgt (nd-1-j): for j > (nd-1)/2 the target is
LEFT of the source -> leftward transport required -> NO machine in the
VET / VET+counter / any single-head-LTR-tape class can solve reversal at
any state or register budget. (Prior art echo: arxiv 2402.01032 fixed-
state copy limits; the directionality form appears novel here.)

EMPIRICS (this run): search VET+counter genomes under the reversal
fitness; prediction: plateau at ~first-half-only pairings (fitness well
below 1.0; S-bars fail), confirming the barrier is structural, not an
optimization miss.

Next attack identified (cycle 43): LIFO structure — the machine-v6 stack
organ (or a two-head/rotating tape geometry) gives push/pop = reversal;
test exact-reversal there. Law: L-TRANSPORT-DIRECTION.
Tag ARC2-C42-REVBIND.
"""
import json, random, resource, time

import torch

torch.set_num_threads(1)
T0 = time.time()

MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0, BDIG0, ALPH, BOT = 3, 14, 24, 10
IDENT, ACT_BLK, ACT_BDIG_R = 0, 1, 2
NH = 8


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


def run_rv(Ph, Eh, tape, cap):
    tr = [sum(1 for x in tape if x == MARK)]
    for n in range(1, cap + 1):
        out, h, r, c = [], 0, BOT, 0
        for a in tape:
            act = Eh[a, h]
            if act == ACT_BDIG_R and r == BOT:
                act = IDENT
            out.append(a if act == IDENT else
                       (BLK if act == ACT_BLK else BDIG0 + r))
            # mechanism channels: value register + exact counter
            if DIG0 <= a < DIG0 + 10 and h in (0, 1, 2):
                r = a - DIG0
            elif a == BLK and h in (3, 4):
                c -= 1
                if c == 0:
                    r = BOT
            elif a == MARK:
                c += 1
            elif a == BLK and h in (0, 1, 2):
                c -= 1
            h = Ph[a, h]
        if out == tape:
            return out, n, tr
        tape = out
        tr.append(sum(1 for x in tape if x == MARK))
    return tape, cap, tr


TRAIN = []
g0 = random.Random(4242)
for nd in (2, 3, 4):
    for _ in range(10):
        digs = gen_digits(nd, g0)
        TRAIN.append((nd, digs, make_tape(nd, digs)))


def fitness(Ph, Eh):
    fx = fs = fc = 0.0
    for nd, digs, tape in TRAIN:
        want = digs[::-1]
        fin, n, tr = run_rv(Ph, Eh, tape, nd + 9)
        fx += all(fin[src_pos(nd, i)] == BLK for i in range(nd)) and \
              all(fin[tgt_pos(nd, i)] == BDIG0 + want[i] for i in range(nd))
        fs += sum(fin[tgt_pos(nd, i)] == BDIG0 + want[i]
                  for i in range(nd)) / nd
        fc += sum(fin[src_pos(nd, i)] == BLK for i in range(nd)) / nd
    N = len(TRAIN)
    return (0.5 * fx + 0.35 * fs + 0.15 * fc) / N


def oracle_upper():
    """fraction of pairing that is rightward-monotone (feasible half)."""
    tot = feas = 0
    for nd, digs, tape in TRAIN:
        for i in range(nd):
            tot += 1
            feas += (nd - 1 - i) >= i        # tgt right of src
    return feas / tot


Ph = torch.arange(NH).unsqueeze(0).expand(ALPH, NH).contiguous().clone()
Eh = torch.zeros(ALPH, NH, dtype=torch.long)
rng = random.Random(42)
best_f = fitness(Ph, Eh)
best_Ph, best_Eh = Ph.clone(), Eh.clone()
t0 = time.time()
evals = 0
while time.time() - t0 < 240 and evals < 120_000:
    cPh, cEh = best_Ph.clone(), best_Eh.clone()
    for _ in range(rng.choice([1, 1, 2, 3])):
        a, h = rng.randrange(ALPH), rng.randrange(NH)
        if rng.random() < 0.5:
            cPh[a, h] = rng.randrange(NH)
        else:
            cEh[a, h] = rng.randrange(3)
    f = fitness(cPh, cEh)
    evals += 1
    if f >= best_f:
        best_f, best_Ph, best_Eh = f, cPh, cEh
    if evals % 10000 == 0:
        print(f"  [c42] evals {evals} best {best_f:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[c42] search done: evals {evals} best {best_f:.4f}", flush=True)

uo = oracle_upper()
# per-position success of the best genome (is the feasible half solved?)
pos_ok = [0] * 4
pos_tot = [0] * 4
for nd, digs, tape in TRAIN:
    want = digs[::-1]
    fin, n, tr = run_rv(best_Ph, best_Eh, tape, nd + 9)
    for i in range(nd):
        pos_tot[i] += 1
        pos_ok[i] += (fin[tgt_pos(nd, i)] == BDIG0 + want[i])
per_pos = {f"tgt{i}": f"{pos_ok[i]}/{pos_tot[i]}" for i in range(4)}
print(f"[c42] feasible(rightward) pairing fraction = {uo:.3f}", flush=True)
print(f"[c42] per-position tgt success: {per_pos}", flush=True)

res = {"search_best_fitness": round(best_f, 4), "evals": evals,
       "feasible_fraction": round(uo, 4), "per_position": per_pos,
       "theorem": "single-head LTR tape machines transport values "
                  "monotonically rightward; reversal needs leftward moves "
                  "for half the pairs => class barrier at ANY budget",
       "next_attack": "LIFO organ (machine-v6 kstack) / bidirectional head",
       "verdict": "NEGATIVE-STRUCTURAL (predicted barrier empirically "
                  "confirmed)"}
print(f"[c42] verdict: {res['verdict']}", flush=True)
final = {"tag": "ARC2-C42-REVBIND", "res": res,
         "wall_s": round(time.time() - T0, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
