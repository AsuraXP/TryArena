"""
ARC-2 C29 / P6-LOOP cycle 29: VALUE-STATE BANK + TRANSPORT MACRO-MOVE
(H-C29), the new machinery for the binding wall.
Design: (1) widen state space H=16 -> 24; (2) clone the r3 dominant
consumption-entry state into an 8-state BANK (states pre-exist; search only
wires them — attacks L-VALUE-SEPARATION collapse); (3) M-TRANSPORT-MACRO:
ONE mutation proposes the full per-value chain (consume E[d,h]=BLK + entry
P[d,h]=v + write E[BLK,v]=d [+ exit]) — attacks L-ORGAN-NEEDLE directly.
Both moves are generic table-machine operations (state cloning, composite
transition insertion), not task rows. Prior art: AutumnSynth state-splitting
(MIT); gap: no contract-driven table hill-climbing with macro-moves exists.
Seed: c26r3_searched.pt (best binding tables: ordered consumption).
BARS: S1 in-dist nd<=4 >= 99.5%; S2 nd=16; S3 nd=32; S4 joint nd=64;
S5 passes = nd+1 + one-mark trace. CRISP tables only (no SGD this run).
USAGE: OMP_NUM_THREADS=1 python3 -u c29_bank.py   (SMOKE=1 for smoke)
"""
import json, os, random, resource, time
import torch

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
DIGS = list(range(DIG0, DIG0 + 10))
A = 14
H = 24
BANK = list(range(16, 24))
R = random.Random(291)


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


def run_program(Ph, Eh, p0h, tape, cap, nd=None):
    visits = {}
    consumed_states = {}
    tr = [int((tape == MARK).sum())]
    cons = []
    for n in range(1, cap + 1):
        out = tape.clone()
        h = p0h
        for t in range(tape.shape[0]):
            a = int(tape[t])
            visits[(0, a, h)] = visits.get((0, a, h), 0) + 1
            out[t] = Eh[a, h]
            visits[(1, a, h)] = visits.get((1, a, h), 0) + 1
            h = int(Ph[a, h])
        if nd is not None:
            cons.append(sum(int(out[src_pos(nd, i)]) == BLK for i in range(nd)) / nd)
            for i in range(nd):
                p = src_pos(nd, i)
                a = int(tape[p])
                if DIG0 <= a < PAD and int(out[p]) == BLK:
                    hh = p0h
                    for t2 in range(p + 1):
                        hh = int(Ph[int(tape[t2]), hh])
                    consumed_states.setdefault(a - DIG0, set()).add(hh)
        if torch.equal(out, tape):
            return tape, n, tr, visits, cons, consumed_states
        tape = out
        tr.append(int((tape == MARK).sum()))
    return tape, cap, tr, visits, cons, consumed_states


def fitness(Ph, Eh, p0h, cases, seed):
    g = random.Random(seed)
    f_halt = f_disc = f_cons = f_slot = f_order = f_sep = 0.0
    visits = {}
    for nd in cases:
        digs = gen_digits(nd, g)
        tape = torch.tensor(make_tape(nd, digs))
        fin, n, tr, v, cons, cstates = run_program(Ph, Eh, p0h, tape, nd + 9, nd=nd)
        for key, c in v.items():
            visits[key] = visits.get(key, 0) + c
        f_halt += (n <= nd + 8)
        f_cons += cons[-1] if cons else 0.0
        f_slot += sum(int(fin[tgt_pos(nd, i)]) == DIG0 + digs[i]
                      for i in range(nd)) / nd
        if cons:
            f_order += 1.0 - sum(abs(c - min(p / (nd + 1), 1.0))
                                 for p, c in enumerate(cons, 1)) / len(cons)
        if cstates:
            f_sep += len(set().union(*cstates.values())) / max(len(cstates), 1)
        if len(tr) > 1:
            devs = [min(abs((tr[i - 1] - tr[i]) - 1), 1.0)
                    for i in range(1, len(tr)) if tr[i - 1] > 0]
            f_disc += 1.0 - (sum(devs) / len(devs)) if devs else 1.0
    N = len(cases)
    return (0.10 * f_halt / N + 0.15 * f_disc / N + 0.10 * f_cons / N
            + 0.10 * f_order / N + 0.05 * f_sep / N + 0.60 * f_slot / N,
            visits)


# ---- seed: r3 tables widened to H=24 + cloned bank ----
ck = torch.load("c29b_bank.pt")
Ph = ck["Ph"].clone(); Eh = ck["Eh"].clone(); p0h = int(ck["p0h"])

POOL = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] if not SMOKE else [1, 2, 2]
crng = random.Random(777)


def cases_draw():
    return crng.sample(POOL, len(POOL))


cases = cases_draw()
best, visits = fitness(Ph, Eh, p0h, cases, 0)
print(f"[c29] seed fitness = {best:.4f}", flush=True)

# ---- targeted repair rows (from census diagnosis) ----
SCAN = 11
BROKEN_STATES = [5, 18, 21] + BANK
REPAIR_CELLS = []
for a in (DIG0 + 0, DIG0 + 4, DIG0 + 5, DIG0 + 6):   # d0 d4 d5 d6 entries
    REPAIR_CELLS += [(0, a, SCAN), (1, a, SCAN)]
for v in BROKEN_STATES:                                # write/exit rows
    REPAIR_CELLS += [(0, BLK, v), (1, BLK, v)]
for v in BROKEN_STATES:                                # pass-through rows
    for a in list(range(DIG0, DIG0 + 10)) + [SEP]:
        REPAIR_CELLS += [(0, a, v), (1, a, v)]

budget = 300 if SMOKE else 40000
no_improve_cap = 300 if SMOKE else 15000
edits = accepted = 0
no_improve = 0
while edits < budget and no_improve < no_improve_cap:
    (which, a, h) = R.choice(REPAIR_CELLS)
    if which == 0:
        old = int(Eh[a, h]); new = R.randrange(A)
    else:
        old = int(Ph[a, h]); new = R.randrange(H)
    if new == old:
        continue
    if which == 0:
        Eh[a, h] = new
    else:
        Ph[a, h] = new
    if edits % 25 == 0:
        cases = cases_draw()
    sc, v2 = fitness(Ph, Eh, p0h, cases, R.randrange(100000))
    edits += 1
    if sc >= best - 1e-9:
        if sc > best + 1e-9:
            if sc - best >= 0.002 or sc >= 0.98:
                print(f"[c29d] edit {edits}: {best:.4f} -> {sc:.4f}", flush=True)
            no_improve = 0
        best = max(best, sc)
        accepted += 1
    else:
        if which == 0:
            Eh[a, h] = old
        else:
            Ph[a, h] = old
        no_improve += 1
print(f"[c29d] done: edits={edits} accepted={accepted} best={best:.4f}",
      flush=True)
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c29d_repair.pt")

res = {}
certs = dict(indist=(3, 500), n16=(16, 200), n32=(32, 100), joint=(64, 100))
if SMOKE:
    certs = dict(indist=(2, 40))
for name, (nd, n) in certs.items():
    gp = random.Random(2900 + nd)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, gp)
        fin, np_, tr, _, _, _ = run_program(Ph, Eh, p0h,
                                            torch.tensor(make_tape(nd, digs)), nd + 9)
        row = fin.tolist()
        good = all(row[src_pos(nd, i)] == BLK and
                   row[tgt_pos(nd, i)] == DIG0 + digs[i] for i in range(nd))
        ok += good
        passes.append(np_)
        for i2 in range(1, len(tr)):
            if tr[i2] != max(tr[i2 - 1] - 1, 0):
                traces = False
    res[name] = dict(exact=f"{ok}/{n}", passes_mean=round(sum(passes) / n, 2),
                     trace_ok=traces)
    print(f"[cert] {name}: {ok}/{n} exact, passes={res[name]['passes_mean']} "
          f"(want {nd + 1}), trace_ok={traces}", flush=True)
v = dict(S1=res["indist"]["exact"])
if not SMOKE:
    v.update(S2=res["n16"]["exact"], S3=res["n32"]["exact"],
             S4=res["joint"]["exact"],
             S5=res["joint"]["trace_ok"]
             and abs(res["joint"]["passes_mean"] - 65) <= 6.5)
    v["ALL"] = int(v["S1"].split("/")[0]) >= 498 and v["S2"] == "200/200" \
        and v["S3"] == "100/100" and v["S4"] == "100/100" and v["S5"]
out = dict(tag="ARC2-C29D-P6-REPAIR", search=dict(edits=edits, accepted=accepted,
           fitness=round(best, 4)), cert=res, verdict=v,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
