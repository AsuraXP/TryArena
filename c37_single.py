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
BDIGS = list(range(14, 24))
A = 24
H = 24
BDIG0 = 14
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
    cons_incs = []
    prev_consumed = 0
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
        cur_consumed = sum(int(out[src_pos(nd, i)]) == BLK for i in range(nd)) \
            if nd is not None else 0
        cons_incs.append(cur_consumed - prev_consumed)
        prev_consumed = cur_consumed
        if torch.equal(out, tape):
            return tape, n, tr, visits, cons, consumed_states, cons_incs
        tape = out
        tr.append(int((tape == MARK).sum()))
    return tape, cap, tr, visits, cons, consumed_states, cons_incs


def fitness(Ph, Eh, p0h, cases, seed):
    g = random.Random(seed)
    f_halt = f_disc = f_cons = f_slot = f_order = f_sep = f_single = 0.0
    visits = {}
    for nd in cases:
        digs = gen_digits(nd, g)
        tape = torch.tensor(make_tape(nd, digs))
        fin, n, tr, v, cons, cstates, cinc = run_program(Ph, Eh, p0h, tape, nd + 9, nd=nd)
        for key, c in v.items():
            visits[key] = visits.get(key, 0) + c
        f_halt += (n <= nd + 8)
        f_cons += cons[-1] if cons else 0.0
        if cinc:
            f_single += sum(1.0 if c <= 1 else max(0.0, 1.0 - (c - 1))
                            for c in cinc) / len(cinc)
        f_slot += sum(int(fin[tgt_pos(nd, i)]) == BDIG0 + digs[i]
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
            + 0.55 * f_slot / N + 0.10 * f_single / N, visits)


# ---- seed: r3 tables widened to H=24 + cloned bank ----
ck = torch.load("c24d_searched.pt")  # CLEAN counter seed
Ph = torch.zeros(A, H, dtype=torch.long)
Eh = torch.zeros(A, H, dtype=torch.long)
for a in range(A):                          # identity defaults everywhere
    Eh[a, :] = a
    Ph[a, :] = torch.arange(H)
Ph[:14, :16] = ck["Ph"]
Eh[:14, :16] = ck["Eh"]
for h in range(16, H):                     # bank: identity rows
    Ph[:, h] = torch.arange(H)
    Eh[:, h] = torch.arange(A)
p0h = int(ck["p0h"])

# find dominant consumption-entry state in seed, clone it into the bank
cases0 = [2, 3, 3, 4, 4, 4]
g0 = random.Random(7)
entry_count = {}
for nd in cases0:
    digs = gen_digits(nd, g0)
    _, _, _, _, _, cs, _ = run_program(Ph, Eh, p0h, torch.tensor(make_tape(nd, digs)),
                                    nd + 9, nd=nd)
    for d, sts in cs.items():
        for s in sts:
            entry_count[s] = entry_count.get(s, 0) + 1
if entry_count:
    s_star = max(entry_count, key=entry_count.get)
else:
    s_star = 11
for bk in BANK:
    Ph[:, bk] = Ph[:, s_star].clone()
    Eh[:, bk] = Eh[:, s_star].clone()
print(f"[c37] bank cloned from entry state h{s_star}: {entry_count}", flush=True)

POOL = [1, 2, 3, 4, 4, 6, 8, 12] if not SMOKE else [1, 2, 3]
crng = random.Random(777)


def cases_draw():
    return crng.sample(POOL, len(POOL))


cases = cases_draw()
best, visits = fitness(Ph, Eh, p0h, cases, 0)
print(f"[c37] seed fitness = {best:.4f}", flush=True)

budget = 400 if SMOKE else 70000
no_improve_cap = 400 if SMOKE else 25000
edits = accepted = macros = 0
no_improve = 0
while edits < budget and no_improve < no_improve_cap:
    mv = R.random()
    undo = None
    if mv < 0.30:                                   # transport MACRO move
        d = DIG0 + R.randrange(10)
        digit_keys = [k for k in visits if k[1] == d and k[0] == 0]
        if digit_keys and R.random() < 0.7:
            (_, _, h) = R.choice(digit_keys)
        else:
            h = R.randrange(16)
        v = R.choice(BANK)
        cells = [(0, d, h, int(Eh[d, h])), (1, d, h, int(Ph[d, h])),
                 (0, BLK, v, int(Eh[BLK, v]))]
        if R.random() < 0.5:
            cells.append((1, BLK, v, int(Ph[BLK, v])))
        for (which, a, hh, old) in cells:
            if which == 0:
                Eh[a, hh] = BLK if a == d else BDIG0 + (d - DIG0)
            else:
                Ph[a, hh] = v if a == d else 13
        undo = ("macro", cells)
        macros += 1
    elif mv < 0.60:                                 # single-cell edit
        (which, a, h) = R.choice(list(visits.keys())) if R.random() < 0.5 \
            else (R.randrange(2), R.randrange(A), R.randrange(H))
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
        undo = ("cell", which, a, h, old)
    elif mv < 0.72:                                 # clone
        u, v = R.randrange(H), R.randrange(H)
        if u == v:
            continue
        undo = ("clone", v, Eh[:, v].clone(), Ph[:, v].clone())
        Eh[:, v] = Eh[:, u]; Ph[:, v] = Ph[:, u]
    elif mv < 0.84:                                 # retarget
        u, v = R.randrange(H), R.randrange(H)
        if u == v:
            continue
        mask = (Ph == u)
        if not mask.any():
            continue
        undo = ("retarget", mask.clone(), u)
        Ph[mask] = v
    elif mv < 0.92:                                 # delimiter edit
        a = R.choice([SEP, MARK, BLK])
        h, new = R.randrange(H), R.randrange(H)
        old = int(Ph[a, h])
        if new == old:
            continue
        Ph[a, h] = new
        undo = ("cell", 1, a, h, old)
    else:                                           # digit block shift
        h = R.randrange(H)
        s = R.choice([1, -1])
        fam = R.choice([DIGS, BDIGS])
        base = DIG0 if fam is DIGS else BDIG0
        undo = ("block", h, Eh[fam, h].clone(), fam)
        for d0 in fam:
            Eh[d0, h] = base + ((d0 - base + s) % 10)
    if edits % 25 == 0:
        cases = cases_draw()
    sc, v2 = fitness(Ph, Eh, p0h, cases, R.randrange(100000))
    edits += 1
    if sc >= best - 1e-9:
        if sc > best + 1e-9:
            if sc - best >= 0.005 or sc >= 0.98:
                print(f"[c37] edit {edits} (macros {macros}): {best:.4f} -> "
                      f"{sc:.4f}", flush=True)
            no_improve = 0
        best = max(best, sc)
        visits = v2
        accepted += 1
    else:
        kind = undo[0]
        if kind == "macro":
            for (which, a, hh, old) in undo[1]:
                if which == 0:
                    Eh[a, hh] = old
                else:
                    Ph[a, hh] = old
        elif kind == "cell":
            _, which, a, h, old = undo
            if which == 0:
                Eh[a, h] = old
            else:
                Ph[a, h] = old
        elif kind == "clone":
            _, v, eold, pold = undo
            Eh[:, v] = eold; Ph[:, v] = pold
        elif kind == "retarget":
            _, mask, u = undo
            Ph[mask] = u
        elif kind == "block":
            _, h, eold, fam = undo
            Eh[fam, h] = eold
        no_improve += 1
print(f"[c37] done: edits={edits} accepted={accepted} macros={macros} "
      f"best={best:.4f}", flush=True)
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c37_single.pt")

res = {}
certs = dict(indist=(3, 500), n16=(16, 200), n32=(32, 100), joint=(64, 100))
if SMOKE:
    certs = dict(indist=(2, 40))
for name, (nd, n) in certs.items():
    gp = random.Random(2900 + nd)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, gp)
        fin, np_, tr, _, _, _, _ = run_program(Ph, Eh, p0h,
                                            torch.tensor(make_tape(nd, digs)), nd + 9)
        row = fin.tolist()
        good = all(row[src_pos(nd, i)] == BLK and
                   row[tgt_pos(nd, i)] == BDIG0 + digs[i] for i in range(nd))
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
out = dict(tag="ARC2-C37-P6-SINGLE", search=dict(edits=edits, accepted=accepted,
           macros=macros, fitness=round(best, 4)), cert=res, verdict=v,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
