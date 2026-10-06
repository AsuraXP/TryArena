"""
ARC-2 C29e / P6-LOOP cycle 30: CEGIS-style counterexample-guided repair on
the binding tables (H-C29E). Census runs CERT geometry (nd=2..4 fresh),
localizes the (token,state) cells visited while failing source digits die,
then targeted hill-climb restricted to exactly those cells + BLK rows of
states on failing trajectories. Prior art: CEGIS/MaxSAT fault localization
(MENTOR AAAI; APR timed systems) — transferred to Mealy tables under a
fixpoint contract (novel application). Seed: c29d_repair.pt (best tables).
BARS: S1 in-dist nd<=4 >= 99.5%; S2 nd=16; S3 nd=32; S4 joint nd=64;
S5 passes = nd+1 + one-mark trace. CRISP tables, fitness-contract only.
USAGE: OMP_NUM_THREADS=1 python3 -u c29e_cegis.py   (SMOKE=1 for smoke)
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
R = random.Random(301)


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


def run_full(Ph, Eh, p0h, tape, cap):
    """run to fixpoint/cap; return final tape, passes, trace, per-pass
    (state-before, token) lists aligned with positions."""
    tr = [int((tape == MARK).sum())]
    history = []
    for n in range(1, cap + 1):
        out = tape.clone()
        h = p0h
        visits = []
        for t in range(tape.shape[0]):
            a = int(tape[t])
            visits.append((a, h))
            out[t] = Eh[a, h]
            h = int(Ph[a, h])
        history.append((tape.tolist(), visits))
        if torch.equal(out, tape):
            return out, n, tr, history
        tape = out
        tr.append(int((tape == MARK).sum()))
    return tape, cap, tr, history


def fitness(Ph, Eh, p0h, cases, seed):
    g = random.Random(seed)
    f_halt = f_disc = f_cons = f_slot = f_order = f_sep = 0.0
    visits = {}
    for nd in cases:
        digs = gen_digits(nd, g)
        tape = torch.tensor(make_tape(nd, digs))
        fin, n, tr, _ = run_full(Ph, Eh, p0h, tape, nd + 9)
        f_halt += (n <= nd + 8)
        f_cons += sum(int(fin[src_pos(nd, i)]) == BLK for i in range(nd)) / nd
        f_slot += sum(int(fin[tgt_pos(nd, i)]) == DIG0 + digs[i]
                      for i in range(nd)) / nd
        if len(tr) > 1:
            devs = [min(abs((tr[i - 1] - tr[i]) - 1), 1.0)
                    for i in range(1, len(tr)) if tr[i - 1] > 0]
            f_disc += 1.0 - (sum(devs) / len(devs)) if devs else 1.0
    N = len(cases)
    return (0.10 * f_halt / N + 0.15 * f_disc / N + 0.10 * f_cons / N
            + 0.05 * f_order / N + 0.60 * f_slot / N)


ck = torch.load("c29d_repair.pt")
Ph, Eh, p0h = ck["Ph"].clone(), ck["Eh"].clone(), int(ck["p0h"])

# ---------- PHASE A: cert-geometry failure census ----------
gc = random.Random(3030)
repair_cells = set()
fail_cases = 0
census_cases = 40 if SMOKE else 300
for _ in range(census_cases):
    nd = gc.randrange(2, 5)
    digs = gen_digits(nd, gc)
    fin, n, tr, history = run_full(Ph, Eh, p0h, torch.tensor(make_tape(nd, digs)), nd + 9)
    row = fin.tolist()
    bad = [i for i in range(nd)
           if row[tgt_pos(nd, i)] != DIG0 + digs[i] or row[src_pos(nd, i)] != BLK]
    if not bad:
        continue
    fail_cases += 1
    states_on_fail_traj = set()
    for (tape_row, visits) in history:
        for t, (a, h) in enumerate(visits):
            states_on_fail_traj.add(h)
            # source digit visits at source positions (any pass where still digit)
            if DIG0 <= a < PAD:
                for i in bad:
                    if t == src_pos(nd, i):
                        repair_cells.add((0, a, h))
                        repair_cells.add((1, a, h))
    for h in states_on_fail_traj:
        repair_cells.add((0, BLK, h))
        repair_cells.add((1, BLK, h))
        repair_cells.add((0, SEP, h))
        repair_cells.add((1, SEP, h))
repair_cells = list(repair_cells)
print(f"[c29e] census: {fail_cases}/{census_cases} failing cases, "
      f"{len(repair_cells)} localized repair cells", flush=True)

POOL = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] if not SMOKE else [1, 2, 2]
crng = random.Random(777)


def cases_draw():
    return crng.sample(POOL, len(POOL))


cases = cases_draw()
best = fitness(Ph, Eh, p0h, cases, 0)
print(f"[c29e] seed fitness = {best:.4f}", flush=True)

# ---------- PHASE B: localized hill-climb ----------
budget = 400 if SMOKE else 45000
no_improve_cap = 400 if SMOKE else 16000
edits = accepted = 0
no_improve = 0
while edits < budget and no_improve < no_improve_cap:
    (which, a, h) = R.choice(repair_cells)
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
    sc = fitness(Ph, Eh, p0h, cases, R.randrange(100000))
    edits += 1
    if sc >= best - 1e-9:
        if sc > best + 1e-9:
            if sc - best >= 0.002 or sc >= 0.98:
                print(f"[c29e] edit {edits}: {best:.4f} -> {sc:.4f}", flush=True)
            no_improve = 0
        best = max(best, sc)
        accepted += 1
    else:
        if which == 0:
            Eh[a, h] = old
        else:
            Ph[a, h] = old
        no_improve += 1
print(f"[c29e] done: edits={edits} accepted={accepted} best={best:.4f}", flush=True)
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c29e_cegis.pt")

# ---------- certs ----------
res = {}
certs = dict(indist=(3, 500), n16=(16, 200), n32=(32, 100), joint=(64, 100))
if SMOKE:
    certs = dict(indist=(2, 40))
for name, (nd, n) in certs.items():
    gp = random.Random(2900 + nd)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, gp)
        fin, np_, tr, _ = run_full(Ph, Eh, p0h,
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
out = dict(tag="ARC2-C29E-P6-CEGIS",
           search=dict(edits=edits, accepted=accepted, fitness=round(best, 4),
                       repair_cells=len(repair_cells), fail_cases=fail_cases),
           cert=res, verdict=v,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
