"""
ARC-2 C31 / P6-LOOP cycle 31: repair v2 on the disciplined c29h tables.
Narrow CEGIS localization: for failing source digits collect ONLY (a) the
E/P cells at visits where that digit was scanned, (b) E[BLK]/P[BLK] of the
states entered by those visits (transport continuation). Fitness keeps the
DEPTH POOL (L-DEPTH-POOL) so S5 is preserved. Prior art: CEGIS fault
localization (MENTOR AAAI'25); mutation = narrower localization + depth-
preserving fitness. NO TF arm (operator directive).
BARS: S1 nd<=4 >=99.5%; S2 nd=16; S3 nd=32; S4 nd=64; S5 passes nd+1 + trace.
USAGE: OMP_NUM_THREADS=1 python3 -u c31_repair2.py   (SMOKE=1 for smoke)
"""
import json, os, random, resource, time
import torch

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0, BDIG0 = 3, 14
A, H = 24, 24
R = random.Random(311)


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
        history.append((tape.tolist(), visits, out.tolist()))
        if torch.equal(out, tape):
            return out, n, tr, history
        tape = out
        tr.append(int((tape == MARK).sum()))
    return tape, cap, tr, history


def fitness(Ph, Eh, p0h, cases, seed):
    g = random.Random(seed)
    f_halt = f_disc = f_cons = f_slot = f_order = 0.0
    for nd in cases:
        digs = gen_digits(nd, g)
        tape = torch.tensor(make_tape(nd, digs))
        fin, n, tr, _ = run_full(Ph, Eh, p0h, tape, nd + 9)
        f_halt += (n <= nd + 8)
        f_cons += sum(int(fin[src_pos(nd, i)]) == BLK for i in range(nd)) / nd
        f_slot += sum(int(fin[tgt_pos(nd, i)]) == BDIG0 + digs[i]
                      for i in range(nd)) / nd
        if len(tr) > 1:
            devs = [min(abs((tr[i - 1] - tr[i]) - 1), 1.0)
                    for i in range(1, len(tr)) if tr[i - 1] > 0]
            f_disc += 1.0 - (sum(devs) / len(devs)) if devs else 1.0
    N = len(cases)
    return (0.10 * f_halt / N + 0.20 * f_disc / N + 0.10 * f_cons / N
            + 0.60 * f_slot / N)


ck = torch.load("c29h_depth.pt")
Ph, Eh, p0h = ck["Ph"].clone(), ck["Eh"].clone(), int(ck["p0h"])

# ---------- census (narrow) ----------
gc = random.Random(3131)
repair_cells = set()
fail_cases = 0
census_cases = 30 if SMOKE else 300
per_digit_fail = [0] * 10
for _ in range(census_cases):
    nd = gc.randrange(2, 5)
    digs = gen_digits(nd, gc)
    fin, n, tr, history = run_full(Ph, Eh, p0h, torch.tensor(make_tape(nd, digs)), nd + 9)
    row = fin.tolist()
    bad = [i for i in range(nd)
           if row[tgt_pos(nd, i)] != BDIG0 + digs[i] or row[src_pos(nd, i)] != BLK]
    if not bad:
        continue
    fail_cases += 1
    for i in bad:
        per_digit_fail[digs[i]] += 1
    for (tape_row, visits, out_row) in history:
        for i in bad:
            p = src_pos(nd, i)
            a0 = tape_row[p]
            if not (DIG0 <= a0 < PAD):
                continue                       # already consumed this pass
            (a, h) = visits[p]
            repair_cells.add((0, a, h))
            repair_cells.add((1, a, h))
            h_next = int(Ph[a, h])
            repair_cells.add((0, BLK, h_next))
            repair_cells.add((1, BLK, h_next))
            repair_cells.add((0, SEP, h_next))
            repair_cells.add((1, SEP, h_next))
repair_cells = list(repair_cells)
print(f"[c32] census: {fail_cases}/{census_cases} failing; per-digit fails "
      f"{per_digit_fail}; {len(repair_cells)} repair cells", flush=True)

POOL = [1, 2, 3, 4, 4, 6, 8, 12] if not SMOKE else [1, 2, 3]
crng = random.Random(777)


def cases_draw():
    return crng.sample(POOL, len(POOL))


cases = cases_draw()
VAL_SEED = 424242                      # frozen held-out gate
val_cases = [2, 3, 3, 4, 4, 2, 3, 4]
best = fitness(Ph, Eh, p0h, cases, 0)
best_val = fitness(Ph, Eh, p0h, val_cases, VAL_SEED)
print(f"[c32] seed fitness = {best:.4f} val = {best_val:.4f}", flush=True)

budget = 400 if SMOKE else 50000
no_improve_cap = 400 if SMOKE else 18000
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
        sv = fitness(Ph, Eh, p0h, val_cases, VAL_SEED)
        if sv < best_val - 1e-9:       # val gate: reject train-only gains
            if which == 0:
                Eh[a, h] = old
            else:
                Ph[a, h] = old
            no_improve += 1
            continue
        if sc > best + 1e-9:
            if sc - best >= 0.002 or sc >= 0.98:
                print(f"[c32] edit {edits}: {best:.4f} -> {sc:.4f} "
                      f"(val {sv:.4f})", flush=True)
            no_improve = 0
        best = max(best, sc)
        best_val = max(best_val, sv)
        accepted += 1
    else:
        if which == 0:
            Eh[a, h] = old
        else:
            Ph[a, h] = old
        no_improve += 1
print(f"[c32] done: edits={edits} accepted={accepted} best={best:.4f}", flush=True)
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c32_split.pt")

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
out = dict(tag="ARC2-C32-P6-SPLIT",
           search=dict(edits=edits, accepted=accepted, fitness=round(best, 4),
                       repair_cells=len(repair_cells), fail_cases=fail_cases,
                       per_digit_fail=per_digit_fail),
           cert=res, verdict=v,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
