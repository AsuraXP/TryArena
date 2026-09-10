"""
ARC-2 C26a / P6-LOOP: DISCOVER ITERATED VARIABLE BINDING (move semantics)
on the P4-DISC loop skeleton. Tape: [MARK x nd][SEP][V: nd digits LSB-first]
[SEP][slot: nd BLK][PAD]. Contract: fixpoint; terminal slot = V, V region
consumed to BLK. Expected emergent organ: 1-digit transport states (state
carries the bound value across regions); protocol must be discovered from
the terminal contract alone (identity init, contract-decomposed fitness,
neutral-drift search = c24g playbook).
BARS: S1 in-dist nd<=4 >= 99.5%; S2 nd=16 200/200; S3 nd=32 100/100
(unseen); S4 joint nd=64 100/100; S5 passes = nd+1 exact + one-mark trace.
USAGE: OMP_NUM_THREADS=1 python3 -u c26a_search.py   (SMOKE=1 for smoke)
"""
import json, os, random, resource, time
import torch

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
DIGS = list(range(DIG0, DIG0 + 10))
A, H = 14, 16


def make_tape(nd, digs):
    return [MARK] * nd + [SEP] + [DIG0 + d for d in digs] + [SEP] \
        + [BLK] * nd + [PAD]


def gen_digits(nd, r):
    return [r.randrange(10) for _ in range(nd - 1)] + [r.randrange(1, 9)]


def run_program(Ph, Eh, p0h, tape, cap):
    visits = {}
    tr = [int((tape == MARK).sum())]
    for n in range(1, cap + 1):
        out = tape.clone()
        h = p0h
        for t in range(tape.shape[0]):
            a = int(tape[t])
            visits[(0, a, h)] = visits.get((0, a, h), 0) + 1
            out[t] = Eh[a, h]
            visits[(1, a, h)] = visits.get((1, a, h), 0) + 1
            h = int(Ph[a, h])
        if torch.equal(out, tape):
            return tape, n, tr, visits
        tape = out
        tr.append(int((tape == MARK).sum()))
    return tape, cap, tr, visits


def slots_of(tape, nd):
    """slot digits = last nd non-PAD positions."""
    row = tape.tolist()[:-1]
    return row[-nd:]


def fitness(Ph, Eh, p0h, cases, seed=0):
    g = random.Random(seed)
    f_halt = f_disc = f_prog = f_slot = 0.0
    visits = {}
    for nd in cases:
        digs = gen_digits(nd, g)
        tgt = [DIG0 + d for d in digs]
        tape = torch.tensor(make_tape(nd, digs))
        fin, n, tr, v = run_program(Ph, Eh, p0h, tape, nd + 9)
        for key, c in v.items():
            visits[key] = visits.get(key, 0) + c
        f_halt += (n <= nd + 8)
        f_prog += (nd - tr[-1]) / nd                    # marks consumed
        if len(tr) > 1:
            devs = [min(abs((tr[i - 1] - tr[i]) - 1), 1.0)
                    for i in range(1, len(tr)) if tr[i - 1] > 0]
            f_disc += 1.0 - (sum(devs) / len(devs)) if devs else 1.0
        got = slots_of(fin, nd)
        f_slot += sum(a == b for a, b in zip(got, tgt)) / nd
    N = len(cases)
    return (0.10 * f_halt / N + 0.15 * f_disc / N + 0.15 * f_prog / N
            + 0.60 * f_slot / N, visits)


CASE_POOL = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] if not SMOKE else [1, 2, 2]
_caserng = random.Random(777)

def cases_draw():
    return _caserng.sample(CASE_POOL, len(CASE_POOL))

# seed: counter program discovered in P4-DISC (modular reuse, test 2)
ck0 = torch.load("c24d_searched.pt")
Ph, Eh, p0h = ck0["Ph"].clone(), ck0["Eh"].clone(), int(ck0["p0h"])
cases = cases_draw()
best, visits = fitness(Ph, Eh, p0h, cases)
print(f"[c26c] identity-init fitness = {best:.4f}", flush=True)

budget = 400 if SMOKE else 70000
no_improve_cap = 400 if SMOKE else 22000
edits = accepted = 0
no_improve = 0
last_rep = time.time()
while edits < budget and no_improve < no_improve_cap:
    import random as _r
    mv = _r.random()
    undo = None
    if mv < 0.40:                                     # single-cell edit
        key = list(visits.keys())
        (which, a, h) = _r.choice(key) if _r.random() < 0.5 \
            else (_r.randrange(2), _r.randrange(A), _r.randrange(H))
        if which == 0:
            old = int(Eh[a, h]); new = _r.randrange(A)
        else:
            old = int(Ph[a, h]); new = _r.randrange(H)
        if new == old:
            continue
        if which == 0:
            Eh[a, h] = new
        else:
            Ph[a, h] = new
        undo = ("cell", which, a, h, old)
    elif mv < 0.55:                                   # CLONE state
        u, v = _r.randrange(H), _r.randrange(H)
        if u == v:
            continue
        undo = ("clone", v, Eh[:, v].clone(), Ph[:, v].clone())
        Eh[:, v] = Eh[:, u]; Ph[:, v] = Ph[:, u]
    elif mv < 0.70:                                   # RETARGET u -> v
        u, v = _r.randrange(H), _r.randrange(H)
        if u == v:
            continue
        mask = (Ph == u)
        if not mask.any():
            continue
        undo = ("retarget", mask.clone(), u)
        Ph[mask] = v
    elif mv < 0.85:                                   # delimiter/special edit
        a = _r.choice([SEP, MARK, BLK])
        h, new = _r.randrange(H), _r.randrange(H)
        old = int(Ph[a, h])
        if new == old:
            continue
        Ph[a, h] = new
        undo = ("cell", 1, a, h, old)
    else:                                             # digit block permute
        h = _r.randrange(H)
        s = _r.choice([1, -1])
        undo = ("block", h, Eh[DIGS, h].clone())
        for d in DIGS:
            Eh[d, h] = DIG0 + ((d - DIG0 + s) % 10)
    if edits % 25 == 0:
        cases = cases_draw()
    sc, v2 = fitness(Ph, Eh, p0h, cases)
    edits += 1
    if sc >= best - 1e-9:                             # neutral drift allowed
        if sc > best + 1e-9:
            if sc - best >= 0.005 or sc >= 0.98:
                print(f"[c26c] edit {edits}: {best:.4f} -> {sc:.4f}", flush=True)
            no_improve = 0
        best = max(best, sc)
        visits = v2
        accepted += 1
    else:
        kind = undo[0]
        if kind == "cell":
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
            _, h, eold = undo
            Eh[DIGS, h] = eold
        no_improve += 1
    if time.time() - last_rep > 120:
        print(f"[c26c] ... edits={edits} best={best:.4f} accepted={accepted}",
              flush=True)
        last_rep = time.time()
print(f"[c26c] done: edits={edits} accepted={accepted} best={best:.4f}",
      flush=True)
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c26c_searched.pt")

res = {}
certs = dict(indist=(3, 500), n16=(16, 200), n32=(32, 100), joint=(64, 100))
if SMOKE:
    certs = dict(indist=(2, 30))
for name, (nd, n) in certs.items():
    g = random.Random(2600 + nd)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, g)
        tgt = [DIG0 + d for d in digs]
        fin, np_, tr, _ = run_program(Ph, Eh, p0h,
                                      torch.tensor(make_tape(nd, digs)), nd + 9)
        ok += (slots_of(fin, nd) == tgt)
        passes.append(np_)
        for i in range(1, len(tr)):
            if tr[i] != max(tr[i - 1] - 1, 0):
                traces = False
    res[name] = dict(exact=f"{ok}/{n}", passes_mean=round(sum(passes) / n, 2),
                     trace_ok=traces)
    print(f"[cert] {name}: {ok}/{n} exact, passes={res[name]['passes_mean']} "
          f"(want {nd + 1}), trace_ok={traces}", flush=True)
if SMOKE:
    v = dict(S1=res["indist"]["exact"])
else:
    v = dict(S1=res["indist"]["exact"], S2=res["n16"]["exact"],
             S3=res["n32"]["exact"], S4=res["joint"]["exact"],
             S5=res["joint"]["trace_ok"]
             and abs(res["joint"]["passes_mean"] - 65) <= 6.5)
    v["ALL"] = int(v["S1"].split("/")[0]) >= 498 and v["S2"] == "200/200" \
        and v["S3"] == "100/100" and v["S4"] == "100/100" and v["S5"]
out = dict(tag="ARC2-C26C-P6-SEARCH2", search=dict(edits=edits, accepted=accepted,
           fitness=round(best, 4)), cert=res, verdict=v,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
