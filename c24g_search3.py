"""
ARC-2 C24g / P4-DISC run 5b: STATE-MACHINE SEARCH (clone/retarget/neutral
drift) seeded from the discovered counter program.
================================================================================
C24e post-mortem: (i) strict-improvement acceptance blocked NEUTRAL bridge
edits (e.g. a phase-transition P[SEP,h]=h' is fitness-neutral but opens the
digit phase to specialization); (ii) no clone/retarget moves, so new
functional states could not be created; (iii) block-shift moves implemented
the wrong algorithm family (per-pass digitwise shift != number +1).
Mutations M5: moves = single-cell | state CLONE (copy all rows u->v) |
RETARGET (redirect incoming transitions) | delimiter phase edits | block
shifts; acceptance = sc >= best (neutral drift allowed, revert only on
worse). All moves generic machine-edit operations; no task rows.
BARS: S1 in-dist >= 99.5%; S2 k=16; S3 k=64; S4 joint k=64 x L=120;
S5 passes=k+1 + trace. Wall < 25 min. NO TF arm.
USAGE: OMP_NUM_THREADS=1 python3 -u c24g_search3.py
"""
import json, os, random, resource, time
import torch

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
rng = random.Random(139)
MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
DIGS = list(range(DIG0, DIG0 + 10))
A, H = 14, 16

def make_tape(k, digs):
    return [MARK] * k + [SEP] + [DIG0 + d for d in digs] + [PAD]

def gen_digits(nd, r):
    return [r.randrange(10) for _ in range(nd - 1)] + [r.randrange(1, 9)]

def oracle_inc_k(digs, k):
    got = list(map(int, str(int("".join(map(str, digs[::-1]))) + k)))[::-1]
    assert len(got) == len(digs)
    return got

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

def fitness(Ph, Eh, p0h, cases, seed=0):
    g = random.Random(seed)
    f_halt = f_disc = f_prog = f_dig = 0.0
    visits = {}
    for (k, nd) in cases:
        digs = gen_digits(nd, g)
        tgt = oracle_inc_k(digs, k)
        tape = torch.tensor(make_tape(k, digs))
        fin, n, tr, v = run_program(Ph, Eh, p0h, tape, k + 9)
        for key, c in v.items():
            visits[key] = visits.get(key, 0) + c
        f_halt += (n <= k + 8)
        f_prog += (k - tr[-1]) / k
        if len(tr) > 1:
            devs = [min(abs((tr[i - 1] - tr[i]) - 1), 1.0)
                    for i in range(1, len(tr)) if tr[i - 1] > 0]
            f_disc += 1.0 - (sum(devs) / len(devs)) if devs else 1.0
        got = [int(t) - DIG0 for t in fin.tolist() if DIG0 <= t < PAD]
        if len(got) == len(tgt):
            f_dig += sum(a == b for a, b in zip(got, tgt)) / len(tgt)
    N = len(cases)
    return (0.10 * f_halt / N + 0.15 * f_disc / N + 0.15 * f_prog / N
            + 0.60 * f_dig / N, visits)

cases = [(1, 6)] * 4 + [(1, 10)] * 4 + [(2, 8)] * 4 + [(3, 10)] * 4 \
    + [(4, 12)] * 4

ck = torch.load("c24d_searched.pt")
Ph, Eh, p0h = ck["Ph"].clone(), ck["Eh"].clone(), int(ck["p0h"])
best, visits = fitness(Ph, Eh, p0h, cases)
print(f"[c24g] resumed fitness = {best:.4f}", flush=True)

budget = 300 if SMOKE else 45000
no_improve_cap = 300 if SMOKE else 14000
edits = accepted = 0
no_improve = 0
while edits < budget and no_improve < no_improve_cap:
    move = rng.random()
    undo = None
    if move < 0.40:                                   # single-cell edit
        (which, a, h) = rng.choice(list(visits.keys())) if rng.random() < 0.5 \
            else (rng.randrange(2), rng.randrange(A), rng.randrange(H))
        if which == 0:
            old = int(Eh[a, h]); new = rng.randrange(A)
        else:
            old = int(Ph[a, h]); new = rng.randrange(H)
        if new == old:
            continue
        if which == 0:
            Eh[a, h] = new
        else:
            Ph[a, h] = new
        undo = ("cell", which, a, h, old)
    elif move < 0.55:                                 # CLONE state u -> v
        u, v = rng.randrange(H), rng.randrange(H)
        if u == v:
            continue
        undo = ("clone", v, Eh[:, v].clone(), Ph[:, v].clone())
        Eh[:, v] = Eh[:, u]; Ph[:, v] = Ph[:, u]
    elif move < 0.70:                                 # RETARGET u -> v
        u, v = rng.randrange(H), rng.randrange(H)
        if u == v:
            continue
        mask = (Ph == u)
        if not mask.any():
            continue
        undo = ("retarget", mask.clone(), u)
        Ph[mask] = v
    elif move < 0.85:                                 # delimiter phase edit
        a = rng.choice([SEP, MARK, BLK])
        h, new = rng.randrange(H), rng.randrange(H)
        old = int(Ph[a, h])
        if new == old:
            continue
        Ph[a, h] = new
        undo = ("cell", 1, a, h, old)
    else:                                             # digit block shift
        h = rng.randrange(H)
        s = rng.choice([1, -1])
        undo = ("block", h, Eh[DIGS, h].clone())
        for d in DIGS:
            Eh[d, h] = DIG0 + ((d - DIG0 + s) % 10)
    sc, v2 = fitness(Ph, Eh, p0h, cases)
    edits += 1
    if sc >= best - 1e-9:                             # neutral drift allowed
        if sc > best + 1e-9:
            if sc - best >= 0.005 or sc >= 0.98:
                print(f"[c24g] edit {edits}: {best:.4f} -> {sc:.4f}", flush=True)
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
print(f"[c24g] done: edits={edits} accepted={accepted} best={best:.4f}", flush=True)
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c24g_searched.pt")

res = {}
for name, (k, nd, n) in dict(indist=(2, 10, 500), k16=(16, 40, 200),
                             k64=(64, 40, 100), joint=(64, 120, 100)).items():
    g = random.Random(500 + k)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, g)
        fin, np_, tr, _ = run_program(Ph, Eh, p0h,
                                      torch.tensor(make_tape(k, digs)), k + 9)
        got = [int(t) - DIG0 for t in fin.tolist() if DIG0 <= t < PAD]
        ok += (got == oracle_inc_k(digs, k))
        passes.append(np_)
        for i in range(1, len(tr)):
            if tr[i] != max(tr[i - 1] - 1, 0):
                traces = False
    res[name] = dict(exact=f"{ok}/{n}", passes_mean=round(sum(passes) / n, 2),
                     trace_ok=traces)
    print(f"[cert] {name}: {ok}/{n} exact, passes={res[name]['passes_mean']} "
          f"(want {k + 1}), trace_ok={traces}", flush=True)
v = dict(S1=res["indist"]["exact"], S2=res["k16"]["exact"], S3=res["k64"]["exact"],
         S4=res["joint"]["exact"],
         S5=res["joint"]["trace_ok"] and abs(res["joint"]["passes_mean"] - 65) <= 6.5)
v["ALL"] = int(v["S1"].split("/")[0]) >= 498 and v["S2"] == "200/200" \
    and v["S3"] == "100/100" and v["S4"] == "100/100" and v["S5"]
out = dict(tag="ARC2-C24G-P4-SEARCH3", search=dict(edits=edits, accepted=accepted,
           fitness=round(best, 4)), cert=res, verdict=v,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
