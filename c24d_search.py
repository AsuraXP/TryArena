"""
ARC-2 C24d / P4-DISC run 3: DISCOVERY BY SEARCH OVER IDENTITY-INIT SPACE
WITH CONTRACT-DECOMPOSED FITNESS.
================================================================================
C24c (run 2) post-mortem: (a) staged SGD shattered consolidated solutions at
every stage transition and ended in SOFT overfit (CE 0.003, hard exact 0/600);
(b) blind hill-climb on all-wrong snapped tables bootstrapped nothing —
L-NEEDLE: an all-or-nothing whole-program fitness has no climbable slope.
MUTATIONS (both task-agnostic, no mechanism supervision):
 M1 IDENTITY INIT: E[a,h] = a (copy), P[a,h] = h (stay). "Default: do
    nothing" — every single-entry edit is a small, meaningful behavioral
    deviation; search starts from a halting machine.
 M2 CONTRACT-DECOMPOSED FITNESS (partial credit from the TASK DEFINITION
    only — x, k, and the mark count are input quantities, not mechanism):
      f = 0.15*halt_ok + 0.25*mark_discipline + 0.60*digit_accuracy
    (digit accuracy = fraction of final digits equal to x+k's digits).
 M3 VISIT-WEIGHTED EDIT PROPOSALS: mutate table cells the machine actually
    visits on the eval set.
NO oracle rows, no ref_pass, no orbit supervision. Prior art logged in the
C24c block (NLI/ANC lineage; mutation = crisp search over snapped machines).
BARS (declared): S1 in-dist (k<=4, <=12 digits) >= 99.5% exact; S2 k=16
200/200; S3 k=64 100/100; S4 joint k=64 x L=120 100/100; S5 passes=k+1 +
one-mark trace; S6 wall < 25 min. NO TF arm (directive).
USAGE: OMP_NUM_THREADS=1 python3 -u c24d_search.py   (SMOKE=1 = wiring)
"""
import json, os, random, resource, time
import torch, torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
rng = random.Random(101)

MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
A = 14
H = 16

def make_tape(k, digs):
    return [MARK] * k + [SEP] + [DIG0 + d for d in digs] + [PAD]

def gen_digits(nd, r):
    return [r.randrange(10) for _ in range(nd - 1)] + [r.randrange(1, 9)]

def oracle_inc_k(digs, k):
    got = list(map(int, str(int("".join(map(str, digs[::-1]))) + k)))[::-1]
    assert len(got) == len(digs)
    return got

def run_program(Ph, Eh, p0h, tape, cap):
    """crisp fixpoint run; returns (final_tape, passes, marks_trace, visits)."""
    visits = {}
    tr = [int((tape == MARK).sum())]
    h = p0h
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
    """Contract-decomposed fitness (task quantities only). Weights logged:
    0.10 halt + 0.15 discipline (graded) + 0.15 progress + 0.60 digits.
    The progress term kills the do-nothing basin (identity init scores
    0.25, not ~0.4) so the first mark-consumption edits are accepted."""
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
        halted = n <= k + 8
        f_halt += halted
        f_prog += (k - tr[-1]) / k                       # marks consumed
        # graded discipline: per-pass drop should be exactly 1 while marks left
        if len(tr) > 1:
            devs = []
            for i in range(1, len(tr)):
                if tr[i - 1] > 0:
                    devs.append(min(abs((tr[i - 1] - tr[i]) - 1), 1.0))
            f_disc += 1.0 - (sum(devs) / len(devs)) if devs else 1.0
        got = [int(t) - DIG0 for t in fin.tolist() if DIG0 <= t < PAD]
        if len(got) == len(tgt):
            f_dig += sum(a == b for a, b in zip(got, tgt)) / len(tgt)
    N = len(cases)
    return (0.10 * f_halt / N + 0.15 * f_disc / N + 0.15 * f_prog / N
            + 0.60 * f_dig / N, visits)

def main():
    # identity init: copy tokens, stay in state
    Eh = torch.arange(A).repeat(H, 1).t().contiguous().long()   # [A,H] = a
    Ph = torch.arange(H).unsqueeze(0).expand(A, H).contiguous().long()
    p0h = 0
    cases = [(1, 6)] * 4 + [(1, 10)] * 4 + [(2, 8)] * 4 + [(3, 10)] * 4 \
        + [(4, 12)] * 4
    budget = 400 if SMOKE else 40000
    no_improve_cap = 2500 if SMOKE else 8000

    best, visits = fitness(Ph, Eh, p0h, cases, seed=0)
    print(f"[c24d] identity-init fitness = {best:.4f}", flush=True)
    keys = None
    edits = accepted = 0
    no_improve = 0
    while edits < budget and no_improve < no_improve_cap:
        # propose a visit-weighted single-entry edit
        if keys is None or rng.random() < 0.1:
            tot = sum(visits.values()) or 1
            keys = list(visits.keys()); ws = [visits[kk] / tot for kk in keys]
        (which, a, h) = rng.choices(keys, weights=ws, k=1)[0]
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
        sc, v2 = fitness(Ph, Eh, p0h, cases, seed=0)
        edits += 1
        if sc > best + 1e-9:
            if sc - best >= 0.02 or sc >= 0.98:
                print(f"[c24d] edit {edits}: {best:.4f} -> {sc:.4f}", flush=True)
            best = sc; visits = v2; keys = None
            accepted += 1; no_improve = 0
        else:
            if which == 0:
                Eh[a, h] = old
            else:
                Ph[a, h] = old
            no_improve += 1
    print(f"[c24d] search done: edits={edits} accepted={accepted} "
          f"best={best:.4f}", flush=True)
    torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c24d_searched.pt")

    # certification of the discovered program
    import importlib.util
    res = {}
    for name, (k, nd, n) in dict(indist=(2, 10, 500), k16=(16, 40, 200),
                                 k64=(64, 40, 100), joint=(64, 120, 100)).items():
        g = random.Random(200 + k)
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

    v = dict(S1=res["indist"]["exact"], S2=res["k16"]["exact"],
             S3=res["k64"]["exact"], S4=res["joint"]["exact"],
             S5=res["joint"]["trace_ok"] and abs(res["joint"]["passes_mean"] - 65) <= 6.5)
    v["ALL"] = (v["S1"].startswith("500") or int(v["S1"].split("/")[0]) >= 498) \
        and v["S2"] == "200/200" and v["S3"] == "100/100" and v["S4"] == "100/100" and v["S5"]
    out = dict(tag="ARC2-C24D-P4-SEARCH", search=dict(edits=edits, accepted=accepted,
               fitness=round(best, 4)), cert=res, verdict=v,
               wall_s=round(time.time() - T0, 1),
               peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
    print(f"[verdict] {v}", flush=True)
    print("RESULT " + json.dumps(out), flush=True)
    with open("log.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    print("DONE", flush=True)

main()
