"""
ARC-2 C26r / P6-LOOP run 2: VARIABLE BINDING via STAGED CONTRACT +
INTERLEAVED LAYOUT (H-C26R). Tape: [MARK x nd][SEP][V1 BLK V2 BLK ... Vnd BLK]
[PAD] — target cell ADJACENT to each source digit. Contract: source consumed
to BLK, adjacent target filled with its value (binding move), fixpoint halt.
Pipeline: Stage A search rewards CONSUMPTION only (every consume-edit pays);
gate on consumption >= 0.95; Stage B search adds slot credit; Stage C
crisp-STE SGD refinement from searched tables. Counter organ seeded from
P4-DISC (reuse test 3). Prior art: AGCL subgoal curricula (2304.05271);
Turing Programs "algorithm = iterative copy with local mods" (2407.03310);
gap: none discovers the protocol endogenously / synthesizes crisp tables.
BARS: S1 in-dist nd<=4 >= 99.5%; S2 nd=16; S3 nd=32; S4 joint nd=64;
S5 passes = nd+1 + one-mark trace. NO TF arm (operator directive).
USAGE: OMP_NUM_THREADS=1 python3 -u c26r_staged.py   (SMOKE=1 for smoke)
"""
import json, os, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
DIGS = list(range(DIG0, DIG0 + 10))
A, H = 14, 16
R = random.Random(271)


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


def fitness(Ph, Eh, p0h, cases, seed, w_cons, w_slot):
    g = random.Random(seed)
    f_halt = f_disc = f_cons = f_slot = 0.0
    visits = {}
    for nd in cases:
        digs = gen_digits(nd, g)
        tape = torch.tensor(make_tape(nd, digs))
        fin, n, tr, v = run_program(Ph, Eh, p0h, tape, nd + 9)
        for key, c in v.items():
            visits[key] = visits.get(key, 0) + c
        f_halt += (n <= nd + 8)
        f_cons += sum(int(fin[src_pos(nd, i)]) == BLK for i in range(nd)) / nd
        f_slot += sum(int(fin[tgt_pos(nd, i)]) == DIG0 + digs[i]
                      for i in range(nd)) / nd
        if len(tr) > 1:
            devs = [min(abs((tr[i - 1] - tr[i]) - 1), 1.0)
                    for i in range(1, len(tr)) if tr[i - 1] > 0]
            f_disc += 1.0 - (sum(devs) / len(devs)) if devs else 1.0
    N = len(cases)
    return (0.10 * f_halt / N + 0.15 * f_disc / N + w_cons * f_cons / N
            + w_slot * f_slot / N, visits)


POOL = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] if not SMOKE else [1, 2, 2]
crng = random.Random(777)


def cases_draw():
    return crng.sample(POOL, len(POOL))


ck0 = torch.load("c24d_searched.pt")
Ph, Eh, p0h = ck0["Ph"].clone(), ck0["Eh"].clone(), int(ck0["p0h"])
cases = cases_draw()
best, visits = fitness(Ph, Eh, p0h, cases, 0, 0.75, 0.0)
print(f"[c26r] seed fitness(stage A) = {best:.4f}", flush=True)

# ---------------- Stage A: consumption ----------------
budgetA = 400 if SMOKE else 30000
edits = accepted = 0
no_improve = 0
cons_gate = 0.0
while edits < budgetA and no_improve < (400 if SMOKE else 15000):
    mv = R.random()
    undo = None
    if mv < 0.40:
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
    elif mv < 0.55:
        u, v = R.randrange(H), R.randrange(H)
        if u == v:
            continue
        undo = ("clone", v, Eh[:, v].clone(), Ph[:, v].clone())
        Eh[:, v] = Eh[:, u]; Ph[:, v] = Ph[:, u]
    elif mv < 0.70:
        u, v = R.randrange(H), R.randrange(H)
        if u == v:
            continue
        mask = (Ph == u)
        if not mask.any():
            continue
        undo = ("retarget", mask.clone(), u)
        Ph[mask] = v
    elif mv < 0.85:
        a = R.choice([SEP, MARK, BLK])
        h, new = R.randrange(H), R.randrange(H)
        old = int(Ph[a, h])
        if new == old:
            continue
        Ph[a, h] = new
        undo = ("cell", 1, a, h, old)
    else:
        h = R.randrange(H)
        s = R.choice([1, -1])
        undo = ("block", h, Eh[DIGS, h].clone())
        for d in DIGS:
            Eh[d, h] = DIG0 + ((d - DIG0 + s) % 10)
    if edits % 25 == 0:
        cases = cases_draw()
    sc, v2 = fitness(Ph, Eh, p0h, cases, R.randrange(100000), 0.75, 0.0)
    edits += 1
    if sc >= best - 1e-9:
        if sc > best + 1e-9:
            if sc - best >= 0.005:
                print(f"[c26r] A edit {edits}: {best:.4f} -> {sc:.4f}", flush=True)
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
print(f"[c26r] Stage A done: edits={edits} best={best:.4f}", flush=True)

# ---------------- Stage B: + slot credit ----------------
bestB, visitsB = fitness(Ph, Eh, p0h, cases, 0, 0.15, 0.60)
budgetB = 400 if SMOKE else 60000
editsB = acceptedB = 0
no_improve = 0
while editsB < budgetB and no_improve < (400 if SMOKE else 20000):
    mv = R.random()
    undo = None
    if mv < 0.45:
        (which, a, h) = R.choice(list(visitsB.keys())) if R.random() < 0.5 \
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
    elif mv < 0.60:
        u, v = R.randrange(H), R.randrange(H)
        if u == v:
            continue
        undo = ("clone", v, Eh[:, v].clone(), Ph[:, v].clone())
        Eh[:, v] = Eh[:, u]; Ph[:, v] = Ph[:, u]
    elif mv < 0.72:
        u, v = R.randrange(H), R.randrange(H)
        if u == v:
            continue
        mask = (Ph == u)
        if not mask.any():
            continue
        undo = ("retarget", mask.clone(), u)
        Ph[mask] = v
    elif mv < 0.86:
        a = R.choice([SEP, MARK, BLK])
        h, new = R.randrange(H), R.randrange(H)
        old = int(Ph[a, h])
        if new == old:
            continue
        Ph[a, h] = new
        undo = ("cell", 1, a, h, old)
    else:
        h = R.randrange(H)
        s = R.choice([1, -1])
        undo = ("block", h, Eh[DIGS, h].clone())
        for d in DIGS:
            Eh[d, h] = DIG0 + ((d - DIG0 + s) % 10)
    if editsB % 25 == 0:
        cases = cases_draw()
    sc, v2 = fitness(Ph, Eh, p0h, cases, R.randrange(100000), 0.15, 0.60)
    editsB += 1
    if sc >= bestB - 1e-9:
        if sc > bestB + 1e-9:
            if scB_delta := sc - bestB >= 0.005:
                print(f"[c26r] B edit {editsB}: {bestB:.4f} -> {sc:.4f}",
                      flush=True)
            no_improve = 0
        bestB = max(bestB, sc)
        visitsB = v2
        acceptedB += 1
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
print(f"[c26r] Stage B done: edits={editsB} best={bestB:.4f}", flush=True)
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c26r_searched.pt")

# ---------------- Stage C: crisp-STE SGD refinement ----------------
class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(A, H, H))
        self.E = nn.Parameter(torch.zeros(A, H, A))
        self.p0 = nn.Parameter(torch.zeros(H))

    def one_pass_soft(self, p_tape, B, T):
        Psoft = F.softmax(self.P, -1)
        p = F.softmax(self.p0, -1).unsqueeze(0).expand(B, H).contiguous()
        ph = F.one_hot(p.argmax(-1), H).float()
        p = ph + p - p.detach()
        elos = []
        for t in range(T):
            elos.append(torch.einsum("ba,ahv,bh->bv", p_tape[:, t], self.E, p))
            p = torch.einsum("ba,bh,ahH->bH", p_tape[:, t], p, Psoft)
            ph = F.one_hot(p.argmax(-1), H).float()
            p = ph + p - p.detach()
        return torch.stack(elos, 1)

    def forward_hard(self, tape):
        B, T = tape.shape
        Eh = self.E.argmax(-1)
        Ph = self.P.argmax(-1)
        h = int(self.p0.argmax())
        outs = []
        for t in range(T):
            a = tape[:, t]
            outs.append(Eh[a, h])
            h = Ph[a, h]
        return torch.stack(outs, 1)

    @torch.no_grad()
    def run_fixpoint(self, tape, cap):
        tr = [int((tape == MARK).sum())]
        for n in range(1, cap + 1):
            nxt = self.forward_hard(tape.unsqueeze(0)).squeeze(0)
            if torch.equal(nxt, tape):
                return tape, n, tr
            tape = nxt
            tr.append(int((tape == MARK).sum()))
        return tape, cap, tr


m = Disc()
with torch.no_grad():
    m.P.copy_(F.one_hot(Ph, H).float() * 8 - 1)
    m.E.copy_(F.one_hot(Eh, A).float() * 8 - 1)
    m.p0[p0h] = 8.0
opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
g = random.Random(277)


def gen_batch(batch, ndmax):
    tapes, nds = [], []
    for _ in range(batch):
        nd = g.randrange(1, ndmax + 1)
        tapes.append(make_tape(nd, gen_digits(nd, g)))
        nds.append(nd)
    T = max(len(t) for t in tapes)
    x = torch.full((batch, T), PAD, dtype=torch.long)
    for b, t in enumerate(tapes):
        x[b, :len(t)] = torch.tensor(t)
    return x, nds


def terminal_loss(x, nds):
    B, T = x.shape
    p_tape = F.one_hot(x, A).float()
    ndmax = max(nds)
    elo = None
    for _ in range(ndmax + 1):
        elo = m.one_pass_soft(p_tape, B, T)
        sm = F.softmax(elo, -1)
        hard = F.one_hot(elo.argmax(-1), A).float()
        p_tape = hard + sm - sm.detach()
    y = torch.full((B, T), PAD, dtype=torch.long)
    for b in range(B):
        nd = nds[b]
        row = x[b].tolist()
        y[b, :nd] = SEP
        for i in range(nd):
            y[b, src_pos(nd, i)] = BLK
            y[b, tgt_pos(nd, i)] = row[src_pos(nd, i)]
    msk = (y != PAD)
    return F.cross_entropy(elo[msk], y[msk])


@torch.no_grad()
def probe(ndmax, n=64):
    gp = random.Random(time.time_ns() % 1000003)
    ok = 0
    for _ in range(n):
        nd = gp.randrange(1, ndmax + 1)
        digs = gen_digits(nd, gp)
        tape, np_, _ = m.run_fixpoint(torch.tensor(make_tape(nd, digs)), nd + 9)
        row = tape.tolist()
        good = all(row[src_pos(nd, i)] == BLK and
                   row[tgt_pos(nd, i)] == DIG0 + digs[i] for i in range(nd))
        ok += good
    return ok, n


stages = [1, 2, 3, 4] if not SMOKE else [1, 2]
step = 0
best_ok, best_sd = -1, None
for ndmax in stages:
    stag = 0
    lr = 5e-3
    for g_ in opt.param_groups:
        g_["lr"] = lr
    for s in range(1000 if SMOKE else 4000):
        step += 1
        if s % 1500 == 1499:
            lr = max(lr / 2, 5e-4)
            for g_ in opt.param_groups:
                g_["lr"] = lr
        x, nds = gen_batch(32, ndmax)
        loss = terminal_loss(x, nds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if s % 200 == 0:
            ok, n = probe(ndmax)
            print(f"[c26r] C stage nd<={ndmax} s{s} lr={lr:.5f} "
                  f"CE {loss.item():.4f} crisp-probe {ok}/{n}", flush=True)
            if ok > best_ok:
                best_ok = ok
                best_sd = {k: v.clone() for k, v in m.state_dict().items()}
            if ok >= 60:
                stag = 1
                break
    if not stag:
        print(f"[c26r] C stage nd<={ndmax} did not crystallize", flush=True)
if best_sd is not None:
    m.load_state_dict(best_sd)
    print(f"[c26r] restored best-probe checkpoint ({best_ok}/64)", flush=True)
torch.save(m.state_dict(), "c26r_staged.pt")

# ---------------- certs ----------------
res = {}
certs = dict(indist=(3, 500), n16=(16, 200), n32=(32, 100), joint=(64, 100))
if SMOKE:
    certs = dict(indist=(2, 40))
for name, (nd, n) in certs.items():
    gp = random.Random(2700 + nd)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, gp)
        tape, np_, tr = m.run_fixpoint(torch.tensor(make_tape(nd, digs)), nd + 9)
        row = tape.tolist()
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
out = dict(tag="ARC2-C26R-P6-STAGED",
           search=dict(edits_A=edits, edits_B=editsB,
                       fitA=round(best, 4), fitB=round(bestB, 4)),
           cert=res, verdict=v, steps=step,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
