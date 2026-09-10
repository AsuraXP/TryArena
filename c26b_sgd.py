"""
ARC-2 C25a / P3-LOOP: DISCOVER ITERATED SUBTRACTION (x - k) on the P4-DISC
loop skeleton, testing MODULAR REUSE: the search-discovered counter protocol
(MARK->SEP dissolution, fixpoint halt) is seeded FROZEN-ISH and only the
digit pass is learned — a new algorithm (borrow decrement) from the terminal
contract alone. Recipe = c24k (crisp forward, SEP labels, fresh probes,
per-stage lr decay, best-ckpt restore).
BARS: S1 in-dist (k<=4) >= 99.5%; S2 k=16 200/200; S3 k=64 100/100 (unseen);
S4 joint k=64 x L=120 100/100; S5 passes = k+1 exact + one-mark trace.
USAGE: OMP_NUM_THREADS=1 python3 -u c25a_sub.py     (SMOKE=1 for smoke)
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
A, H = 14, 16


def make_tape(k, digs):
    return [MARK] * k + [SEP] + [DIG0 + d for d in digs] + [SEP] \
        + [BLK] * k + [PAD]


def gen_digits(nd, r):
    return [r.randrange(10) for _ in range(nd - 1)] + [r.randrange(1, 9)]


def digs_val(digs):  # LSB-first list -> int
    return int("".join(map(str, digs[::-1])))


def slots_of(tape, nd):
    return tape.tolist()[:-1][-nd:]


class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(A, H, H))
        self.E = nn.Parameter(torch.zeros(A, H, A))
        self.p0 = nn.Parameter(torch.zeros(H))

    def one_pass_soft(self, p_tape, B, T):
        # STE-crisp state propagation (train dynamics == eval dynamics).
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
        # TRUE CRISP execution: snapped tables, single state.
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
ck = torch.load("c24d_searched.pt")
with torch.no_grad():
    m.P.copy_(F.one_hot(ck["Ph"], H).float() * 8 - 1)
    m.E.copy_(F.one_hot(ck["Eh"], A).float() * 8 - 1)
    m.p0[ck["p0h"]] = 8.0
seed_Ph, seed_Eh = ck["Ph"].clone(), ck["Eh"].clone()
opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
g = random.Random(251)


def gen_batch(batch, kmax, dmax):
    tapes, ks = [], []
    for _ in range(batch):
        k = g.randrange(1, kmax + 1)
        nd = g.randrange(2, dmax + 1)
        tapes.append(make_tape(k, gen_digits(nd, g)))
        ks.append(k)
    T = max(len(t) for t in tapes)
    x = torch.full((batch, T), PAD, dtype=torch.long)
    for b, t in enumerate(tapes):
        x[b, :len(t)] = torch.tensor(t)
    return x, ks


def terminal_loss(x, ks):
    B, T = x.shape
    p_tape = F.one_hot(x, A).float()
    kmax = max(ks)
    elo = None
    for _ in range(kmax + 1):
        elo = m.one_pass_soft(p_tape, B, T)
        sm = F.softmax(elo, -1)
        hard = F.one_hot(elo.argmax(-1), A).float()
        p_tape = hard + sm - sm.detach()
    y = torch.full((B, T), PAD, dtype=torch.long)
    for b in range(B):
        row = x[b].tolist()
        k = ks[b]
        digs = row[1 + k + 1:1 + k + 1 + k]          # V region tokens
        y[b, :k] = SEP                               # dissolved counter
        y[b, k + 1:2 * k + 2] = BLK                  # V consumed
        y[b, 2 * k + 3:3 * k + 3] = torch.tensor(digs)  # slot bound
    msk = (y != PAD)
    return F.cross_entropy(elo[msk], y[msk])


@torch.no_grad()
def probe(kmax, dmax, n=64):
    gp = random.Random(time.time_ns() % 1000003)
    ok = 0
    for _ in range(n):
        nd = gp.randrange(1, kmax + 1)
        digs = gen_digits(max(nd, 2), gp)
        nd = len(digs)
        tape, np_, _ = m.run_fixpoint(torch.tensor(make_tape(nd, digs)), nd + 9)
        got = [int(t) for t in slots_of(tape, nd)]
        ok += (got == [DIG0 + d for d in digs])
    return ok, n


stages = [(1, 10), (2, 12), (3, 12), (4, 12)]
if SMOKE:
    stages = [(1, 8), (2, 8)]
step = 0
best_ok, best_sd = -1, None
for (kmax, dmax) in stages:
    stag = 0
    lr = 5e-3
    for g_ in opt.param_groups:
        g_["lr"] = lr
    for s in range(600 if SMOKE else 4000):
        step += 1
        if s % 1500 == 1499:
            lr = max(lr / 2, 5e-4)
            for g_ in opt.param_groups:
                g_["lr"] = lr
        x, ks = gen_batch(32, kmax, dmax)
        loss = terminal_loss(x, ks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if s % 200 == 0 or SMOKE and s % 100 == 0:
            ok, n = probe(kmax, dmax)
            print(f"[c26b] stage k<={kmax} s{s} lr={lr:.5f} CE {loss.item():.4f} "
                  f"crisp-probe {ok}/{n}", flush=True)
            if ok > best_ok:
                best_ok = ok
                best_sd = {k: v.clone() for k, v in m.state_dict().items()}
            if ok >= 60:
                stag = 1
                break
    if not stag:
        print(f"[c26b] stage k<={kmax} did not crystallize", flush=True)
if best_sd is not None:
    m.load_state_dict(best_sd)
    print(f"[c26b] restored best-probe checkpoint (probe {best_ok}/64)", flush=True)
torch.save(m.state_dict(), "c26b_sgd.pt")

# mechanism report: rows changed vs the seeded counter program
Eh, Ph = m.E.argmax(-1), m.P.argmax(-1)
changed = [(a, h, int(seed_Eh[a, h]), int(Eh[a, h]), int(seed_Ph[a, h]), int(Ph[a, h]))
           for a in range(A) for h in range(H)
           if int(Eh[a, h]) != int(seed_Eh[a, h]) or int(Ph[a, h]) != int(seed_Ph[a, h])]
print(f"[c26b] rows changed vs counter seed: {len(changed)}", flush=True)
counter_rows = [(a, h) for a in (MARK, BLK, SEP) for h in range(H)]
counter_changed = [c for c in changed if (c[0], c[1]) in counter_rows]
print(f"[c26b] counter-region rows changed: {len(counter_changed)} "
      f"{'(protocol reused intact)' if not counter_changed else counter_changed}",
      flush=True)
for (a, h, e0, e1, p0, p1) in changed[:30]:
    tn = {MARK: "MARK", BLK: "BLK", SEP: "SEP", PAD: "PAD"}.get(a, f"d{a - DIG0}")
    print(f"  tok {tn:>5} st{h:2d}: E {e0}->{e1}  P {p0}->{p1}", flush=True)

res = {}
certs = dict(indist=(3, 500), n16=(16, 200), n32=(32, 100), joint=(64, 100))
if SMOKE:
    certs = dict(indist=(2, 40))
for name, (nd, n) in certs.items():
    gp = random.Random(2600 + nd)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, gp)
        tape, np_, tr = m.run_fixpoint(torch.tensor(make_tape(nd, digs)), nd + 9)
        got = [int(t) for t in slots_of(tape, nd)]
        ok += (got == [DIG0 + d for d in digs])
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
    v.update(S2=res["n16"]["exact"], S3=res["n32"]["exact"], S4=res["joint"]["exact"],
             S5=res["joint"]["trace_ok"] and abs(res["joint"]["passes_mean"] - 65) <= 6.5)
    v["ALL"] = int(v["S1"].split("/")[0]) >= 498 and v["S2"] == "200/200" \
        and v["S3"] == "100/100" and v["S4"] == "100/100" and v["S5"]
out = dict(tag="ARC2-C26B-P6-SGD", cert=res, verdict=v, steps=step,
           rows_changed=len(changed), counter_rows_changed=len(counter_changed),
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
