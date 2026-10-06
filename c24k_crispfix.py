"""
ARC-2 C24f / P4-DISC run 5a: SGD REFINEMENT seeded from the discovered
counter program (c24d_searched.pt). Counter protocol is DISCOVERED and
stable; gradient refines the digit rows under the same terminal contract.
No rows frozen (logged), low lr, adaptive stage advancement via the
label-free HARD gate (advance when hard exact >= 60/64 in-stage probe).
BARS: S1 in-dist >= 99.5%; S2 k=16; S3 k=64; S4 joint k=64 x L=120;
S5 passes=k+1 + trace. NO TF arm. USAGE: OMP_NUM_THREADS=1 python3 -u c24f_sgd.py
"""
import json, os, random, resource, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
rng = random.Random(127)
MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
A, H = 14, 16

def make_tape(k, digs):
    return [MARK] * k + [SEP] + [DIG0 + d for d in digs] + [PAD]

def gen_digits(nd, r):
    return [r.randrange(10) for _ in range(nd - 1)] + [r.randrange(1, 9)]

def oracle_inc_k(digs, k):
    got = list(map(int, str(int("".join(map(str, digs[::-1]))) + k)))[::-1]
    assert len(got) == len(digs)
    return got

class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(torch.zeros(A, H, H))
        self.E = nn.Parameter(torch.zeros(A, H, A))
        self.p0 = nn.Parameter(torch.zeros(H))

    def one_pass_soft(self, p_tape, B, T):
        # M6: STE-crisp STATE propagation — train dynamics == eval dynamics.
        # c24f (run 5a) hid its solution in state SUPERPOSITION: soft-mixture
        # probes passed 64/64 but crisp snap failed 0/60 (mixed tok0/tok2/
        # digit rows in state 11). Crisp state removes the hiding place;
        # gradient flows through the straight-through copy.
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
        # M10 (run 10): TRUE CRISP execution — snapped tables, single state.
        # Runs 5a/7/8/9 measured SOFT state mixtures here and overstated
        # capability (L-SUPERPOSITION-HIDE); a cert must run the machine it
        # ships.
        B, T = tape.shape
        Eh = self.E.argmax(-1); Ph = self.P.argmax(-1)
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
opt = torch.optim.AdamW(m.parameters(), lr=5e-3)
g = random.Random(131)

def gen_batch(batch, kmax, dmax):
    tapes, ks = [], []
    for _ in range(batch):
        k = g.randrange(1, kmax + 1)
        nd = g.randrange(2, dmax + 1)
        tapes.append(make_tape(k, gen_digits(nd, g))); ks.append(k)
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
        digs = [t - DIG0 for t in row if DIG0 <= t < PAD]
        gd = oracle_inc_k(digs, ks[b])
        sep = row.index(SEP)
        y[b, sep + 1:sep + 1 + len(gd)] = torch.tensor(gd) + DIG0
        y[b, :sep] = SEP  # discovered protocol dissolves marks into SEPs
    msk = (y != PAD)
    return F.cross_entropy(elo[msk], y[msk])

@torch.no_grad()
def probe(kmax, dmax, n=64):
    gp = random.Random(time.time_ns() % 1000003)
    ok = 0
    for _ in range(n):
        k = gp.randrange(1, kmax + 1)
        digs = gen_digits(gp.randrange(2, dmax + 1), gp)
        tape, np_, _ = m.run_fixpoint(torch.tensor(make_tape(k, digs)), k + 9)
        got = [int(t) - DIG0 for t in tape.tolist() if DIG0 <= t < PAD]
        ok += (got == oracle_inc_k(digs, k))
    return ok, n

stages = [(1, 10), (2, 12), (3, 12), (4, 12)]
step = 0
best_ok, best_sd = -1, None
for (kmax, dmax) in stages:
    stag = 0
    lr = 5e-3
    for g_ in opt.param_groups:
        g_["lr"] = lr
    for s in range(4000):
        step += 1
        if s % 1500 == 1499:                       # per-stage lr decay
            lr = max(lr / 2, 5e-4)
            for g_ in opt.param_groups:
                g_["lr"] = lr
        x, ks = gen_batch(32, kmax, dmax)
        loss = terminal_loss(x, ks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if s % 200 == 0:
            ok, n = probe(kmax, dmax)
            print(f"[c24k] stage k<={kmax} s{s} lr={lr:.5f} CE {loss.item():.4f} "
                  f"hard-probe {ok}/{n}", flush=True)
            if ok > best_ok:
                best_ok = ok
                best_sd = {k: v.clone() for k, v in m.state_dict().items()}
            if ok >= 60:
                stag = 1
                break
    if not stag:
        print(f"[c24k] stage k<={kmax} did not crystallize in 4000 steps", flush=True)
if best_sd is not None:
    m.load_state_dict(best_sd)
    print(f"[c24k] restored best-probe checkpoint (probe {best_ok}/64)", flush=True)
torch.save(m.state_dict(), "c24k_crispfix.pt")

res = {}
for name, (k, nd, n) in dict(indist=(2, 10, 500), k16=(16, 40, 200),
                             k64=(64, 40, 100), joint=(64, 120, 100)).items():
    gp = random.Random(400 + k)
    ok, passes, traces = 0, [], True
    for _ in range(n):
        digs = gen_digits(nd, gp)
        tape, np_, tr = m.run_fixpoint(torch.tensor(make_tape(k, digs)), k + 9)
        got = [int(t) - DIG0 for t in tape.tolist() if DIG0 <= t < PAD]
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
out = dict(tag="ARC2-C24K-P4-CRISPFIX", cert=res, verdict=v, steps=step,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
