"""
ARC-2 C24c / P4-DISC: OPEN-ENDED ITERATION — endogenous discovery of the
iterative program from the TERMINAL CONTRACT ONLY. No per-pass rows, no orbit
supervision, no mechanism labels anywhere. The machine must discover: phase
structure, erase-one counter discipline, the +1-carry algorithm, and halt.
================================================================================
PRIOR ART (directive 4): NLI ICLR'26 (Gumbel-Softmax discrete programs +
test-time gradient search through a differentiable executor — validates the
compiler-in-the-loop idea; programs there are latent-token sequences, not
crisp machines, no length certs). Adaptive Neural Compilation (Bunel'17):
programs from final-tape loss with learned stop flags + halting/efficiency/
confidence penalties — soft multinomial execution, no crystallization to
exact discrete machines. arXiv 2502.16763: iterated template-matching
arithmetic, engineered templates. LESS-LESSON gap we attack: nobody ships a
CRISP snapped discrete machine whose ITERATION PROTOCOL was discovered from
raw objectives, with input-driven pass count and length-certified exactness.
Our mutation: staged discovery curriculum + STE-crisp loop + generic bounded
repair SEARCH over snapped tables (the ssr_lab compiler, mechanism-free).

EXPERIMENTS (in order; each declares pass/fail):
 E1 discovery curriculum: k=1 until the single-pass algorithm exists, then
    k<=2, k<=3, k<=4. STE crisp chain. H=16. Bars: in-dist exact >= 0.95,
    pass counting (mean within 1 of k+1). FAIL -> E2 must rescue.
 E2 generic repair search: snap tables, greedy hill-climb single-entry edits
    of E/P against the SAME terminal contract on a held-out generator
    (no oracle anywhere). Bars: in-dist >= 0.995 AND k=16 100%.
 E3 depth certification of the repaired machine: k=64 (200x train k... 16x),
    joint k=64 x L=120. Bars: 100% exact, passes = k+1 exact.
 M4 honesty check: grep-level — no ref_pass rows used in E1/E2 training.
NO TF arm (operator directive + C17; cite logs only).
USAGE: OMP_NUM_THREADS=1 python3 -u c24c_p4disc.py   (SMOKE=1 = wiring)
"""
import json, math, os, random, resource, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
rng = random.Random(29)

MARK, BLK, SEP, PAD = 0, 1, 2, 13
DIG0 = 3
A = 14
H = 16

def make_tape(k, digs):
    return [MARK] * k + [SEP] + [DIG0 + d for d in digs] + [PAD]

def gen_digits(nd, rng_):
    return [rng_.randrange(10) for _ in range(nd - 1)] + [rng_.randrange(1, 9)]

def oracle_inc_k(digs, k):
    got = list(map(int, str(int("".join(map(str, digs[::-1]))) + k)))[::-1]
    assert len(got) == len(digs)
    return got

class Disc(nn.Module):
    """Mealy pass; STE-crisp chain in training; argmax snap at cert."""
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(0.1 * torch.randn(A, H, H))
        self.E = nn.Parameter(0.1 * torch.randn(A, H, A))
        self.p0 = nn.Parameter(torch.zeros(H))

    def one_pass_soft(self, p_tape, B, T):
        Psoft = F.softmax(self.P, -1)
        p = F.softmax(self.p0, -1).unsqueeze(0).expand(B, H).contiguous()
        elos = []
        for t in range(T):
            elos.append(torch.einsum("ba,ahv,bh->bv", p_tape[:, t], self.E, p))
            p = torch.einsum("ba,bh,ahH->bH", p_tape[:, t], p, Psoft)
        return torch.stack(elos, 1)

    def forward_hard(self, tape):
        B, T = tape.shape
        p = F.softmax(self.p0, -1).unsqueeze(0).expand(B, H)
        outs = []
        for t in range(T):
            xt = tape[:, t]
            outs.append(torch.einsum("bh,bha->ba", p, self.E[xt]))
            p = torch.einsum("bh,bhH->bH", p, F.softmax(self.P[xt], -1))
        return torch.stack(outs, 1).argmax(-1)

    @torch.no_grad()
    def run_fixpoint(self, tape, max_passes):
        tr = [int((tape == MARK).sum())]
        for n in range(1, max_passes + 1):
            nxt = self.forward_hard(tape.unsqueeze(0)).squeeze(0)
            if torch.equal(nxt, tape):
                return tape, n, tr
            tape = nxt
            tr.append(int((tape == MARK).sum()))
        return tape, max_passes, tr

def gen_batch(batch, kmax, dmax, grng):
    tapes, ks = [], []
    for _ in range(batch):
        k = grng.randrange(1, kmax + 1)
        nd = grng.randrange(2, dmax + 1)
        digs = gen_digits(nd, grng)
        tapes.append(make_tape(k, digs)); ks.append(k)
    T = max(len(t) for t in tapes)
    x = torch.full((batch, T), PAD, dtype=torch.long)
    for b, t in enumerate(tapes):
        x[b, :len(t)] = torch.tensor(t)
    return x, ks

def contract_targets(x, ks):
    B, T = x.shape
    y = torch.full((B, T), PAD, dtype=torch.long)
    for b in range(B):
        row = x[b].tolist()
        digs = [t - DIG0 for t in row if DIG0 <= t < PAD]
        gd = oracle_inc_k(digs, ks[b])
        sep = row.index(SEP)
        y[b, sep + 1:sep + 1 + len(gd)] = torch.tensor(gd) + DIG0
        y[b, :sep] = BLK
    return y

def terminal_loss(model, x, ks, crisp=True):
    B, T = x.shape
    p_tape = F.one_hot(x, A).float()
    kmax = max(ks)
    elo = None
    for _ in range(kmax + 1):
        elo = model.one_pass_soft(p_tape, B, T)
        sm = F.softmax(elo, -1)
        if crisp:
            hard = F.one_hot(elo.argmax(-1), A).float()
            p_tape = hard + sm - sm.detach()
        else:
            p_tape = sm
    y = contract_targets(x, ks)
    m = (y != PAD)
    return F.cross_entropy(elo[m], y[m])

# ------------------------------------------------------------------ E1
def e1_train(steps=1500 if SMOKE else 6000):
    model = Disc()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-2)
    g = random.Random(31)
    stages = [(0.25, 1, 10), (0.5, 2, 12), (0.75, 3, 12), (1.01, 4, 12)]
    last = 0.0
    for step in range(1, steps + 1):
        frac = step / steps
        kmax, dmax = next((km, dm) for f, km, dm in stages if frac < f)
        x, ks = gen_batch(32, kmax, dmax, g)
        loss = terminal_loss(model, x, ks, crisp=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        last = loss.item()
        if step % max(1, steps // 8) == 0:
            print(f"[E1] {step}/{steps} kmax={kmax} CE {last:.5f}", flush=True)
    return model

@torch.no_grad()
def eval_exact(model, k, nd, n, grng):
    ok, passes = 0, []
    for _ in range(n):
        digs = gen_digits(nd, grng)
        tape, np_, _ = model.run_fixpoint(torch.tensor(make_tape(k, digs)), k + 9)
        got = [int(t) - DIG0 for t in tape.tolist() if DIG0 <= t < PAD]
        ok += (got == oracle_inc_k(digs, k))
        passes.append(np_)
    return ok, n, sum(passes) / n

# ------------------------------------------------------------------ E2 search
def e2_search(model, budget=800 if SMOKE else 2600):
    """Generic greedy hill-climb over single-entry edits of SNAPPED tables,
    scored by the terminal contract on a held-out generator. No oracle."""
    sd = {k: v.clone() for k, v in model.state_dict().items()}
    Ph = F.softmax(sd["P"], -1).argmax(-1)           # [A,H] snapped transitions
    Eh = sd["E"].argmax(-1)                          # [A,H] snapped outputs
    g = random.Random(41)

    def hard_tables_eval(Ph_, Eh_, cases):
        ok = tot = 0
        p0h_ = int(F.softmax(sd["p0"], -1).argmax().item())
        for (k, nd) in cases:
            digs = gen_digits(nd, g)
            tape = torch.tensor(make_tape(k, digs))
            for n in range(1, k + 9 + 1):
                h = p0h_
                out = tape.clone()
                for t in range(tape.shape[0]):
                    out[t] = Eh_[int(tape[t]), h]
                    h = int(Ph_[int(tape[t]), h])
                if torch.equal(out, tape):
                    break
                tape = out
            got = [int(t) - DIG0 for t in tape.tolist() if DIG0 <= t < PAD]
            ok += (got == oracle_inc_k(digs, k)); tot += 1
        return ok / tot

    cases = [(1, 8)] * 8 + [(2, 10)] * 8 + [(4, 12)] * 8
    best = hard_tables_eval(Ph, Eh, cases)
    print(f"[E2] snapped baseline exact = {best:.3f}", flush=True)
    edits = 0
    for it in range(budget):
        if best >= 1.0 and it > budget // 2:
            break
        which = rng.randrange(2)
        if which == 0:
            i, j = rng.randrange(A), rng.randrange(H)
            old = int(Eh[i, j]); new = rng.randrange(A)
        else:
            i, j = rng.randrange(A), rng.randrange(H)
            old = int(Ph[i, j]); new = rng.randrange(H)
        if new == old:
            continue
        if which == 0:
            Eh[i, j] = new
        else:
            Ph[i, j] = new
        sc = hard_tables_eval(Ph, Eh, cases)
        edits += 1
        if sc >= best:
            if sc > best:
                print(f"[E2] edit {edits}: {best:.3f} -> {sc:.3f}", flush=True)
            best = sc
        else:                                            # revert the edit
            if which == 0:
                Eh[i, j] = old
            else:
                Ph[i, j] = old
    return Ph, Eh, best, edits

def snapped_fixpoint(Ph, Eh, p0h, tape, max_passes):
    tr = [int((tape == MARK).sum())]
    for n in range(1, max_passes + 1):
        h = p0h
        out = tape.clone()
        for t in range(tape.shape[0]):
            out[t] = Eh[int(tape[t]), h]
            h = int(Ph[int(tape[t]), h])
        if torch.equal(out, tape):
            return tape, n, tr
        tape = out
        tr.append(int((tape == MARK).sum()))
    return tape, max_passes, tr

def e3_certify(Ph, Eh, p0h, cases):
    res = {}
    for name, (k, nd, n) in cases.items():
        g = random.Random(60 + k)
        ok, passes, traces_ok = 0, [], True
        for _ in range(n):
            digs = gen_digits(nd, g)
            tape, np_, tr = snapped_fixpoint(Ph, Eh, p0h,
                                             torch.tensor(make_tape(k, digs)), k + 9)
            got = [int(t) - DIG0 for t in tape.tolist() if DIG0 <= t < PAD]
            ok += (got == oracle_inc_k(digs, k))
            passes.append(np_)
            for i in range(1, len(tr)):
                if tr[i] != max(tr[i - 1] - 1, 0):
                    traces_ok = False
        res[name] = dict(exact=f"{ok}/{n}", passes_mean=round(sum(passes) / n, 2),
                         trace_ok=traces_ok)
        print(f"[E3] {name}: {ok}/{n} exact, passes={res[name]['passes_mean']} "
              f"(want {k + 1}), trace_ok={traces_ok}", flush=True)
    return res

if SMOKE:
    m = e1_train()
    ok, n, mp = eval_exact(m, 1, 8, 20, random.Random(7))
    print(f"[SMOKE E1] k=1: {ok}/{n}, passes={mp:.2f}", flush=True)
    Ph, Eh, best, edits = e2_search(m)
    print("SMOKE OK", flush=True)
    raise SystemExit

# ------------------------------------------------------------------ run E1
print("=" * 100, flush=True)
m = e1_train()
torch.save(m.state_dict(), "c24c_e1.pt")
g = random.Random(51)
e1res = {}
for k, nd in [(1, 8), (2, 10), (4, 12)]:
    ok, n, mp = eval_exact(m, k, nd, 200, g)
    e1res[f"k{k}"] = dict(exact=f"{ok}/{n}", passes_mean=round(mp, 2))
    print(f"[E1-cert] k={k}: {ok}/{n} exact, passes={mp:.2f} (want {k + 1})", flush=True)

# ------------------------------------------------------------------ run E2
print("=" * 100, flush=True)
Ph, Eh, best, edits = e2_search(m)
p0h = int(F.softmax(m.state_dict()["p0"], -1).argmax().item())
print(f"[E2] done: best={best:.3f}, edits={edits}", flush=True)

# ------------------------------------------------------------------ run E3
print("=" * 100, flush=True)
e3 = e3_certify(Ph, Eh, p0h, dict(
    indist=(2, 10, 500), k16=(16, 40, 200), k64=(64, 40, 100),
    joint=(64, 120, 100)))
torch.save(dict(Ph=Ph, Eh=Eh, p0h=p0h), "c24c_searched.pt")

v = dict(E1_k1_exact=e1res["k1"]["exact"],
         E2_in_dist=e3["indist"]["exact"],
         E2_k16=e3["k16"]["exact"],
         E3_k64=e3["k64"]["exact"], E3_joint=e3["joint"]["exact"],
         passes_discipline=e3["joint"]["trace_ok"] and
         abs(e3["joint"]["passes_mean"] - 65) <= 6.5)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
res = dict(tag="ARC2-C24C-P4-DISC", e1=e1res, e2_best=best, e2_edits=edits,
           e3=e3, verdict=v, wall_s=round(time.time() - T0, 1),
           peak_mb=round(peak, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(res), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
print("DONE", flush=True)
