"""
ARC-2 C24: P4 MULTI-PASS MACHINE — input-driven iteration count + mechanism
halt + learned transducer pass (the program's first ENDOGENOUS iteration:
the pass program and the counter discipline are LEARNED, not hand-coded as
in the cycle-3 IFT / cycle-5 sort loops).
================================================================================
TASK: iterated increment — tape = [MARK x k][SEP][digits LSB-first][PAD].
One pass = a learned Mealy transducer over the tape; the machine iterates
passes until tape FIXPOINT (cycle-5 mechanism halt). Target: digits = x + k
with full carry chains; counter discipline (erase exactly one MARK per pass)
emerges from the input-output contract + curriculum over k.

PRIOR ART (directive 4, logged): looped/adaptive-compute lineage = Neural
GPU, ACT (Graves'16), Universal Transformer (Dehghani'19), DEQ; ACT
ponder-cost halting has degenerate regimes (LT2 appendix A.4) and naive
early-exit collapses representations (LoopFormer ICLR'26) -> we use
MECHANISM halt (fixpoint), not learned halt probabilities. Fan et al.'24:
adaptive stopping improves length generalization. TFs on multi-step CA:
single-step learnable, multi-step collapses without intermediate context
(NCA-conv survey '19); LifeGPT needs an EXTERNAL autoregressive loop.

SUCCESS BARS (declared before launch):
 M1 in-dist exact >= 99.5% (k<=4, <=12 digits, 500 samples)
 M2 depth: 100% exact at k=16 (200x) AND k=64 (100x), 40 digits
 M3 joint: k=64, 120 digits (>=7680 cell-steps), 100% exact, 100 samples
 M4 halt discipline: passes used = k+1 (+/-10% mean), counter trace shows
    exactly one MARK erased per acting pass
 M5 CA-k rule-90 stretch arm: DEFERRED to C24b (2-head write design) —
    declared, not attempted this cycle (logged, honest)
 M6 wall < 25 min, peak < 1GB, one thread
No TF arm (C17 directive; cite existing logs + light-cone/TC0 argument).
HONESTY: layout (counter region + data region) is designer-supplied; fully
open-ended protocol discovery remains open (logged in PROBLEM_MAP).

ARMS:
 A  = end-to-end from final-tape contract only (the P4 claim).
 A2 = repair of A: GUMBEL-HARD tape chain — run 1 (soft chain, tau-anneal)
      failed 0/900 because training tapes never contained BLK tokens: the
      loop's pass-2+ inputs are OOD (counter rows untrained) -> no fixpoint.
      Gumbel-hard makes the training loop CRISP, exposing the true orbit.
 B  = same arch, per-pass rows supervised over the FULL reference orbit
      (every tape_p -> tape_{p+1} pair incl. the empty-counter identity
      pass) — run 1 supervised only pass 0 and failed at pass 2+ for the
      same BLK-OOD reason (single-pass rows were 0.997-confident perfect).
USAGE: OMP_NUM_THREADS=1 python3 -u c24_multipass.py   (SMOKE=1 = wiring)
"""
import json, math, os, random, resource, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
rng = random.Random(17)

# ------------------------------------------------------------- tape alphabet
MARK, BLK, SEP, PAD = 0, 1, 2, 13          # digits 3..12 ; PAD ends tape
DIG0 = 3
A = 14

def make_tape(k, digs):                    # digs LSB-first
    return [MARK] * k + [SEP] + [DIG0 + d for d in digs] + [PAD]

def gen_digits(nd, rng_):
    """overflow-free by construction: top digit 1..8 so x+k never grows the
    digit region (growing-region handling is C25 scope, logged)."""
    return [rng_.randrange(10) for _ in range(nd - 1)] + [rng_.randrange(1, 9)]

def oracle_inc_k(digs, k):
    got = list(map(int, str(int("".join(map(str, digs[::-1]))) + k)))[::-1]
    assert len(got) == len(digs), "overflow: generator must prevent this"
    return got

# ------------------------------------------------- reference pass (oracle)
def ref_pass(tape):
    """Mechanism reference: erase LEFTMOST mark (if any), and iff a mark was
    erased this pass add 1 to the LSB-first number. Same-length tape->tape."""
    out = list(tape)
    erased = False
    for t, tok in enumerate(tape):
        if not erased and tok == MARK:
            out[t] = BLK; erased = True; break
    if not erased:
        return out, False                   # identity -> fixpoint halt
    carry = 1
    for t in range(len(tape)):
        if tape[t] == SEP:
            break
    i = t + 1
    while i < len(tape) and tape[i] != PAD and carry:
        d = tape[i] - DIG0 + carry
        out[i] = DIG0 + d % 10; carry = d // 10; i += 1
    return out, True

def ref_run(tape, max_passes=4096):
    n = 0
    while n < max_passes:
        nxt, acted = ref_pass(tape)
        n += 1
        if not acted:
            return nxt, n
        tape = nxt
    return None, n

# oracle-first protocol (cycle 3): verify reference semantics BEFORE learning
ok = True
for _ in range(500):
    k = rng.randrange(1, 13); nd = rng.randrange(2, 15)
    digs = gen_digits(nd, rng)
    tape, n = ref_run(make_tape(k, digs))
    got = [t - DIG0 for t in tape if DIG0 <= t < PAD]
    if got != oracle_inc_k(digs, k) or n != k + 1:
        ok = False
print(f"[oracle] reference pass semantics: {'OK' if ok else 'BROKEN'}", flush=True)
assert ok

# ------------------------------------------------------------- soft machine
H = 8                                       # (phase2 x carry2 x erased2) room

class SoftPass(nn.Module):
    """Soft Mealy pass: p(h_t) = p(h_{t-1}) @ P[x_t];  out_t = E[x_t, h_{t-1}].
    Training is the soft automaton; certification snaps argmax (crisp)."""
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(0.1 * torch.randn(A, H, H))
        self.E = nn.Parameter(0.1 * torch.randn(A, H, A))
        self.p0 = nn.Parameter(torch.zeros(H))

    def forward(self, tape, hard=False):
        # tape: [B,T] long. Returns logits [B,T,A] (or hard tokens [B,T]).
        B, T = tape.shape
        p = F.softmax(self.p0, -1).unsqueeze(0).expand(B, H)
        outs = []
        for t in range(T):
            xt = tape[:, t]
            outs.append(torch.einsum("bh,bha->ba", p, self.E[xt]))
            p = torch.einsum("bh,bhH->bH", p, F.softmax(self.P[xt], -1))
        lg = torch.stack(outs, 1)
        return lg.argmax(-1) if hard else lg

    @torch.no_grad()
    def run_fixpoint(self, tape, max_passes):
        """hard iterate to fixpoint; returns (tape, passes, counter_trace)."""
        tr = [int((tape == MARK).sum())]
        for n in range(1, max_passes + 1):
            nxt = self.forward(tape.unsqueeze(0), hard=True).squeeze(0)
            if torch.equal(nxt, tape):
                return tape, n, tr
            tape = nxt
            tr.append(int((tape == MARK).sum()))
        return tape, max_passes, tr

def gen_batch(batch, kmax, dmax):
    tapes, tgts, ks = [], [], []
    for _ in range(batch):
        k = rng.randrange(1, kmax + 1)
        nd = rng.randrange(2, dmax + 1)
        digs = gen_digits(nd, rng)
        tapes.append(make_tape(k, digs))
        tgts.append(oracle_inc_k(digs, k))
        ks.append(k)
    T = max(len(t) for t in tapes)
    x = torch.full((batch, T), PAD, dtype=torch.long)
    for b, t in enumerate(tapes):
        x[b, :len(t)] = torch.tensor(t)
    return x, tgts, ks, T

# ------------------------------------------------------------------ arms
def orbit_pairs(tape):
    """Full reference orbit: every (tape_p, tape_{p+1}) pair incl. the
    final no-mark identity fixpoint. Training MUST cover this orbit —
    run-1 failure diagnosis: BLK-input rows + no-mark identity were OOD."""
    pairs, seen = [], set()
    while True:
        key = tuple(tape)
        if key in seen:
            break
        seen.add(key)
        nxt, acted = ref_pass(tape)
        pairs.append((list(tape), nxt))
        if not acted:
            break
        tape = nxt
    return pairs

def train_arm(tag, steps, per_pass=False, crisp_chain=False):
    model = SoftPass()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-2)
    sched = [(steps // 3, 1, 8), (2 * steps // 3, 2, 10), (steps, 4, 12)]
    for step in range(1, steps + 1):
        kmax, dmax = next((km, dm) for s, km, dm in sched if step <= s)
        x, tgts, ks, T = gen_batch(32, kmax, dmax)
        if per_pass:
            pairs = []
            for b in range(x.shape[0]):
                pairs.extend(orbit_pairs(x[b].tolist()))
            xb = torch.tensor([p[0] for p in pairs])
            yb = torch.tensor([p[1] for p in pairs])
            loss = F.cross_entropy(model(xb).reshape(-1, A), yb.reshape(-1))
        else:
            B, Tt = x.shape
            tgt_final = [oracle_inc_k([t - DIG0 for t in x[b].tolist()
                                       if DIG0 <= t < PAD], ks[b]) for b in range(B)]
            # unroll kmax+1 passes; contract = final tape (goal state, not
            # per-pass mechanism). crisp_chain (arm A2): STRAIGHT-THROUGH
            # crisp tape between passes — run-1's soft chain kept BLK-input
            # rows untrained (no crisp exposure); STE gives the true orbit
            # in the forward pass with gradient through the logits.
            p_tape = F.one_hot(x, A).float()
            Psoft = F.softmax(model.P, -1)
            for _ in range(kmax + 1):
                p = F.softmax(model.p0, -1).unsqueeze(0).expand(B, H).contiguous()
                elos = []
                for t in range(Tt):
                    elos.append(torch.einsum("ba,ahv,bh->bv", p_tape[:, t], model.E, p))
                    p = torch.einsum("ba,bh,ahH->bH", p_tape[:, t], p, Psoft)
                elo = torch.stack(elos, 1)
                sm = F.softmax(elo, -1)
                if crisp_chain:
                    hard = F.one_hot(elo.argmax(-1), A).float()
                    p_tape = hard + sm - sm.detach()        # STE
                else:
                    p_tape = sm
            y = torch.full((B, Tt), PAD, dtype=torch.long)
            for b in range(B):
                gd = tgt_final[b]
                sep = int((x[b] == SEP).nonzero()[0])
                y[b, sep + 1:sep + 1 + len(gd)] = torch.tensor(gd) + DIG0
                y[b, :sep] = BLK                              # counter contract
            m = (y != PAD)
            loss = F.cross_entropy(elo[m], y[m])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % max(1, steps // 6) == 0:
            print(f"[{tag}] {step}/{steps} CE {loss.item():.5f}", flush=True)
    return model

def diagnose(model, tag):
    sd = model.state_dict()
    p0e = -(F.softmax(sd["p0"], -1) * F.log_softmax(sd["p0"], -1)).sum().item()
    tape, n, tr = model.run_fixpoint(torch.tensor(make_tape(2, [7, 9])), max_passes=12)
    got = [int(t) - DIG0 for t in tape.tolist() if DIG0 <= t < PAD]
    print(f"[diag:{tag}] p0 entropy {p0e:.3f} (ln{H}={math.log(H):.2f}); "
          f"97,k2 -> {got} passes={n} trace={tr} (want [9,9], 3 passes)",
          flush=True)

# ------------------------------------------------------------------ certify
def certify(model, tag, cases):
    res = {}
    for name, (k, nd, n) in cases.items():
        fails, passes_used, traces_ok = 0, [], True
        for _ in range(n):
            digs = gen_digits(nd, rng)
            tape, np_, tr = model.run_fixpoint(torch.tensor(make_tape(k, digs)),
                                               max_passes=k + 9)   # FAIL FAST
            got = [int(t) - DIG0 for t in tape.tolist() if DIG0 <= t < PAD]
            if got != oracle_inc_k(digs, k):
                fails += 1
            passes_used.append(np_)
            for i in range(1, len(tr)):
                if tr[i] != max(tr[i - 1] - 1, 0):
                    traces_ok = False
        mu = sum(passes_used) / len(passes_used)
        res[name] = dict(exact=f"{n - fails}/{n}", passes_mean=round(mu, 3),
                         trace_ok=bool(traces_ok))
        print(f"[{tag}] {name}: {n - fails}/{n} exact, passes={mu:.2f} "
              f"(want {k + 1}), trace_ok={traces_ok}", flush=True)
    return res

if SMOKE:
    m = train_arm("SMOKE-A", 60)
    r = certify(m, "SMOKE-A", {"k2d4": (2, 4, 5)})
    print("SMOKE OK", flush=True)
    raise SystemExit

CASES = dict(indist=(2, 8, 500), k16=(16, 40, 200), k64=(64, 40, 100),
             joint=(64, 120, 100))

mA = train_arm("armA-e2e", 3000, per_pass=False)
torch.save(mA.state_dict(), "c24_armA.pt")
diagnose(mA, "armA")
certA = certify(mA, "armA-e2e", CASES)

mA2 = train_arm("armA2-crisp", 4500, per_pass=False, crisp_chain=True)
torch.save(mA2.state_dict(), "c24_armA2.pt")
diagnose(mA2, "armA2")
certA2 = certify(mA2, "armA2-crisp", CASES)

mB = train_arm("armB-orbit", 1500, per_pass=True)
torch.save(mB.state_dict(), "c24_armB.pt")
diagnose(mB, "armB")
certB = certify(mB, "armB-orbit", CASES)

def passed(cert, k):
    a = cert["indist"]["exact"].split("/"); m1 = int(a[0]) / int(a[1]) >= 0.995
    m2 = cert["k16"]["exact"].startswith("200") and cert["k64"]["exact"].startswith("100")
    m3 = cert["joint"]["exact"].startswith("100")
    m4 = cert["joint"]["trace_ok"] and abs(cert["joint"]["passes_mean"] - (k + 1)) <= 0.1 * (k + 1)
    return dict(M1=m1, M2=m2, M3=m3, M4=m4, ALL=all([m1, m2, m3, m4]))

verdictA, verdictA2, verdictB = passed(certA, 2), passed(certA2, 2), passed(certB, 2)
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
wall = time.time() - T0
print(f"[verdict] armA={verdictA} armA2={verdictA2} armB={verdictB}", flush=True)
res = dict(tag="ARC2-C24-MULTIPASS", armA=dict(verdict=verdictA, cert=certA),
           armA2=dict(verdict=verdictA2, cert=certA2),
           armB=dict(verdict=verdictB, cert=certB), wall_s=round(wall, 1),
           peak_mb=round(peak, 1))
print("RESULT " + json.dumps(res), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
print("DONE", flush=True)
