"""
ARC-2 C24b: MULTI-PASS LOOP, SECOND INSTANCE — rule-90 cellular automaton
evolved k steps, k given on the tape (input-driven pass count). C24 certified
the loop on iterated increment (armB); C24b tests whether the SAME loop
mechanism hosts a DIFFERENT learned pass: a CA step is a light-cone task —
each pass extends information radius by 1, so k steps need k passes (TFs need
k layers; cite-only per C17). Prior art logged in C24 block (log.md).
================================================================================
TAPE: [MARK x k][SEP][bits][PAD]. Pass = Mealy transducer with a LOOKAHEAD
write head E[x_t, x_{t+1}, h] (cycle-3 factored-head precedent; old-tape
lookahead + current-pass state = distinct position roles, L-DETERMINISM ok):
data cell out_i = b_{i-1} (state) XOR b_{i+1} (lookahead, PAD=0 boundary).
Orbit-supervised rows (L-ORBIT-COVERAGE), fixpoint halt (L-MECHANISM-HALT).

BARS (declared before launch):
 B1 in-dist (k<=4, L<=15): >= 99.5% exact, 500 samples
 B2 depth: k=16 L=31 200/200 AND k=64 L=31 100/100
 B3 joint: k=64 x L=127, 100/100 exact
 B4 passes = k+1 exact at every scale; one-mark-per-pass trace
 B5 wall < 20 min, peak < 1GB, one thread
USAGE: OMP_NUM_THREADS=1 python3 -u c24b_ca.py   (SMOKE=1 = wiring)
"""
import json, math, os, random, resource, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
SMOKE = os.environ.get("SMOKE") == "1"
rng = random.Random(23)

MARK, BLK, SEP, PAD = 0, 1, 2, 3           # bits 4,5
B0 = 4
A = 6
H = 12

def make_tape(k, bits):
    return [MARK] * k + [SEP] + [B0 + b for b in bits] + [PAD]

def ca90_step(bits):
    L = len(bits)
    return [(bits[i - 1] if i > 0 else 0) ^ (bits[i + 1] if i < L - 1 else 0)
            for i in range(L)]

def ca90_k(bits, k):
    for _ in range(k):
        bits = ca90_step(bits)
    return bits

def ref_pass(tape):
    """erase LEFTMOST mark; iff erased, apply one rule-90 step to the bits."""
    out = list(tape)
    erased = False
    for t, tok in enumerate(tape):
        if not erased and tok == MARK:
            out[t] = BLK; erased = True; break
    if not erased:
        return out, False
    sep = tape.index(SEP)
    end = tape.index(PAD)
    bits = [tape[i] - B0 for i in range(sep + 1, end)]
    nb = ca90_step(bits)
    for i, v in enumerate(nb):
        out[sep + 1 + i] = B0 + v
    return out, True

def orbit_pairs(tape):
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

# oracle-first
ok = True
for _ in range(500):
    k = rng.randrange(1, 13); L = rng.randrange(3, 20)
    bits = [rng.randrange(2) for _ in range(L)]
    tape = make_tape(k, bits); n = 0
    while True:
        nxt, acted = ref_pass(tape); n += 1
        if not acted:
            break
        tape = nxt
    got = [t - B0 for t in tape if t in (B0, B0 + 1)]
    if got != ca90_k(bits, k) or n != k + 1:
        ok = False
print(f"[oracle] CA reference pass semantics: {'OK' if ok else 'BROKEN'}", flush=True)
assert ok

class LookaheadPass(nn.Module):
    """out_t = E[x_t, x_{t+1}, h_{t-1}];  h_t = h_{t-1} @ P[x_t].
    x_{t+1} is read from the OLD tape (2-tape semantics)."""
    def __init__(self):
        super().__init__()
        self.P = nn.Parameter(0.1 * torch.randn(A, H, H))
        self.E = nn.Parameter(0.1 * torch.randn(A, A, H, A))
        self.p0 = nn.Parameter(torch.zeros(H))

    def forward(self, tape, hard=False):
        B, T = tape.shape
        nxt = torch.cat([tape[:, 1:], tape[:, -1:]], 1)     # old-tape lookahead
        p = F.softmax(self.p0, -1).unsqueeze(0).expand(B, H)
        outs = []
        for t in range(T):
            outs.append(torch.einsum("bh,bha->ba", p, self.E[tape[:, t], nxt[:, t]]))
            p = torch.einsum("bh,bhH->bH", p, F.softmax(self.P[tape[:, t]], -1))
        lg = torch.stack(outs, 1)
        return lg.argmax(-1) if hard else lg

    @torch.no_grad()
    def run_fixpoint(self, tape, max_passes):
        tr = [int((tape == MARK).sum())]
        for n in range(1, max_passes + 1):
            nxt = self.forward(tape.unsqueeze(0), hard=True).squeeze(0)
            if torch.equal(nxt, tape):
                return tape, n, tr
            tape = nxt
            tr.append(int((tape == MARK).sum()))
        return tape, max_passes, tr

def gen_batch(batch, kmax, lmax):
    xs = []
    for _ in range(batch):
        k = rng.randrange(1, kmax + 1)
        L = rng.randrange(3, lmax + 1)
        bits = [rng.randrange(2) for _ in range(L)]
        xs.append(make_tape(k, bits))
    T = max(len(t) for t in xs)
    x = torch.full((batch, T), PAD, dtype=torch.long)
    for b, t in enumerate(xs):
        x[b, :len(t)] = torch.tensor(t)
    return x

model = LookaheadPass()
opt = torch.optim.AdamW(model.parameters(), lr=2e-2)
STEPS = 400 if SMOKE else 3000
sched = [(STEPS // 3, 1, 7), (2 * STEPS // 3, 2, 11), (5 * STEPS // 6, 4, 15), (STEPS, 4, 21)]
for step in range(1, STEPS + 1):
    kmax, lmax = next((km, lm) for s, km, lm in sched if step <= s)
    x = gen_batch(32, kmax, lmax)
    pairs = []
    for b in range(x.shape[0]):
        pairs.extend(orbit_pairs(x[b].tolist()))
    xb = torch.tensor([p[0] for p in pairs])
    yb = torch.tensor([p[1] for p in pairs])
    loss = F.cross_entropy(model(xb).reshape(-1, A), yb.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % max(1, STEPS // 6) == 0:
        print(f"[train] {step}/{STEPS} CE {loss.item():.5f}", flush=True)
torch.save(model.state_dict(), "c24b_ca.pt")

def certify(cases):
    res = {}
    for name, (k, L, n) in cases.items():
        fails, pu, traces_ok = 0, [], True
        for _ in range(n):
            bits = [rng.randrange(2) for _ in range(L)]
            tape, np_, tr = model.run_fixpoint(torch.tensor(make_tape(k, bits)),
                                               max_passes=k + 9)
            got = [int(t) - B0 for t in tape.tolist() if t in (B0, B0 + 1)]
            if got != ca90_k(bits, k):
                fails += 1
            pu.append(np_)
            for i in range(1, len(tr)):
                if tr[i] != max(tr[i - 1] - 1, 0):
                    traces_ok = False
        mu = sum(pu) / len(pu)
        res[name] = dict(exact=f"{n - fails}/{n}", passes_mean=round(mu, 3),
                         trace_ok=bool(traces_ok))
        print(f"[cert] {name}: {n - fails}/{n} exact, passes={mu:.2f} "
              f"(want {k + 1}), trace_ok={traces_ok}", flush=True)
    return res

if SMOKE:
    certify({"k2L5": (2, 5, 5)})
    print("SMOKE OK", flush=True)
    raise SystemExit

cert = certify(dict(indist=(2, 8, 500), k16=(16, 31, 200), k64=(64, 31, 100),
                    joint=(64, 127, 100)))
a = cert["indist"]["exact"].split("/")
v = dict(B1=int(a[0]) / int(a[1]) >= 0.995,
         B2=cert["k16"]["exact"] == "200/200" and cert["k64"]["exact"] == "100/100",
         B3=cert["joint"]["exact"] == "100/100",
         B4=cert["joint"]["trace_ok"] and abs(cert["joint"]["passes_mean"] - 65) <= 6.5)
v["ALL"] = all(v.values())
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
res = dict(tag="ARC2-C24B-CA-LOOP", verdict=v, cert=cert,
           wall_s=round(time.time() - T0, 1), peak_mb=round(peak, 1))
print(f"[verdict] {v}", flush=True)
print("RESULT " + json.dumps(res), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
print("DONE", flush=True)
