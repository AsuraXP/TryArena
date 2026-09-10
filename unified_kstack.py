"""
ARC-2 CYCLE 18 / MACHINE v6: DEPTH-K STACK READOUT ORGAN (beyond Markov completion)
=====================================================================================
C14-C17: the r0 stack organ keys its readout on (top, empty, prevPop) —
L-MARKOV-COMPLETION: everything it can express is a function of those three
features. Querying the k-th stack element (k >= 2) is the first task the
whole organ family cannot solve: the answer needs a joint (STATE, QUERY)
interaction — which element was asked FOR. Carry table keys (carry,a,b), not
a free query; SRAM is content-addressed but has no STACK ORDER; the host is
a d16 linear recurrence that would have to rediscover the stack encoding
from zero-init under 2400-step duty (L-DUTY).
Phase-1 validation (searched 2026-08-22): Kaiser et al. arXiv 1506.02516
"Learning to Transduce with Unbounded Memory" (differentiable stack/queue/
dequeue transducers) — readout is TOP-ONLY; Hu et al. ECCV-2018 (neural
module stack) — top-pointer only. No prior work: query-keyed k-th-element
readout over EXACT (one-hot) stack state, learned, as a routed organ inside
a heterogeneous machine, benchmarked with exact analytic-oracle dCE.
MUTATION: r0 organ -> top-4 exact features: 8 s-bits (value+valid per depth)
+ 1 prevPop bit; query state one-hot over {none-push, none-pop, Q0..Q3} (6).
Readout = learned additive table A (20 rows) + learned state x query
BILINEAR M (9x6, zero-init = L-GATE-INIT on the interaction; its inputs are
exact features, so gradients are full-rank from step 1). Expressibility
(proven by construction):
  Qk pos  (pp=0): s-dep = sum_j M[2j,kk] s_j  -> set col kk = col (none-push)
                  except depth k-1 carries the answer term
  pop pos (qo=none-pop): s-dep = (M[2j,5]-M[2j,4]) s_j -> col5-col4 carries
                  f(s1) = the echo answer; A cancels s-dep at push positions
Task T5 "kstack": triplet stream (op, Qk, ans): op = stack operation
(push 0/1 = tokens 0/1, pop = token 2 allowed only when depth >= 2 so a
query stays answerable), then k uniform in 1..min(4, depth-after-op),
ans = the k-th element (token 0/1; the simulator treats t%3==2 as NOT an
operation — the answer is not a push). Oracle = exact entropy:
op: depth>=2 -> -(0.3 ln0.3)*2 - 0.4 ln0.4, else ln2; Q: ln kmax; ans: 0.
The answer depends ONLY on the top-4 stack elements => length-invariant by
construction (the @16384 probe must track @4096; KSTACK cap 4096 > any
eval depth needed for top-4, per the C16 cap lesson).
Protocol: machine v6 = v5 (dual-gating) + 4 Q tokens + new r0 organ
(~26.9k params), 12000 cycling steps over 5 tasks (2400 each), ckpts
3000/6000/9000/12000; per ckpt: kstack+echo+mod7+add @4096, ICL target
@4096/@16384, kstack @16384. NO TF arm (operator directive C17).
SUCCESS = (i) kstack dCE @4096 <= 0.01 at 9k and 12k (dCE 0 = answer CE 0,
  the oracle is exact);
          (ii) kstack @16384 within 3x of @4096 at every ckpt;
          (iii) no regression at final: echo <= -0.25, icl tgt @4096 <= 0.005,
          mod7 <= 0.01, add <= 0.02;
          (iv) routing 1.0 on all five streams.
USAGE: OMP_NUM_THREADS=1 python3 -u unified_kstack.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
t_start = time.time()

g = {"__name__": "u6"}
exec(open("unified_stable.py").read().split("\nRESULTS = {}")[0], g)
SSMBlock, n_params, eval_task = g["SSMBlock"], g["n_params"], g["eval_task"]
gen_echo_t, gen_icl_t, gen_mod7_t, gen_add_t = (
    g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"], g["gen_add_t"])

VOCAB, NK, K0, AD = 79, 16, 6, 45
Q0 = 75                       # query tokens Q0..Q3 = k-1, k in 1..4
LN2 = math.log(2.0)
CFG = dict(steps=12000, batch=32, train_len=63, ckpts=[3000, 6000, 9000, 12000],
           d_model=16, KSTACK=4096)
print(f"[setup] unified-kstack (machine v6, depth-k readout) cfg={CFG} VOCAB={VOCAB}", flush=True)

# ---------------------------------------------------------------- kstack data
def gen_kstack_t(batch, length, rng):
    """triplet stream (op, Qk, ans), ngen = (length+3)//3 groups.
    Oracle (repo convention, gen_add): nll[j] = H when x[j] generated;
    o[i] = H(y[i]) — y = x[1:length+1]."""
    ngen = (length + 3) // 3
    H_POP = -(2 * 0.3 * math.log(0.3) + 0.4 * math.log(0.4))  # d>=2: K0 .3, K1 .3, P .4
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll = [], []
        stack = []
        for t in range(ngen):
            d = len(stack)
            op = 2 if (d >= 2 and rng.random() < 0.4) else rng.randrange(2)
            if op in (0, 1):                              # push
                if len(stack) < CFG["KSTACK"]:
                    stack.append(op)
            else:                                         # pop (d >= 2 -> depth stays >= 1)
                stack.pop()
            kmax = min(4, len(stack))                     # >= 1 always
            k = rng.randrange(1, kmax + 1)
            ans = stack[-k]                               # depth >= k guaranteed
            x.append(op);              nll.append(H_POP if d >= 2 else LN2)
            x.append(Q0 + k - 1);      nll.append(math.log(float(kmax)))
            x.append(ans);             nll.append(0.0)
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

# ---------------------------------------------------------------- model
class MachineV6(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.router = nn.Sequential(nn.Linear(3 * d, 16), nn.GELU(), nn.Linear(16, 4))
        self.hosts = nn.ModuleList([SSMBlock(d) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Linear(d, VOCAB) for _ in range(4)])
        for h in self.heads:                              # L-GATE-INIT
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.head_gate = nn.Parameter(torch.zeros(4))     # DUAL-GATING (v5)
        # r0: depth-k readout organ — additive table + state x query bilinear
        self.kstack_add = nn.Parameter(0.1 * torch.randn(20, VOCAB))
        self.kstack_m = nn.Parameter(torch.zeros(9, 6, VOCAB))
        self.W_readout = nn.Linear(d, 32)
        self.organ_gate = nn.Parameter(torch.tensor(0.0))
        self.carry_table = nn.Parameter(0.1 * torch.randn(2, 10, 10, 10))

    def _kstack_logits(self, x):
        """exact top-4 stack features + query state -> additive + bilinear."""
        B, L = x.shape
        KS = CFG["KSTACK"]
        f = torch.zeros(B, L, 9)              # s-bits: (valid_j, value_j) x4, prevPop
        qo = torch.zeros(B, L, 6)             # 0..3 = Q0..Q3; 4 = none-push; 5 = none-pop
        idx = torch.zeros(B, L, 6, dtype=torch.long)
        idx[:, :, 4] = 12 + 4                 # q-state rows 12..17
        idx[:, :, 5] = 18                      # prevPop rows 18/19
        for b in range(B):
            ks = int(x[b].max()) >= Q0        # kstack row? (echo rows: no Q tokens)
            stack = []
            for t in range(L):
                tok = int(x[b, t])
                if not (ks and t % 3 != 0):   # only op positions mutate the stack
                    if tok in (0, 1):
                        if len(stack) < KS:
                            stack.append(tok)
                    elif tok == 2:
                        if stack:
                            stack.pop()
                for j in range(4):
                    if len(stack) > j:
                        f[b, t, 2 * j] = 1.0
                        f[b, t, 2 * j + 1] = stack[-(j + 1)]
                        idx[b, t, j] = j * 3 + 1 + int(stack[-(j + 1)])
                qk = tok - Q0 if Q0 <= tok <= Q0 + 3 else 4 + int(tok == 2)
                qo[b, t, qk] = 1.0
                idx[b, t, 4] = 12 + qk
                if tok == 2:
                    f[b, t, 8] = 1.0
                    idx[b, t, 5] = 19
        A = self.kstack_add
        add = (A[idx[:, :, 0]] + A[idx[:, :, 1]] + A[idx[:, :, 2]]
               + A[idx[:, :, 3]] + A[idx[:, :, 4]] + A[idx[:, :, 5]])
        return add + torch.einsum("blm,bln,mnv->blv", f, qo, self.kstack_m)

    def _sram_logits(self, x):
        B, L = x.shape
        d = CFG["d_model"]
        K1, V0, V1 = K0 + NK, K0 + NK, K0 + 2 * NK
        slots = torch.zeros(B, NK, d, device=x.device)
        seen = torch.zeros(B, NK, dtype=torch.bool, device=x.device)
        last_key = torch.full((B,), -1, dtype=torch.long, device=x.device)
        emb = self.emb(x)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for t in range(L):
            tok = x[:, t]
            is_key = (tok >= K0) & (tok < K1)
            is_val = (tok >= V0) & (tok < V1)
            k = (tok - K0).clamp(0, NK - 1)
            idx = k.view(B, 1, 1).expand(B, 1, d)
            cand = self.W_readout(slots.gather(1, idx).squeeze(1))
            mask = (is_key & seen.gather(1, k.unsqueeze(1)).squeeze(1)).float()
            out[:, t, K0:V1] = cand * mask.unsqueeze(-1)
            for b in range(B):
                if bool(is_val[b]) and int(last_key[b]) >= 0:
                    slots[b, int(last_key[b])] = emb[b, t].detach()
                    seen[b, int(last_key[b])] = True
                if bool(is_key[b]):
                    last_key[b] = int(k[b])
        return out

    def _carry_logits(self, x):
        """state = carry bit per row; transition exact by mechanism."""
        B, L = x.shape
        out = torch.zeros(B, L, VOCAB, device=x.device)
        last_a = torch.full((B,), -1, dtype=torch.long, device=x.device)
        carry = torch.zeros(B, dtype=torch.bool)
        for t in range(L):
            tok = x[:, t]
            is_A = (tok >= AD) & (tok < AD + 10)
            is_B = (tok >= AD + 10) & (tok < AD + 20)
            a = (tok - AD).clamp(0, 9)
            b = (tok - AD - 10).clamp(0, 9)
            if bool(is_B.any()):
                ai = last_a.clamp(0, 9)
                s = ai + b + carry.long()
                ct = self.carry_table[carry.long(), ai, b]
                mask = is_B.float().unsqueeze(-1)
                out[:, t, AD + 20:AD + 30] = ct * mask
                carry = s >= 10
            if bool(is_A.any()):
                last_a = torch.where(is_A, a, last_a)
        return out

    def forward(self, x):
        B, L = x.shape
        rl = self.router(self.emb(x[:, :3]).reshape(B, -1))
        task = rl.argmax(-1)
        hg = torch.exp(self.head_gate)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for r in range(4):
            idx = (task == r).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            xr = x[idx]
            hr = self.norms[r](self.hosts[r](self.emb(xr)))
            lg = hg[r] * self.heads[r](hr)
            if r == 0:
                lg = lg + self._kstack_logits(xr)
            elif r == 1:
                lg = lg + torch.exp(self.organ_gate) * self._sram_logits(xr)
            elif r == 3:
                lg = lg + self._carry_logits(xr)
            out.scatter_(0, idx.view(-1, 1, 1).expand_as(lg), lg)
        return out, rl

# ---------------------------------------------------------------- experiment
def train_step(model, opt, x, y, task):
    logits, rl = model(x)
    l_lm = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    l_rt = F.cross_entropy(rl, task)
    loss = l_lm + 0.5 * l_rt
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    return l_lm.item(), l_rt.item()

TASKS = ["echo", "kstack", "icl", "mod7", "add"]
TID = {"echo": 0, "kstack": 0, "icl": 1, "mod7": 2, "add": 3}

def gen_pure(name, batch, length, rng):
    if name == "echo":
        x, y, _ = gen_echo_t(batch, 64, rng)
        return x[:, :length], y[:, :length], torch.zeros(batch, dtype=torch.long)
    if name == "kstack":
        x, y, _ = gen_kstack_t(batch, length, rng)
        return x, y, torch.full((batch,), TID["kstack"], dtype=torch.long)
    if name == "icl":
        x, y, _ = gen_icl_t(batch, 64, rng)
        return x, y, torch.ones(batch, dtype=torch.long)
    if name == "mod7":
        x, y, _ = gen_mod7_t(batch, length, rng)
        return x, y, torch.full((batch,), 2, dtype=torch.long)
    x, y, _ = gen_add_t(batch, length, rng)
    return x, y, torch.full((batch,), 3, dtype=torch.long)

@torch.no_grad()
def icl_probe(model, L, reps=1):
    model.eval()
    tgt_ce = tgt_n = 0.0
    for i in range(reps):
        rng = random.Random(950_000 + L + i)
        x, y, o = gen_icl_t(1, L, rng)
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return round(tgt_ce / tgt_n, 4)

@torch.no_grad()
def kstack_probe(model, L, reps=1):
    """dCE of the whole kstack stream at L (length-invariance probe)."""
    model.eval()
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(960_000 + L + i)
        x, y, o = gen_kstack_t(1, L, rng)
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
    return round((ce - orc) / n, 4)

def eval_all(model, tag, results):
    r = {"echo": eval_task(model, gen_echo_t, 4096, 1, 0),
         "kstack": eval_task(model, gen_kstack_t, 4096, 1, 0),
         "icl": eval_task(model, gen_icl_t, 4096, 2, 1, tgt=True),
         "mod7": eval_task(model, gen_mod7_t, 4096, 1, 2),
         "add": eval_task(model, gen_add_t, 4096, 1, 3),
         "icl_t16384": icl_probe(model, 16384, 1),
         "kstack_16384": kstack_probe(model, 16384, 1),
         "head_gates": [round(float(v), 3) for v in torch.exp(model.head_gate)],
         "organ_gate": round(float(torch.exp(model.organ_gate)), 3),
         "kstack_m_abs": round(float(model.kstack_m.abs().sum()), 1)}
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)

RESULTS = {}
torch.manual_seed(0)
m6 = MachineV6()
print(f"[arm] machine v6 (depth-k readout organ) params={n_params(m6)}", flush=True)
m6.train()
opt = torch.optim.AdamW(m6.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
ckpts_done = set()
for step in range(1, CFG["steps"] + 1):
    name = TASKS[(step - 1) % 5]
    x, y, task = gen_pure(name, CFG["batch"], CFG["train_len"], rng)
    lm, rt = train_step(m6, opt, x, y, task)
    if step in CFG["ckpts"] and step not in ckpts_done:
        torch.save(m6.state_dict(), f"unified_kstack_{step}.pt")
        ckpts_done.add(step)
        print(f"    [v6] checkpoint at step {step}", flush=True)
    if step % 3000 == 0:
        print(f"  [v6 s0] step {step}/{CFG['steps']} lm {lm:.4f} rt {rt:.4f} "
              f"head_gates {torch.exp(m6.head_gate).tolist()} "
              f"kstack_m_abs {float(m6.kstack_m.abs().sum()):.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[v6] trained in {time.time()-t0:.0f}s", flush=True)
torch.save(m6.state_dict(), "unified_kstack_final.pt")

print("[eval] checkpoints @4096 (+ @16384 probes):", flush=True)
for c in CFG["ckpts"]:
    m6.load_state_dict(torch.load(f"unified_kstack_{c}.pt"))
    eval_all(m6, f"v6_{c // 1000}k", RESULTS)
del m6

print("\n" + "=" * 100)
print("MACHINE v6 DEPTH-K READOUT — v5 ref: echo -0.3153, icl tgt 0.0002/0.0004,")
print("mod7 0.0034, add 0.0095, routing 1.0. kstack is the new capability.")
print("SUCCESS: (i) kstack @4096 <= 0.01 at 9k/12k; (ii) @16384 within 3x @4096;")
print("(iii) echo <= -0.25, icl tgt <= 0.005, mod7 <= 0.01, add <= 0.02 at 12k.")
print("=" * 100)
for k in ["v6_3k", "v6_6k", "v6_9k", "v6_12k"]:
    v = RESULTS[k]
    print(f"{k:<7} kstack {v['kstack'][0]}  kstack16k {v['kstack_16384']}  "
          f"echo {v['echo']}  icl {v['icl']}  icl16k {v['icl_t16384']}  "
          f"mod7 {v['mod7']}  add {v['add']}  hg {v['head_gates']}  "
          f"m_abs {v['kstack_m_abs']}", flush=True)
print("=" * 100)
final = {"tag": "ARC2-C18-MACHINE-V6-DEPTHK", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
