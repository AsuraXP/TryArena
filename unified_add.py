"""
ARC-2 CYCLE 15 / MACHINE v4: the CARRY organ (P3 enters the machine)
====================================================================
Phase-1 validation: TFs learn addition via discovered carry circuits
(Quirke & Barez; arxiv 2402.02619) needing layers + attention over the
carry chain for cascades; the structurally correct solution is known
("an RNN implementing addition with carry; derive the weights on paper"
— smallest-TF-addition discussion); NTK exact-learnability proofs are
infinite-width. Nobody has built the carry transducer as a ROUTED organ
with an exact analytic-oracle benchmark inside a heterogeneous machine.
Mutation: 4th organ family — an ARITHMETIC TRANSDUCER: state = 1-bit
carry, transition EXACT by mechanism (carry' = (a+b+carry) >= 10),
readout = learned table over (carry, a, b) -> sum digit, in its own
10-token alphabet (L-ORGAN-ALPHABET). Not a stack (LIFO), not SRAM
(content-addressed), not plain finite-state (mod-7): input-dependent
arithmetic transition + learned readout.
Task T3: triplet stream a0 b0 c0 a1 b1 c1 ... (random digits; A 45-54,
B 55-64, C 65-74); LM targets: A-pos -> next B (random ln10), B-pos ->
c_t = (a_t + b_t + carry_t) mod 10 (organ's job; oracle 0), C-pos ->
next A (random ln10).
Machine v4 = v3 (iso3: isolated branches + task cycling + sub-vocab
organs + gate on SRAM) + branch r3 (host3 + zero-init head + carry
table 200x10 at tokens 65-74) + 4-way router.
Protocol: 12000 cycling steps (3000/task), ckpts 3000/9000/final,
4096 eval; micro TF 12k on the 4-way mixed stream.
Win: carry dCE @4096 near-oracle (cert line: host-only SSM will be
~ln10=2.302 on the sum positions if it can't hold cascading carries);
other 3 families hold v3 certs; routing 1.0; TF loses the carry chain.
USAGE: OMP_NUM_THREADS=1 python3 -u unified_add.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB, NK = 75, 16
K0 = 6
AD = 45          # add family base: A 45-54, B 55-64, C 65-74
LN10 = math.log(10.0)
CFG = dict(steps=12000, batch=32, train_len=63, ckpts=[3000, 9000],
           d_model=16, KSTACK=4096)
print(f"[setup] unified-add (machine v4) cfg={CFG} VOCAB={VOCAB}", flush=True)
t_start = time.time()

g = {"__name__": "u4"}
exec(open("unified_iso.py").read().split("\nRESULTS = {}")[0], g)
SSMBlock, n_params, eval_task = g["SSMBlock"], g["n_params"], g["eval_task"]
gen_echo_t, gen_icl_t, gen_mod7_t = g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"]

# ---------------------------------------------------------------- add data
def gen_add(batch, length, rng):
    """triplet stream a0 b0 c0 a1 b1 c1 ... (c_t = the SUM, organ's job).
    LM targets: A-pos -> next B (random, ln10); B-pos -> C (sum, oracle 0);
    C-pos -> next A (random, ln10). carry_{t+1} = (a_t+b_t+carry_t) >= 10.
    x has 3*(length//3)+1 tokens so y=x[1:] ends on a B->C (sum) target."""
    # repo convention (gen_echo): nll[j] = entropy recorded when x[j] was
    # generated; y = x[1:length+1]; oracle = nll[1:length+1] so o[i] =
    # H(y[i]) — target B -> LN10 (random), target C -> 0 (the sum),
    # target A -> LN10 (random). 3*ngen >= length + 1.
    ngen = (length + 3) // 3
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll = [], []
        carry = 0
        for t in range(ngen):
            a = rng.randrange(10)
            b = rng.randrange(10)
            c = (a + b + carry) % 10
            carry = 1 if (a + b + carry) >= 10 else 0
            x.append(AD + a); nll.append(LN10)          # A: random input
            x.append(AD + 10 + b); nll.append(LN10)     # B: random input
            x.append(AD + 20 + c); nll.append(0.0)      # C: the sum, exact
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

def gen_add_t(batch, length, rng):
    return gen_add(batch, length, rng)

def gen_mixed4(batch, length, rng):
    n = batch // 4
    x0, y0, _ = gen_echo_t(n, 64, rng)
    x1, y1, _ = gen_icl_t(n, 64, rng)
    x2, y2, _ = gen_mod7_t(n, length, rng)
    x3, y3, _ = gen_add(n, length, rng)
    x = torch.cat([x0[:, :length], x1, x2, x3])
    y = torch.cat([y0[:, :length], y1, y2, y3])
    task = torch.cat([torch.zeros(n), torch.ones(n),
                      torch.full((n,), 2.0), torch.full((n,), 3.0)]).long()
    return x, y, task

# ---------------------------------------------------------------- model
class MachineV4(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.router = nn.Sequential(nn.Linear(3 * d, 16), nn.GELU(), nn.Linear(16, 4))
        self.hosts = nn.ModuleList([SSMBlock(d) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Linear(d, VOCAB) for _ in range(4)])
        for h in self.heads:
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.stack_table = nn.Parameter(0.1 * torch.randn(8, VOCAB))
        self.W_readout = nn.Linear(d, 32)
        self.organ_gate = nn.Parameter(torch.tensor(0.0))
        self.carry_table = nn.Parameter(0.1 * torch.randn(2, 10, 10, 10))  # [carry, a, b] -> C logits

    def _stack_logits(self, x):
        B, L = x.shape
        KS = CFG["KSTACK"]
        feats = torch.empty(B, L, 3, dtype=torch.long)
        for b in range(B):
            stack = []
            for t in range(L):
                tok = int(x[b, t])
                if tok in (0, 1):
                    if len(stack) < KS:
                        stack.append(tok)
                elif tok == 2:
                    if stack:
                        stack.pop()
                feats[b, t, 0] = stack[-1] if stack else 0
                feats[b, t, 1] = 1 if not stack else 0
                feats[b, t, 2] = 1 if tok == 2 else 0
        combo = feats[:, :, 0] + feats[:, :, 1] * 2 + feats[:, :, 2] * 4
        return self.stack_table[combo]

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
                ct = self.carry_table[carry.long(), ai, b]      # (B, 10)
                mask = is_B.float().unsqueeze(-1)
                out[:, t, AD + 20:AD + 30] = ct * mask
                carry = s >= 10                                  # exact transition
            if bool(is_A.any()):
                last_a = torch.where(is_A, a, last_a)
        return out

    def forward(self, x):
        B, L = x.shape
        rl = self.router(self.emb(x[:, :3]).reshape(B, -1))
        task = rl.argmax(-1)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for r in range(4):
            idx = (task == r).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            xr = x[idx]
            hr = self.norms[r](self.hosts[r](self.emb(xr)))
            lg = self.heads[r](hr)
            if r == 0:
                lg = lg + self._stack_logits(xr)
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

def gen_pure(r, batch, length, rng):
    if r == 0:
        x, y, _ = gen_echo_t(batch, 64, rng)
        return x[:, :length], y[:, :length], torch.zeros(batch, dtype=torch.long)
    if r == 1:
        x, y, _ = gen_icl_t(batch, 64, rng)
        return x, y, torch.ones(batch, dtype=torch.long)
    if r == 2:
        x, y, _ = gen_mod7_t(batch, length, rng)
        return x, y, torch.full((batch,), 2, dtype=torch.long)
    x, y, _ = gen_add(batch, length, rng)
    return x, y, torch.full((batch,), 3, dtype=torch.long)

def eval_all(model, tag, results):
    r = {"echo": eval_task(model, gen_echo_t, 4096, 2, 0),
         "icl": eval_task(model, gen_icl_t, 4096, 2, 1, tgt=True),
         "mod7": eval_task(model, gen_mod7_t, 4096, 2, 2),
         "add": eval_task(model, gen_add_t, 4096, 2, 3)}
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)

RESULTS = {}
torch.manual_seed(0)
m4 = MachineV4()
print(f"[arm] machine v4 (4 branches + carry organ) params={n_params(m4)}", flush=True)
m4.train()
opt = torch.optim.AdamW(m4.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
ckpts_done = set()
for step in range(1, CFG["steps"] + 1):
    r = (step - 1) % 4
    x, y, task = gen_pure(r, CFG["batch"], CFG["train_len"], rng)
    lm, rt = train_step(m4, opt, x, y, task)
    if step in CFG["ckpts"] and step not in ckpts_done:
        torch.save(m4.state_dict(), f"unified_add_{step}.pt")
        ckpts_done.add(step)
        print(f"    [add] checkpoint at step {step}", flush=True)
    if step % 2000 == 0:
        print(f"  [v4 s0] step {step}/{CFG['steps']} lm {lm:.4f} rt {rt:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[v4] trained in {time.time()-t0:.0f}s", flush=True)
torch.save(m4.state_dict(), "unified_add_final.pt")

print("[eval] checkpoints @4096:", flush=True)
for c in CFG["ckpts"]:
    m4.load_state_dict(torch.load(f"unified_add_{c}.pt"))
    eval_all(m4, f"v4_{c // 1000}x", RESULTS)
m4.load_state_dict(torch.load("unified_add_final.pt"))
eval_all(m4, "v4_final", RESULTS)
RESULTS["v4_final"]["params"] = n_params(m4)
for c in CFG["ckpts"]:
    RESULTS[f"v4_{c // 1000}x"]["params"] = n_params(m4)
del m4

# protocol baseline: micro TF on the 4-way mixed stream
class MixedTF(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB)
        self.head.weight = self.emb.weight

    @staticmethod
    def sinusoidal(L, d):
        p = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        i = torch.arange(0, d, 2).float()
        p[:, 0::2] = torch.sin(pos * torch.exp(-9 * i / d))
        p[:, 1::2] = torch.cos(pos * torch.exp(-9 * i / d))
        return p

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) + self.sinusoidal(L, self.d_model).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to("cpu")
        return self.head(self.enc(h, mask))

torch.manual_seed(0)
tf = MixedTF()
print(f"[arm] micro TF (protocol baseline, 12k mixed4) params={n_params(tf)}", flush=True)
tf.train()
opt = torch.optim.AdamW(tf.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
for step in range(1, CFG["steps"] + 1):
    x, y, task = gen_mixed4(CFG["batch"], CFG["train_len"], rng)
    loss = F.cross_entropy(tf(x).reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 4000 == 0:
        print(f"  [tf] step {step}/{CFG['steps']} CE {loss.item():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

@torch.no_grad()
def eval_tf_task(model, gen, L, reps=2):
    model.eval()
    bs = max(1, min(4, 4096 // L))
    ce = orc = n = 0.0
    tgt_ce = tgt_n = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gen(bs, L, rng)
        nll = -F.log_softmax(model(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return (round((ce - orc) / n, 4), round(tgt_ce / tgt_n, 4))

r = {"echo": eval_tf_task(tf, gen_echo_t, 4096, 2),
     "icl": eval_tf_task(tf, gen_icl_t, 4096, 2),
     "mod7": eval_tf_task(tf, gen_mod7_t, 4096, 2),
     "add": eval_tf_task(tf, gen_add_t, 4096, 2),
     "params": n_params(tf)}
RESULTS["micro_tf_12k"] = r
print(f"  micro_tf_12k: {r}", flush=True)

print("\n" + "=" * 96)
print("MACHINE v4 (4 families) @4096 (dCE; routing acc in parens; add = total dCE,")
print("sum-position oracle 0, A-position oracle ln10; ln10 = 2.303)")
print("v3 certs: echo -0.2981 | icl 0.0102|0.0047 | mod7 0.0015")
print("=" * 96)
for k in ["v4_3x", "v4_9x", "v4_final", "micro_tf_12k"]:
    v = RESULTS[k]
    print(f"{k:<13} params {v['params']:<7} echo {v['echo']}  icl {v['icl']}  "
          f"mod7 {v['mod7']}  add {v['add']}", flush=True)
print("=" * 96)
final = {"tag": "ARC2-C15-MACHINE-V4-CARRY", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
