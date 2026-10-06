"""
ARC-2 CYCLE 17 / MACHINE v5: STABILITY — dual-gating kills the SRAM transient
=============================================================================
Operator directive (C17): no further TF re-tests — improve OUR system.
The machine's #1 defect (C12-C16 evidence): the SRAM branch oscillates.
  standalone sram_s1 (C12): target CE L2048 0.1190 (transient in the
  standalone organ itself)
  machine v4 (C15): ICL target @4096 0.0228 (3x) -> 0.0021 (9x) ->
  0.1972 (final) — an 80x swing across checkpoints
  machine v4 final ckpt (C16): ICL target @16384 1.6329 vs 0.0025 at the
  9x ckpt — transient states are LENGTH-SENSITIVE (host-organ coupling:
  the co-trained host adds a context-length-dependent offset at the
  query position).
Root-cause split: (a) host-organ coupling — fixable now; (b) organ-intrinsic
readout oscillation (sram_s1) — measured alongside.
FIX: DUAL-GATING — each branch's host head output gets a learned exp-scale
head_gate[r] (init 0 -> scale 1, neutral at t=0, symmetric to the certified
organ_gate). On organ branches the optimizer can close the host's
contribution to exactly zero when it is noise (C12: host on ICL = ln16,
useless). Mechanism: logits_r = exp(head_gate[r]) * head_r(h_r) + organ_r.
If the host's optimal ICL contribution is ~0, the gate closes -> the
query-position readout depends only on the exact register file -> the
length-sensitivity disappears and the trajectory should go monotonic.
Protocol: machine v5 = v4 + head gates (21,309 p), 12000 cycling steps,
4 checkpoints (3000/6000/9000/12000) to catch the transient window; each
ckpt: ICL target @4096 + @16384 (the two failure metrics) + echo/mod7/add
@4096 (must hold cert). NO TF arm (operator directive; baseline cited:
micro_tf_12k from unified_add.log).
SUCCESS = (i) ICL target @4096 <= 0.05 at ckpts 6k/9k/12k;
          (ii) ICL target @16384 within 3x of @4096 at every ckpt;
          (iii) echo <= -0.25, mod7 <= 0.01, add <= 0.02 @4096 at final.
USAGE: OMP_NUM_THREADS=1 python3 -u unified_stable.py
"""
import json, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB, NK, K0, AD = 75, 16, 6, 45
CFG = dict(steps=12000, batch=32, train_len=63, ckpts=[3000, 6000, 9000, 12000],
           d_model=16, KSTACK=4096)
print(f"[setup] unified-stable (machine v5, dual-gating) cfg={CFG}", flush=True)
t_start = time.time()

g = {"__name__": "u5"}
exec(open("unified_add.py").read().split("\nRESULTS = {}")[0], g)
SSMBlock, n_params, eval_task = g["SSMBlock"], g["n_params"], g["eval_task"]
gen_echo_t, gen_icl_t, gen_mod7_t, gen_add_t = (
    g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"], g["gen_add_t"])


class MachineV5(nn.Module):
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
        self.head_gate = nn.Parameter(torch.zeros(4))          # DUAL-GATING
        self.stack_table = nn.Parameter(0.1 * torch.randn(8, VOCAB))
        self.W_readout = nn.Linear(d, 32)
        self.organ_gate = nn.Parameter(torch.tensor(0.0))
        self.carry_table = nn.Parameter(0.1 * torch.randn(2, 10, 10, 10))

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
    x, y, _ = gen_add_t(batch, length, rng)
    return x, y, torch.full((batch,), 3, dtype=torch.long)


@torch.no_grad()
def icl_probe(model, L, reps=2):
    """the two failure metrics: ICL target CE at L (4096 and 16384)."""
    model.eval()
    tgt_ce = tgt_n = 0.0
    for i in range(reps):
        rng = random.Random(950_000 + L + i)
        x, y, o = gen_icl_t(1, L, rng)
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return round(tgt_ce / tgt_n, 4)


def eval_all(model, tag, results):
    r = {"echo": eval_task(model, gen_echo_t, 4096, 1, 0),
         "icl": eval_task(model, gen_icl_t, 4096, 2, 1, tgt=True),
         "mod7": eval_task(model, gen_mod7_t, 4096, 1, 2),
         "add": eval_task(model, gen_add_t, 4096, 1, 3),
         "icl_t16384": icl_probe(model, 16384, 1),
         "head_gates": [round(float(v), 3) for v in torch.exp(model.head_gate)],
         "organ_gate": round(float(torch.exp(model.organ_gate)), 3)}
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)


RESULTS = {}
torch.manual_seed(0)
m5 = MachineV5()
print(f"[arm] machine v5 (dual-gating) params={n_params(m5)}", flush=True)
m5.train()
opt = torch.optim.AdamW(m5.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
ckpts_done = set()
for step in range(1, CFG["steps"] + 1):
    r = (step - 1) % 4
    x, y, task = gen_pure(r, CFG["batch"], CFG["train_len"], rng)
    lm, rt = train_step(m5, opt, x, y, task)
    if step in CFG["ckpts"] and step not in ckpts_done:
        torch.save(m5.state_dict(), f"unified_stable_{step}.pt")
        ckpts_done.add(step)
        print(f"    [v5] checkpoint at step {step}", flush=True)
    if step % 3000 == 0:
        print(f"  [v5 s0] step {step}/{CFG['steps']} lm {lm:.4f} rt {rt:.4f} "
              f"head_gates {torch.exp(m5.head_gate).tolist()} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[v5] trained in {time.time()-t0:.0f}s", flush=True)
torch.save(m5.state_dict(), "unified_stable_final.pt")

print("[eval] checkpoints @4096 (+ ICL target @16384 probe):", flush=True)
for c in CFG["ckpts"]:
    m5.load_state_dict(torch.load(f"unified_stable_{c}.pt"))
    eval_all(m5, f"v5_{c // 1000}k", RESULTS)
del m5

print("\n" + "=" * 100)
print("MACHINE v5 DUAL-GATING — ICL target CE trajectory (v4 ref: 0.0228/0.0021/")
print("0.1972 at 3k/9k/final; v4 final @16384 = 1.6329 vs 9k = 0.0025)")
print("SUCCESS: (i) icl target @4096 <= 0.05 at 6k/9k/12k; (ii) @16384 within")
print("3x of @4096 at every ckpt; (iii) echo <= -0.25, mod7 <= 0.01, add <= 0.02.")
print("=" * 100)
for k in ["v5_3k", "v5_6k", "v5_9k", "v5_12k"]:
    v = RESULTS[k]
    print(f"{k:<7} icl4096 {v['icl']}  icl16384 {v['icl_t16384']}  echo {v['echo']}  "
          f"mod7 {v['mod7']}  add {v['add']}  head_gates {v['head_gates']} og {v['organ_gate']}",
          flush=True)
print("=" * 100)
final = {"tag": "ARC2-C17-MACHINE-V5-DUAL-GATING", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
