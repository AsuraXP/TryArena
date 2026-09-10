"""
ARC-2 CYCLE 14 / P15 FIX: ISOLATED mixture of heterogeneous state machines
==========================================================================
C13 found: the unified shared-host machine has a stability knee ~10k steps —
past it, flat mixed loss hides shared-parameter drift (echo -0.30 -> +1.13,
ICL target 0.21 -> 0.44) while mod7 keeps improving = classic NEGATIVE
TRANSFER on the shared backbone.
Phase-1 validation: negative-transfer fixes (PCGrad/FairBranch/Recon/
Rec-MoELoRA) operate on big transformer backbones via gradient surgery or
low-rank task experts; DTME-MTL warns full duplication can overfit IN THEIR
regime (learned shared features with real transfer value). Our regime is the
opposite: task token vocabularies are DISJOINT, so sharing has ZERO positive
transfer — only interference. Mutation: FULL per-task isolation of the
state machine (each expert = own SSM host + own exact-memory organ), sharing
only the token-disjoint embedding table + the learned per-example router.
Zero sharing => zero interference, by construction.
Branches:
  r0: host0 + exact-stack organ   (Dyck-echo, tokens 0-5)
  r1: host1 + SRAM organ          (ICL cipher,  6-37)
  r2: host2 only                  (mod-7 walk,  38-44)
Router: MLP on the first 3 token embeddings -> 3-way argmax (per-EXAMPLE,
direct CE, L-DIRECT-GRADIENT). All heads zero-init (L-GATE-INIT).
Win = the 20k knee disappears: echo stays ~-0.30 @4096 at 20k (C13 shared
host: +1.13), ICL target returns to standalone-cert level (0.0218-0.0270),
mod7 at/below cert band, routing 1.0, vs protocol TF 20k arm.
USAGE: OMP_NUM_THREADS=1 python3 -u unified_iso.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
VOCAB = 45
NK = 16
LN16 = math.log(16.0)
CFG = dict(steps=20000, batch=32, n_echo=8, n_icl=16, n_mod7=8, train_len=63,
           ckpt_step=10000, d_model=16, KSTACK=4096)
print(f"[setup] unified-iso cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- data
# reuse the C13 data layer verbatim (generators + mixed stream)
u = {"__name__": "u"}
exec(open("unified.py").read().split("\nRESULTS = {}")[0], u)
gen_echo_t, gen_icl_t, gen_mod7_t, gen_mixed = (
    u["gen_echo_t"], u["gen_icl_t"], u["gen_mod7_t"], u["gen_mixed"])
SSMBlock, n_params = u["SSMBlock"], u["n_params"]

# ---------------------------------------------------------------- model
class IsoModel(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)          # shared (token-disjoint)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.router = nn.Sequential(nn.Linear(3 * d, 16), nn.GELU(), nn.Linear(16, 3))
        self.hosts = nn.ModuleList([SSMBlock(d) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Linear(d, VOCAB) for _ in range(3)])
        for h in self.heads:                        # L-GATE-INIT
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.stack_table = nn.Parameter(0.1 * torch.randn(8, VOCAB))   # r0
        self.W_readout = nn.Linear(d, VOCAB)                           # r1

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
        K0, K1, V0, V1 = 6, 6 + NK, 6 + NK, 6 + 2 * NK
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
            out[:, t] = cand * mask.unsqueeze(-1)
            for b in range(B):
                if bool(is_val[b]) and int(last_key[b]) >= 0:
                    slots[b, int(last_key[b])] = emb[b, t].detach()
                    seen[b, int(last_key[b])] = True
                if bool(is_key[b]):
                    last_key[b] = int(k[b])
        return out

    def forward(self, x):
        B, L = x.shape
        rl = self.router(self.emb(x[:, :3]).reshape(B, -1))
        task = rl.argmax(-1)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for r in range(3):
            idx = (task == r).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            xr = x[idx]
            hr = self.norms[r](self.hosts[r](self.emb(xr)))
            lg = self.heads[r](hr)
            if r == 0:
                lg = lg + self._stack_logits(xr)
            elif r == 1:
                lg = lg + self._sram_logits(xr)
            out.scatter_(0, idx.view(-1, 1, 1).expand_as(lg), lg)
        return out, rl

# ---------------------------------------------------------------- train
def train_step(model, opt, x, y, task):
    logits, rl = model(x)
    l_lm = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    l_rt = F.cross_entropy(rl, task)
    loss = l_lm + 0.5 * l_rt
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    return l_lm.item(), l_rt.item()

@torch.no_grad()
def eval_task(model, gen, L, reps=2, task_id=None, tgt=False):
    model.eval()
    bs = max(1, min(4, 4096 // L))
    ce = orc = n = 0.0
    tgt_ce = tgt_n = 0.0
    route_acc = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gen(bs, L, rng)
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        if task_id is not None:
            route_acc += int((rl.argmax(-1) == task_id).float().mean().item())
        if tgt:
            tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    d = round((ce - orc) / n, 4)
    if tgt:
        d = (d, round(tgt_ce / tgt_n, 4))
    if task_id is not None:
        d = (d, route_acc / reps) if not tgt else (d[0], d[1], route_acc / reps)
    return d

def eval_all(model, tag, results):
    r = {"echo": eval_task(model, gen_echo_t, 4096, 2, 0),
         "icl": eval_task(model, gen_icl_t, 4096, 2, 1, tgt=True),
         "mod7": eval_task(model, gen_mod7_t, 4096, 2, 2)}
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)

# ---------------------------------------------------------------- experiment
RESULTS = {}
torch.manual_seed(0)
iso = IsoModel()
print(f"[arm] isolated mixture of state machines params={n_params(iso)}", flush=True)
iso.train()
opt = torch.optim.AdamW(iso.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
ckpt_saved = False
for step in range(1, CFG["steps"] + 1):
    x, y, task = gen_mixed(CFG["batch"], CFG["train_len"], rng)
    lm, rt = train_step(iso, opt, x, y, task)
    if step == CFG["ckpt_step"] and not ckpt_saved:
        torch.save(iso.state_dict(), "unified_iso_10k.pt")
        ckpt_saved = True
    if step % 2500 == 0:
        print(f"  [iso s0] step {step}/{CFG['steps']} lm {lm:.4f} rt {rt:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[iso] trained in {time.time()-t0:.0f}s", flush=True)
torch.save(iso.state_dict(), "unified_iso_20k.pt")

print("[eval] 10k checkpoint @4096:", flush=True)
iso.load_state_dict(torch.load("unified_iso_10k.pt"))
eval_all(iso, "iso_s0_10k", RESULTS)
print("[eval] 20k final @4096:", flush=True)
iso.load_state_dict(torch.load("unified_iso_20k.pt"))
eval_all(iso, "iso_s0_20k", RESULTS)
RESULTS["iso_s0_20k"]["params"] = n_params(iso)
RESULTS["iso_s0_10k"]["params"] = n_params(iso)
del iso

# protocol baseline: micro TF, same mixed stream, 20k steps
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
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(DEVICE)
        return self.head(self.enc(h, mask))

torch.manual_seed(0)
tf = MixedTF()
print(f"[arm] micro TF (protocol baseline, 20k) params={n_params(tf)}", flush=True)
tf.train()
opt = torch.optim.AdamW(tf.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
for step in range(1, CFG["steps"] + 1):
    x, y, task = gen_mixed(CFG["batch"], CFG["train_len"], rng)
    loss = F.cross_entropy(tf(x).reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 5000 == 0:
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
     "params": n_params(tf)}
RESULTS["micro_tf_20k"] = r
print(f"  micro_tf_20k: {r}", flush=True)

print("\n" + "=" * 88)
print("ISOLATED Mixture of State Machines @4096 (dCE; routing acc in parens)")
print("C13 shared-host knee: echo -0.3019->+1.1277, icl tgt 0.2057->0.4399 @20k")
print("=" * 88)
for k in ["iso_s0_10k", "iso_s0_20k", "micro_tf_20k"]:
    v = RESULTS[k]
    print(f"{k:<13} params {v['params']:<7} echo {v['echo']}  icl {v['icl']}  mod7 {v['mod7']}",
          flush=True)
print("=" * 88)
final = {"tag": "ARC2-C14-ISO-MOE", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
