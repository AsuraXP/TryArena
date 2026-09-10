"""
ARC-2 CYCLE 13 / THE MACHINE: one model, learned memory orchestration
=====================================================================
Phase-1 validation: MoE routing (expert choice, EMNLP-2023 "learning to
route") = per-TOKEN routing over HOMOGENEOUS experts; hybrid-memory LLMs
(Hydra, MoM; survey arxiv 2607.25380) = STATIC composition, and the survey
names the open direction verbatim: "adaptive memory orchestration — learned
controllers that dynamically allocate across memory subsystems". Mutation:
PER-EXAMPLE routing over HETEROGENEOUS exact-memory ORGANS sharing one
linear host, with exact-oracle dCE at micro scale.

One model, three task families (disjoint token ranges, shared vocab 45):
  T0 Dyck-echo k=2 (6 tokens, range 0)  -> exact-STACK organ
  T1 ICL 16-key cipher (32 tokens, 6)   -> exact-SRAM organ
  T2 mod-7 walk (7 tokens, 38)          -> host only (finite-state)
Components (all from this repo's certified lines):
  shared SSM host d16/1blk + zero-init host head (L-GATE-INIT)
  stack organ: (top, empty, prevC) table 8x45 + exact K-stack
  sram organ:  16 slots x d16, exact causal writes, W_readout(d16->45)
  router: MLP over h[:, :3] -> 3 tasks, argmax, DIRECT-CE supervised
          (L-DIRECT-GRADIENT => deterministic routing, L-RELIABLE-EXACT)
  logits = host_head(h) + organ[router(x)]   (per-example hard selection,
          L-ROUTING-BEATS-FUSION)
Baseline (protocol): micro TF d64/2L, same mixed stream, 2500 steps.
Win = per-task dCE @4096 ~= the standalone certified lines (echo ~0.01-0.03,
ICL target ~0.03-0.12, mod7 ~0.003-0.01) with routing ~1.0.
USAGE: OMP_NUM_THREADS=1 python3 -u unified.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
V_ECHO, V_ICL, V_MOD7 = 6, 32, 7
OFF_ICL, OFF_MOD7 = V_ECHO, V_ECHO + V_ICL          # 6, 38
VOCAB = V_ECHO + V_ICL + V_MOD7                     # 45
NK = 16
LN2, LN3, LN6, LN16 = math.log(2.0), math.log(3.0), math.log(6.0), math.log(16.0)
CFG = dict(steps=2500, batch=32, n_echo=12, n_icl=10, n_mod7=10, train_len=63,
           eval_lens=[4096], eval_reps=2, d_model=16, KSTACK=4096,
           p_rise=0.9, p_fall=0.02)
print(f"[setup] unified cfg={CFG} VOCAB={VOCAB}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- data
echo_src = open("dyck_echo.py").read()
gE = {"__name__": "u1"}
exec(echo_src.split("\n# ---------------------------------------------------------------- experiment")[0], gE)
strip_src = open("ssm_strip.py").read()
gS = {"__name__": "u2"}
exec(strip_src.split("\n# ---------------------------------------------------------------- experiment")[0], gS)
sram_src = open("sram_icl.py").read()
gR = {"__name__": "u3"}
exec(sram_src.split("\nRESULTS = {}")[0].split("# ---------------------------------------------------------------- host")[0], gR)
gen_icl = gR["gen_icl"]

def gen_echo_t(batch, length, rng):
    x, y, o = gE["gen_echo"](batch, length, rng)
    return x, y, o

def gen_icl_t(batch, length, rng):
    x, y, o = gen_icl(batch, length, rng)
    return x + OFF_ICL, y + OFF_ICL, o

def gen_mod7_t(batch, length, rng):
    x, y, o = gS["gen_mod7"](batch, length, rng)
    shift = OFF_MOD7 - 20          # ssm_strip mod7 tokens start at 20
    return x + shift, y + shift, o

def gen_mixed(batch, length, rng):
    """train stream: all families at 63 tokens (echo sliced 64->63, icl
    natural 63 = (64-2)/2 pairs + query, mod7 63). oracle not needed for
    training (per-family pooled oracles stay in eval)"""
    n0, n1 = CFG["n_echo"], CFG["n_icl"]
    x0, y0, _ = gen_echo_t(n0, 64, rng)
    x0, y0 = x0[:, :length], y0[:, :length]
    x1, y1, _ = gen_icl_t(n1, 64, rng)
    x2, y2, _ = gen_mod7_t(batch - n0 - n1, length, rng)
    return (torch.cat([x0, x1, x2]), torch.cat([y0, y1, y2]),
            torch.cat([torch.zeros(n0), torch.ones(n1),
                       torch.full((batch - n0 - n1,), 2.0)]).long())

# ---------------------------------------------------------------- model
def n_params(m):
    return sum(p.numel() for p in m.parameters())

class SSMBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.Wx = nn.Linear(d, d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)
        self.log_a = nn.Parameter(-3.0 * torch.ones(d))
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h):
        x = self.Wx(self.n1(h))
        a = 1.0 - torch.exp(-F.softplus(self.log_a))
        B, L, d = x.shape
        S = torch.zeros(B, d, device=x.device)
        outs = []
        for t in range(L):
            S = a * S + x[:, t]
            outs.append(self.Wo(S))
        o = torch.stack(outs, 1)
        h = h + o
        return h + self.mlp(self.n2(h))

class UnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.block = SSMBlock(d)
        self.norm = nn.LayerNorm(d)
        self.host_head = nn.Linear(d, VOCAB)
        nn.init.zeros_(self.host_head.weight)     # L-GATE-INIT
        nn.init.zeros_(self.host_head.bias)
        self.router = nn.Sequential(nn.Linear(3 * d, 16), nn.GELU(), nn.Linear(16, 3))
        # stack organ (k=2 echo tokens: opens 0,1; C=2)
        self.stack_table = nn.Parameter(0.1 * torch.randn(8, VOCAB))
        # sram organ (ICL keys 6..21, values 22..37)
        self.W_readout = nn.Linear(d, VOCAB)

    def host_hiddens(self, x):
        h = self.emb(x)
        h = self.block(h)
        return self.norm(h)

    def stack_logits(self, x):
        B, L = x.shape
        d = CFG["d_model"]
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
                empty = 1 if not stack else 0
                top = stack[-1] if stack else 0
                prevC = 1 if tok == 2 else 0
                feats[b, t, 0] = top
                feats[b, t, 1] = empty
                feats[b, t, 2] = prevC
        combo = feats[:, :, 0] + feats[:, :, 1] * 2 + feats[:, :, 2] * 4
        return self.stack_table[combo]

    def sram_logits(self, x, h):
        B, L = x.shape
        d = CFG["d_model"]
        K0, K1, V0, V1 = OFF_ICL, OFF_ICL + NK, OFF_ICL + NK, OFF_ICL + 2 * NK
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
                    last_key[b] = int(k[b])      # shifted index, not raw token
        return out

    def forward(self, x):
        h = self.host_hiddens(x)
        route_logits = self.router(h[:, :3].reshape(x.shape[0], -1))
        task = route_logits.argmax(-1)
        sl = self.stack_logits(x)
        rl = self.sram_logits(x, h)
        organ = torch.where(task[:, None, None] == 0, sl,
               torch.where(task[:, None, None] == 1, rl, sl * 0.0))
        return self.host_head(h) + organ, route_logits

# ---------------------------------------------------------------- train/eval
def train_model(model, seed=0, tag="uni"):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        x, y, task = gen_mixed(CFG["batch"], CFG["train_len"], rng)
        logits, rl = model(x)
        l_lm = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        l_rt = F.cross_entropy(rl, task)
        loss = l_lm + 0.5 * l_rt
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 500 == 0:
            print(f"  [{tag} s{seed}] step {step}/{CFG['steps']} lm {l_lm.item():.4f} "
                  f"rt {l_rt.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    return model

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

# ---------------------------------------------------------------- experiment
RESULTS = {}
torch.manual_seed(0)
uni = UnifiedModel()
print(f"[arm] unified (host + stack + sram + router) params={n_params(uni)}", flush=True)
train_model(uni, 0)
torch.save(uni.state_dict(), "unified_s0.pt")
r = {}
r["echo"] = eval_task(uni, gen_echo_t, 4096, 2, 0)
r["icl"] = eval_task(uni, gen_icl_t, 4096, 2, 1, tgt=True)
r["mod7"] = eval_task(uni, gen_mod7_t, 4096, 2, 2)
r["params"] = n_params(uni)
RESULTS["unified_s0"] = r
print(f"  {r}", flush=True)
del uni

# protocol baseline: micro TF on the same mixed stream
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
print(f"[arm] micro TF (protocol baseline) params={n_params(tf)}", flush=True)
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
    if step % 1000 == 0:
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
RESULTS["micro_tf"] = r
print(f"  {r}", flush=True)

print("\n" + "=" * 84)
print("UNIFIED MACHINE @4096 (dCE per task; echo standalone cert 0.0106, icl")
print("target cert 0.022-0.027, mod7 cert 0.0025-0.0071) | icl = total|target")
print("=" * 84)
for k, v in RESULTS.items():
    print(f"{k:<12} params {v['params']:<7} echo {v['echo']}  icl {v['icl']}  mod7 {v['mod7']}", flush=True)
print("=" * 84)
final = {"tag": "ARC2-C13-UNIFIED", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
