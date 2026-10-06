"""
ARC-2 CYCLE 8 / P11+P1 - LINATTHOST: sub-quadratic SSM host + discrete organ
=============================================================================
Question (the attention-bottleneck axis): the hybrid-v2 negative showed the
machine arm's structured long-context advantage does not survive as a residual
on an ATTENTION host. Does a LINEAR-COMPLEXITY host (no attention at all) change
the picture? Two claims under test:
  (a) a from-scratch diagonal-SSM host alone vs the RoPE host on the same
      benchmark (fuzzy order-2 Markov + structured bounded Dyck-2, train L=64,
      eval 64/512/2048/4096, analytic-oracle dCE);
  (b) SSM host + the same ISA register coprocessor organ (residual fusion,
      L-GATE-INIT) -> does the organ lift the linear host to machine-arm level
      on structured@4096, BEATING the RoPE host (v2 numbers: tf 1.304,
      machine 0.8425, hybrid 2.2667)?

Architecture (from scratch, O(L*d) per token, no pairwise terms):
  SSMBlock: h_t = diag(a) h_{t-1} + W_x x_t,  o_t = W_o h_t,  a = 1-exp(-softplus(log_a))
            + residual + SwiGLU-ish MLP + LayerNorm. Two blocks. d=64.
  SSMHost = emb -> SSMBlock x2 -> norm -> head (tied).
  SSMHostCop = SSMHost backbone + CopLayer organ (reads host hidden states).

Controls: reload hyb2_tf_rope_s0.pt and hyb2_machine_s0.pt (same benchmark,
same seed, same budget) -- no retraining needed.

Win = (b) structured@4096 <= 0.90 AND <= tf_rope (1.304) AND fuzzy@64 <= 0.10.
USAGE: OMP_NUM_THREADS=1 python3 -u linattn_host.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
CFG = dict(steps=2500, batch=32, train_len=64,
           eval_lens=[64, 512, 2048, 4096], eval_reps=2,
           d_model=64, n_layers=2, k=12, d_slot=16)
print(f"[setup] linattn_host cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- benchmark data (identical to hybrid_v2)
V_MARKOV, V_DYCK = 16, 4
VOCAB = V_MARKOV + V_DYCK
_g = torch.Generator().manual_seed(777)
MARKOV_T = F.softmax(1.5 * torch.randn(V_MARKOV, V_MARKOV, V_MARKOV, generator=_g), dim=-1)
MARKOV_P = MARKOV_T.tolist()
MARKOV_CUM = MARKOV_T.cumsum(-1).tolist()

def gen_markov(batch, length, rng):
    import bisect
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        seq = [rng.randrange(V_MARKOV), rng.randrange(V_MARKOV)]
        nll = [math.log(V_MARKOV), math.log(V_MARKOV)]
        while len(seq) < length + 1:
            a, b = seq[-2], seq[-1]
            nxt = min(bisect.bisect(MARKOV_CUM[a][b], rng.random()), V_MARKOV - 1)
            seq.append(nxt)
            nll.append(-math.log(max(MARKOV_P[a][b][nxt], 1e-12)))
        xs.append(seq[:length]); ys.append(seq[1:length + 1])
        os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

def gen_dyck(batch, length, rng, D=6):
    L2, L4 = math.log(2.0), math.log(4.0)
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll, stack = [], [], []
        for _ in range(length + 1):
            d = len(stack)
            if d == 0:
                t = rng.randrange(2); stack.append(t)
                x.append(2 * t); nll.append(L2)
            elif d == D:
                t = stack.pop(); x.append(2 * t + 1); nll.append(0.0)
            else:
                if rng.random() < 0.5:
                    t = rng.randrange(2); stack.append(t)
                    x.append(2 * t); nll.append(L4)
                else:
                    t = stack.pop(); x.append(2 * t + 1); nll.append(L2)
        xs.append([v + V_MARKOV for v in x[:length]])
        ys.append([v + V_MARKOV for v in x[1:length + 1]])
        os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

TASKS = {"fuzzy": gen_markov, "structured": gen_dyck}

# ---------------------------------------------------------------- organ (copied verbatim from hybrid_v2)
def role_basis(k):
    def shift(block, d):
        P = torch.zeros(k, k)
        m = {i: i for i in range(k)}
        for idx, i in enumerate(block):
            m[i] = block[(idx + d) % len(block)]
        for i in range(k):
            P[m[i], i] = 1.0
        return P
    h = k // 2
    A, B, Fu = list(range(h)), list(range(h, k)), list(range(k))
    return torch.stack([torch.eye(k), shift(A, 1), shift(A, -1), shift(B, 1),
                        shift(B, -1), shift(A, 1) @ shift(B, 1),
                        shift(Fu, 1), shift(Fu, -1)])

def st_onehot(p):
    h = torch.zeros_like(p).scatter_(-1, p.argmax(-1, keepdim=True), 1.0)
    return h + p - p.detach()

class CopLayer(nn.Module):
    def __init__(self, d_model, k=12, d_slot=16):
        super().__init__()
        self.k, self.d_slot, self.hard = k, d_slot, False
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8]
                                                for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.alpha = nn.Linear(d_model, 16)
        nn.init.zeros_(self.alpha.bias)
        self.alpha.bias.data[8] = 2.0
        self.readq = nn.Linear(d_model, k)
        self.beta = nn.Linear(d_model, 16)
        self.vcode = nn.Parameter(torch.randn(16, d_slot))
        self.wlog = nn.Parameter(torch.randn(16, k))
        self.S0 = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.out = nn.Linear(d_slot + k * d_slot, d_model)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h):
        B, L, _ = h.shape
        a = F.softmax(self.alpha(h), -1)
        q = F.softmax(self.readq(h), -1)
        beta = F.softmax(self.beta(h), -1)
        w = F.softmax(self.wlog, -1)
        if self.hard:
            a, q, beta, w = st_onehot(a), st_onehot(q), st_onehot(beta), st_onehot(w)
        gb = self.gbits.view(-1, 1, 1)
        Mo = self.PH - gb * torch.einsum("oij,oj,ol->oil", self.PH, w, w)
        u = self.gbits.unsqueeze(-1) * torch.einsum("oij,oj->oi", self.PH, w)
        A = torch.einsum("blo,oij->blij", a, Mo)
        uv = torch.einsum("blo,oi->bli", a, u)
        v = beta @ self.vcode
        b = uv.unsqueeze(-1) * v.unsqueeze(-2)
        S = self.S0.expand(B, -1, -1)
        outs = []
        for t in range(L):
            S = torch.bmm(A[:, t], S) + b[:, t]
            r = torch.einsum("bk,bkd->bd", q[:, t], S)
            outs.append(torch.cat([r, S.reshape(B, -1)], -1))
        o = torch.stack(outs, 1)
        return self.norm(h + self.out(o)), o

# ---------------------------------------------------------------- SSM host (from scratch)
class SSMBlock(nn.Module):
    """Diagonal state-space: h_t = diag(a) h_{t-1} + W_x x_t ; o_t = W_o h_t.
    a = 1 - exp(-softplus(log_a)) in (0,1): per-channel learned decay
    (softplus large -> a->1 -> long memory). O(L*d) time, O(d) state."""
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

class SSMHost(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([SSMBlock(d) for _ in range(CFG["n_layers"])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

class SSMHostCop(nn.Module):
    """SSM backbone + ISA coprocessor organ (residual, L-GATE-INIT)."""
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.host = SSMHost()
        self.cop = CopLayer(d, CFG["k"], CFG["d_slot"])

    def forward(self, x):
        h = self.host.emb(x)
        for blk in self.host.blocks:
            h = blk(h)
        h, _ = self.cop(h)
        return self.host.head(self.host.norm(h))

def set_hard(model, hard):
    for m in model.modules():
        if isinstance(m, CopLayer):
            m.hard = hard

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- train / eval (same recipe as hybrid_v2)
def train(model, seed, tag):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        xm, ym, _ = gen_markov(CFG["batch"] // 2, CFG["train_len"], rng)
        xd, yd, _ = gen_dyck(CFG["batch"] // 2, CFG["train_len"], rng)
        x, y = torch.cat([xm, xd]), torch.cat([ym, yd])
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 500 == 0:
            print(f"  [{tag}] step {step}/{CFG['steps']} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_dce(model, task, L, reps):
    model.eval()
    bs = max(1, min(8, 4096 // L))
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(50_000 + L + i)
        x, y, o = TASKS[task](bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        ce += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
        orc += o.sum().item(); n += y.numel()
    return (ce - orc) / n

# ---------------------------------------------------------------- experiment
ALL = {}
for name, ctor in (("ssm", SSMHost), ("ssm_cop", SSMHostCop)):
    torch.manual_seed(0)
    model = ctor().to(DEVICE)
    print(f"[run] {name} params={n_params(model)} (elapsed {(time.time()-t_start)/60:.1f}min)", flush=True)
    train(model, 0, f"{name}-s0")
    torch.save(model.state_dict(), f"linattn_{name}_s0.pt")
    set_hard(model, True) if "cop" in name else None
    ALL[name] = {f"{t}@{L}": round(eval_dce(model, t, L, CFG["eval_reps"]), 4)
                 for t in TASKS for L in CFG["eval_lens"]}
    del model

# controls: reload hybrid-v2 checkpoints (same benchmark, seed 0, same budget)
import sys
sys.path.insert(0, ".")
_hv2 = {}
_src = open("hybrid_v2.py").read()
_hv2["torch"] = torch; _hv2["nn"] = nn; _hv2["F"] = F; _hv2["DEVICE"] = DEVICE
_hv2["CFG"] = dict(d_model=64, n_layers=2, n_heads=4, k=12, d_slot=16)
_hv2["VOCAB"] = VOCAB
_hv2code = _src[_src.index("def role_basis"):_src.index("class HybridV2")]
_hv2code = _hv2code.replace("def train(", "def _train_v2(").replace("def eval_dce(", "def _eval_v2(")
exec(_hv2code, _hv2)
tf = _hv2["TransformerLM"](); tf.load_state_dict(torch.load("hyb2_tf_rope_s0.pt", weights_only=True))
mach = _hv2["MachineLM"](); mach.load_state_dict(torch.load("hyb2_machine_s0.pt", weights_only=True))
set_hard(mach, True)
ALL["tf_rope(v2)"] = {f"{t}@{L}": round(eval_dce(tf, t, L, CFG["eval_reps"]), 4)
                      for t in TASKS for L in CFG["eval_lens"]}
ALL["machine(v2)"] = {f"{t}@{L}": round(eval_dce(mach, t, L, CFG["eval_reps"]), 4)
                      for t in TASKS for L in CFG["eval_lens"]}

print("\n" + "=" * 100)
print("RESULTS  dCE = excess CE over analytic oracle (nats/token; 0 = perfect)")
print("=" * 100)
hdr = f"{'run':<16}" + "".join(f"{t}@{L:<5}".ljust(13) for t in TASKS for L in CFG["eval_lens"])
print(hdr, flush=True)
for run, r in ALL.items():
    print(f"{run:<16}" + "".join(f"{r[f'{t}@{L}']:<13}" for t in TASKS for L in CFG["eval_lens"]), flush=True)
print("=" * 100)
sc = ALL["ssm_cop"]["structured@4096"]; tfc = ALL["tf_rope(v2)"]["structured@4096"]
fc = ALL["ssm_cop"]["fuzzy@64"]
win = sc <= 0.90 and sc <= tfc and fc <= 0.10
verdict = (f"WIN: SSM+organ structured@4096 {sc} beats RoPE host {tfc} "
           f"and is at machine level, fuzzy@64 {fc}" if win else
           f"NOT-WIN: ssm_cop structured@4096={sc} (tf_rope={tfc}), fuzzy@64={fc}")
final = {"tag": "ARC2-C8-LINATT-HOST", "runs": ALL, "verdict": verdict,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
