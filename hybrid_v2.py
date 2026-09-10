"""
ARC-2 CYCLE 7 / P11 - HYBRID-V2: RoPE attention + GATED register-machine coprocessor
======================================================================================
Self-contained, CPU-runnable (GPU works too). Continues ssr_lab hybrid line:
  round 1 (hybrid_h2h.py)   : parallel coprocessor, late fusion into logits -> FAILED
  round 2 (hybrid_h2h_r2.py): curriculum hybrid, sinusoidal TF baseline
  hybrid-v2 (THIS FILE)     : RoPE backbone (the fair strong baseline) + coprocessor
                              reading the TF's own hidden states, per-token GATED
                              residual fusion. Gate init -2 => starts as pure TF
                              (L-GATE-INIT); structured-first curriculum
                              (L-POLY-INTERFERENCE mitigation).

The claim: ONE model, no task switching, that is fluency-level on the statistical
language (attention's home turf) AND oracle-level on the structured language at
64x training length (machine's home turf, where transformers provably decay).
Win = hybrid dominates tf_rope at long structured lengths AND dominates the pure
machine at short fuzzy lengths, with zero length decay on structured.

Benchmark (analytic oracles -> dCE = model CE - oracle CE, nats/token):
  FUZZY      : order-2 Markov over 16 tokens (statistical; no algorithmic structure)
  STRUCTURED : stochastic bounded-depth Dyck-2 (requires exact stack state)
Train L=64, eval L=64/512/2048/4096.

USAGE: python3 hybrid_v2.py            (CPU smoke = 1 seed)
       FORCE_FULL=1 python3 hybrid_v2.py   (2 seeds per arm)
Prints results table + PASTE-BACK JSON at the end.
"""
import json, math, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, torch.get_num_threads()))
FULL = (DEVICE == "cuda") or os.environ.get("FORCE_FULL") == "1"
CFG = dict(
    seeds=[0, 1] if FULL else [0],
    steps=9000 if FULL else 2500,
    batch=64 if FULL else 32,
    train_len=64,
    eval_lens=[64, 512, 2048, 4096],
    eval_reps=3 if FULL else 2,
    d_model=64,
    n_layers=2,
    n_heads=4,
    k=12, d_slot=16,          # coprocessor: 12 slots, 16 fixed instructions
)
print(f"[setup] device={DEVICE} full={FULL} cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- benchmark data
V_MARKOV, V_DYCK = 16, 4
VOCAB = V_MARKOV + V_DYCK
_g = torch.Generator().manual_seed(777)
MARKOV_T = F.softmax(1.5 * torch.randn(V_MARKOV, V_MARKOV, V_MARKOV,
                                       generator=_g), dim=-1)
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

def gen_mixed(batch, length, rng):
    xs, ys = [], []
    for _ in range(batch):
        gen = gen_markov if rng.random() < 0.5 else gen_dyck
        x, y, _ = gen(1, length, rng)
        xs.append(x[0]); ys.append(y[0])
    return torch.stack(xs), torch.stack(ys)

# ---------------------------------------------------------------- coprocessor
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
    """ISA register machine: 16 fixed instructions (8 role perms x write-bit)."""
    def __init__(self, d_model, k=12, d_slot=16):
        super().__init__()
        self.k, self.d_slot, self.hard = k, d_slot, False
        self.register_buffer("PH", torch.stack([role_basis(k)[o % 8]
                                                for o in range(16)]))
        self.register_buffer("gbits", (torch.arange(16) < 8).float())
        self.alpha = nn.Linear(d_model, 16)
        nn.init.zeros_(self.alpha.bias)
        self.alpha.bias.data[8] = 2.0        # L-GATE-INIT: identity+write at init
        self.readq = nn.Linear(d_model, k)
        self.beta = nn.Linear(d_model, 16)
        self.vcode = nn.Parameter(torch.randn(16, d_slot))
        self.wlog = nn.Parameter(torch.randn(16, k))
        self.S0 = nn.Parameter(0.5 * torch.randn(k, d_slot))
        self.out = nn.Linear(d_slot + k * d_slot, d_model)
        nn.init.zeros_(self.out.weight)      # L-GATE-INIT: zero contribution at init
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
        return self.norm(h + self.out(o)), o    # residual state readout

# ---------------------------------------------------------------- architectures
class RoPEBlock(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh, self.dh = nh, d // nh
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, cos, sin):
        B, L, d = h.shape
        q, k, v = self.qkv(self.n1(h)).chunk(3, -1)
        q = q.view(B, L, self.nh, self.dh).transpose(1, 2)
        k = k.view(B, L, self.nh, self.dh).transpose(1, 2)
        v = v.view(B, L, self.nh, self.dh).transpose(1, 2)
        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]
        c, s = cos[None, None, :, :], sin[None, None, :, :]
        q = torch.stack([q1 * c - q2 * s, q1 * s + q2 * c], dim=-1).flatten(-2)
        k = torch.stack([k1 * c - k2 * s, k1 * s + k2 * c], dim=-1).flatten(-2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        h = h + self.proj(o.transpose(1, 2).reshape(B, L, d))
        return h + self.mlp(self.n2(h))

def rope_cache(L, dh):
    pos = torch.arange(L, device=DEVICE).float()
    inv = 1.0 / (10000 ** (torch.arange(0, dh, 2, device=DEVICE).float() / dh))
    ang = pos[:, None] * inv[None, :]
    return ang.cos(), ang.sin()

class TransformerLM(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([RoPEBlock(d, CFG["n_heads"])
                                     for _ in range(CFG["n_layers"])])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        cos, sin = rope_cache(x.shape[1], CFG["d_model"] // CFG["n_heads"])
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.head(self.norm(h))

class MachineLM(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.cop = CopLayer(d, CFG["k"], CFG["d_slot"])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        h, _ = self.cop(h)
        return self.head(self.norm(h))

class HybridV2(nn.Module):
    """RoPE backbone + GATED coprocessor on top of the TF hidden states.
    h' = h_tf + g_t * proj(cop_state_t);  g = sigmoid(Wg h_tf), init bias -2.
    Gate starts ~0.12 and out-proj starts at zero => model starts as pure TF."""
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.tf = TransformerLM.__new__(TransformerLM)
        TransformerLM.__init__(self.tf)
        self.cop = CopLayer(d, CFG["k"], CFG["d_slot"])
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)     # g0 ~ 0.12: TF-dominant start

    def forward(self, x):
        h = self.tf.emb(x)
        cos, sin = rope_cache(x.shape[1], CFG["d_model"] // CFG["n_heads"])
        for blk in self.tf.blocks:
            h = blk(h, cos, sin)
        hn, _ = self.cop(h)
        g = torch.sigmoid(self.gate(h))
        return self.tf.head(self.tf.norm(h + g * (hn - h)))

def set_hard(model, hard):
    for m in model.modules():
        if isinstance(m, CopLayer):
            m.hard = hard

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- train / eval
def train(model, seed, tag, curriculum=False, structured_only=False):
    model.train().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        if structured_only or (curriculum and step <= int(0.4 * CFG["steps"])):
            x, y = (lambda z: (z[0], z[1]))(gen_dyck(CFG["batch"], CFG["train_len"], rng))
        else:
            x, y = gen_mixed(CFG["batch"], CFG["train_len"], rng)
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % max(1, CFG["steps"] // 4) == 0:
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
        x, y = x.to(DEVICE), y.to(DEVICE)
        lp = F.log_softmax(model(x), -1)
        ce += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
        orc += o.sum().item(); n += y.numel()
    return (ce - orc) / n

def gate_report(model):
    with torch.no_grad():
        x = torch.randint(0, VOCAB, (4, 32), device=DEVICE)
        model.eval()
        if isinstance(model, HybridV2):
            h = model.tf.emb(x)
            cos, sin = rope_cache(32, CFG["d_model"] // CFG["n_heads"])
            for blk in model.tf.blocks:
                h = blk(h, cos, sin)
            g = torch.sigmoid(model.gate(h))
            return round(g.mean().item(), 3)
    return None

def evaluate(model, has_cop):
    res = {}
    modes = [("hard", True), ("soft", False)] if has_cop else [("", None)]
    for mname, hard in modes:
        if hard is not None:
            set_hard(model, hard)
        for task in TASKS:
            for L in CFG["eval_lens"]:
                key = f"{task}@{L}" if not mname else f"{task}@{L}[{mname}]"
                res[key] = round(eval_dce(model, task, L, CFG["eval_reps"]), 4)
    if has_cop:
        L0, L1 = CFG["eval_lens"][0], CFG["eval_lens"][-1]
        d0 = res[f"structured@{L0}[hard]"] - res[f"structured@{L0}[soft]"]
        d1 = res[f"structured@{L1}[hard]"] - res[f"structured@{L1}[soft]"]
        res["crystallized_gate"] = bool(d0 < 0.05 and d1 <= 0.02)
        set_hard(model, True)
    res["gate_mean"] = gate_report(model)
    return res

# ---------------------------------------------------------------- experiment
ARCHS = {
    "tf_rope":     (lambda: TransformerLM(), {}),
    "machine":     (lambda: MachineLM(), dict(structured_only=True)),
    "machine_mixed": (lambda: MachineLM(), {}),
    "hybrid_v2":   (lambda: HybridV2(), dict(curriculum=True)),
}
ALL = {}
for seed in CFG["seeds"]:
    for name, (ctor, tkw) in ARCHS.items():
        torch.manual_seed(seed)
        model = ctor().to(DEVICE)
        print(f"[run] {name} seed={seed} params={n_params(model)} "
              f"(elapsed {(time.time()-t_start)/60:.1f}min)", flush=True)
        train(model, seed, f"{name}-s{seed}", **tkw)
        torch.save(model.state_dict(), f"hyb2_{name}_s{seed}.pt")
        ALL[f"{name}_s{seed}"] = evaluate(model, has_cop="machine" in name
                                          or "hybrid" in name)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

print("\n" + "=" * 100)
print("RESULTS  dCE = excess CE over analytic oracle (nats/token; 0 = perfect)")
print("=" * 100)
hdr = f"{'run':<22}" + "".join(f"{t}@{L:<6}[:13]"[:13].ljust(13)
                               for t in TASKS for L in CFG["eval_lens"])
print(hdr, flush=True)
for run, r in ALL.items():
    cells = []
    for t in TASKS:
        for L in CFG["eval_lens"]:
            k = f"{t}@{L}[hard]" if f"{t}@{L}[hard]" in r else f"{t}@{L}"
            cells.append(f"{r[k]:<13}")
    gate = r.get("crystallized_gate", "-")
    g = r.get("gate_mean", "-")
    print(f"{run:<22}" + "".join(cells) + f"  gate={gate}  g={g}", flush=True)
print("=" * 100)

summary = dict(experiment="ARC2-C7-hybrid-v2", device=DEVICE, full=FULL, cfg=CFG,
               results=ALL, wall_min=round((time.time() - t_start) / 60, 1))
print("\n########## PASTE-BACK BLOCK ##########")
print(json.dumps(summary))
print("############## END ###################")
open("results.jsonl", "a").write(json.dumps(summary) + "\n")
