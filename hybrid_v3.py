"""
ARC-2 CYCLE 8 / P11b - HYBRID-V3: MIXTURE-OF-ARCHITECTURES (MoA)
=================================================================
Fixes the logged hybrid-v2 negative (gate on h_tf collapsed to ~0.94; the
machine arm's structured long-context advantage did not survive as a TF
residual). Recorded next step: "per-example HARD selection with per-arm losses".

Design (Mixture-of-Architectures):
  experts :  (1) TransformerLM (RoPE attention)      -- continuous paradigm
             (2) MachineLM (ISA register coprocessor) -- discrete paradigm
  router  : tiny MLP over the first 8 tokens -> 2 classes (fuzzy/structured)
  training: per-arm LOSSES -- the attention arm sees ONLY the Markov (fuzzy)
            task, the machine arm ONLY Dyck (structured); the router sees
            both with task labels. Hard per-example selection at eval.

Literature position (searched 2026-08-21): heterogeneous-expert MoE exists
on the CAPACITY axis (MoHGE big.LITTLE; coarse-grained MoE over frozen
heterogeneous LLMs, Liu et al. 2025). Nobody routes on the PARADIGM axis
(continuous attention vs discrete state machine) with a length-extrapolation
reasoning benchmark. This is that minimal proof.

Win = system dCE <= best single arm per input on BOTH tasks across all four
eval lengths (i.e. structured@4096 at machine level ~0.84, fuzzy@64 at TF
level ~0.05), with routing accuracy ~1.0.

USAGE: OMP_NUM_THREADS=1 python3 -u hybrid_v3.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
CFG = dict(seeds=[0], steps=2500, batch=32, train_len=64,
           eval_lens=[64, 512, 2048, 4096], eval_reps=2,
           d_model=64, n_layers=2, n_heads=4, k=12, d_slot=16,
           route_prefix=8)
print(f"[setup] hybrid_v3 MoA cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- benchmark data
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

# ---------------------------------------------------------------- architectures
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

class Router(nn.Module):
    """Tiny prefix classifier: first K tokens -> task class (0=fuzzy, 1=structured)."""
    def __init__(self, K):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.fc = nn.Sequential(nn.Linear(K * d, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, x):
        h = self.emb(x[:, :CFG["route_prefix"]]).flatten(1)
        return self.fc(h)

def set_hard(model, hard):
    for m in model.modules():
        if isinstance(m, CopLayer):
            m.hard = hard

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- training
def train_all(tf, mach, rout, seed):
    for name, m in (("tf", tf), ("mach", mach), ("router", rout)):
        m.train()
    opt_tf = torch.optim.AdamW(tf.parameters(), lr=3e-3)
    opt_m = torch.optim.AdamW(mach.parameters(), lr=3e-3)
    opt_r = torch.optim.AdamW(rout.parameters(), lr=3e-2)
    rng = random.Random(seed * 1000 + 17)
    n = CFG["batch"] // 2
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        xm, ym, _ = gen_markov(n, CFG["train_len"], rng)
        xd, yd, _ = gen_dyck(n, CFG["train_len"], rng)
        xmix = torch.cat([xm, xd]); ymix = torch.cat([ym, yd])
        labels = torch.cat([torch.zeros(n), torch.ones(n)]).long()
        # per-arm losses (hard specialization)
        opt_tf.zero_grad(); opt_m.zero_grad(); opt_r.zero_grad()
        l_tf = F.cross_entropy(tf(xm).reshape(-1, VOCAB), ym.reshape(-1))
        l_m = F.cross_entropy(mach(xd).reshape(-1, VOCAB), yd.reshape(-1))
        l_r = F.cross_entropy(rout(xmix), labels)
        (l_tf + l_m + l_r).backward()
        torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(mach.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(rout.parameters(), 1.0)
        opt_tf.step(); opt_m.step(); opt_r.step()
        if step % 500 == 0:
            print(f"  [moa] step {step}/{CFG['steps']} tf {l_tf.item():.4f} "
                  f"mach {l_m.item():.4f} router {l_r.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    set_hard(mach, True)

# ---------------------------------------------------------------- evaluation
@torch.no_grad()
def route(x, rout):
    return rout(x).argmax(-1).cpu()

@torch.no_grad()
def eval_routed(task, L, reps, tf, mach, rout):
    bs = max(1, min(8, 4096 // L))
    ce = orc = n = 0.0
    correct = total = 0
    for i in range(reps):
        rng = random.Random(50_000 + L + i)
        x, y, o = TASKS[task](bs, L, rng)
        r = route(x, rout)
        true = 1 if task == "structured" else 0
        for b in range(bs):
            arm = tf if r[b] == 0 else mach
            lp = F.log_softmax(arm(x[b:b+1]), -1)
            ce += -lp.gather(-1, y[b:b+1].unsqueeze(-1)).sum().item()
            orc += o[b].sum().item(); n += y[b].numel()
            correct += int(r[b] == true); total += 1
    return (ce - orc) / n, correct / total

@torch.no_grad()
def eval_arm(task, L, reps, arm):
    bs = max(1, min(8, 4096 // L))
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(50_000 + L + i)
        x, y, o = TASKS[task](bs, L, rng)
        lp = F.log_softmax(arm(x), -1)
        ce += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
        orc += o.sum().item(); n += y.numel()
    return (ce - orc) / n

# ---------------------------------------------------------------- experiment
torch.manual_seed(0)
tf = TransformerLM(); mach = MachineLM(); rout = Router(CFG["route_prefix"])
print(f"[run] tf params={n_params(tf)} mach params={n_params(mach)} "
      f"router params={n_params(rout)}", flush=True)
train_all(tf, mach, rout, 0)
torch.save({"tf": tf.state_dict(), "mach": mach.state_dict(),
            "router": rout.state_dict()}, "moa_v3_s0.pt")
print("[ckpt] moa_v3_s0.pt saved", flush=True)

res = {"routing": {}, "system": {}}
res["tf_arm"] = {f"{t}@{L}": round(eval_arm(t, L, CFG["eval_reps"], tf), 4)
                 for t in TASKS for L in CFG["eval_lens"]}
res["mach_arm"] = {f"{t}@{L}": round(eval_arm(t, L, CFG["eval_reps"], mach), 4)
                   for t in TASKS for L in CFG["eval_lens"]}
for t in TASKS:
    for L in CFG["eval_lens"]:
        dce, acc = eval_routed(t, L, CFG["eval_reps"], tf, mach, rout)
        res["system"][f"{t}@{L}"] = round(dce, 4)
        res["routing"][f"{t}@{L}"] = round(acc, 4)
        print(f"[eval] {t}@{L}: system dCE {dce:.4f} (tf {res['tf_arm'][f'{t}@{L}']:.4f} "
              f"mach {res['mach_arm'][f'{t}@{L}']:.4f}) routing acc {acc:.3f}", flush=True)

best_of_arms = {f"{t}@{L}": min(res["tf_arm"][f"{t}@{L}"], res["mach_arm"][f"{t}@{L}"])
                for t in TASKS for L in CFG["eval_lens"]}
wins = sum(1 for k in res["system"]
           if res["system"][k] <= min(res["tf_arm"][k], res["mach_arm"][k]) + 1e-9)
verdict = "WIN: system <= best arm on every (task,length)" if wins == len(res["system"]) else \
          f"PARTIAL: system <= best arm on {wins}/{len(res['system'])} (task,length)"
res["verdict"] = verdict

print("\n" + "=" * 100)
print("RESULTS  dCE = excess CE over analytic oracle (nats/token; 0 = perfect)")
print("=" * 100)
hdr = f"{'run':<14}" + "".join(f"{t}@{L:<5}".ljust(13) for t in TASKS for L in CFG["eval_lens"])
print(hdr, flush=True)
for run, r in (("tf_arm", res["tf_arm"]), ("mach_arm", res["mach_arm"]),
               ("BEST-ARM", best_of_arms), ("MOA_SYSTEM", res["system"])):
    print(f"{run:<14}" + "".join(f"{r[f'{t}@{L}']:<13}" for t in TASKS for L in CFG["eval_lens"]), flush=True)
print("=" * 100)
res["wall_s"] = round(time.time() - t_start, 1)
res["peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
final = {"tag": "ARC2-C8-MOA-V3", "tf_arm": res["tf_arm"], "mach_arm": res["mach_arm"],
         "system": res["system"], "routing": res["routing"], "verdict": res["verdict"],
         "wall_s": res["wall_s"], "peak_mb": res["peak_mb"]}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
