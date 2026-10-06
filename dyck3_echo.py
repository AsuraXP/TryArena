"""
ARC-2 CYCLE 11 / DYCK-3-ECHO: does the minimal-state organ generalize in k?
===========================================================================
Dyck-echo (k=2) exactness = 48-param table over minimal Markov state
(top, empty, prev_was_close) + exact stack (L-MARKOV-COMPLETION). Generalize
to Dyck-3 (3 bracket types): minimal state = (top in {0,1,2}, empty, prevC)
= 8 rows x 8 vocab = 64 params. Same architecture, no redesign.
If the 64-param organ stays exact @4096 while the (strong) TF fails, the
organ PATTERN (exact stack + minimal Markov readout) is k-general.
Arms: table3_k3 (64p, 2 seeds), ssm_d16_1 (3.2k, 1 seed), TF strong
(d128/4L/10k). Task: same U-shape, k=3, train L=64, eval 64/4096.
USAGE: OMP_NUM_THREADS=1 python3 -u dyck3_echo.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
K = 3                          # bracket types
VOCAB = K + 1 + K + 1          # opens(3) + C(1) + echoes(3) + Z(1) = 8
LNK = math.log(3.0)
CFG = dict(steps=2500, batch=32, train_len=64,
           eval_lens=[64, 4096], eval_reps=2, KSTACK=4096,
           p_rise=0.9, p_fall=0.02, tf_steps=10000)
print(f"[setup] dyck3-echo cfg={CFG}", flush=True)
t_start = time.time()

def gen_echo3(batch, length, rng):
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll, stack = [], [], []
        while len(x) < length + 1:
            pos = len(x)
            p = CFG["p_rise"] if pos < (length + 1) // 2 else CFG["p_fall"]
            d = len(stack)
            if d == 0 or rng.random() < p:
                t = rng.randrange(K); stack.append(t)
                x.append(t); nll.append(math.log(3.0) if d == 0 else math.log(6.0))
            else:
                stack.pop(); x.append(K); nll.append(math.log(2.0))
                if stack:
                    x.append(K + 1 + stack[-1]); nll.append(0.0)
                else:
                    x.append(2 * K + 1); nll.append(0.0)
        xs.append(x[:length]); ys.append(x[1:length + 1])
        os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

# ---------------------------------------------------------------- models
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

class SSMHost(nn.Module):
    def __init__(self):
        super().__init__()
        d = 16
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([SSMBlock(d) for _ in range(1)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

class StrongTF(nn.Module):
    def __init__(self, d_model=128, n_layers=4, n_heads=8):
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

def features3(x):
    """minimal Markov state: (top in 0..K-1, empty, prev_was_close)"""
    B, L = x.shape
    KS = CFG["KSTACK"]
    feats = torch.empty(B, L, 3, dtype=torch.long)
    for b in range(B):
        stack = []
        for t in range(L):
            tok = int(x[b, t])
            if tok < K:
                if len(stack) < KS:
                    stack.append(tok)
            elif tok == K:
                if stack:
                    stack.pop()
            empty = 1 if not stack else 0
            top = stack[-1] if stack else 0
            feats[b, t, 0] = top
            feats[b, t, 1] = empty
            feats[b, t, 2] = 1 if tok == K else 0
    return feats

class Table3K(nn.Module):
    """64-param organ: 8-row table over minimal Markov state (k=3)."""
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(0.1 * torch.randn(2 * K + 2, VOCAB))

    def forward(self, x):
        f = features3(x)
        combo = f[:, :, 0] + f[:, :, 1] * K + f[:, :, 2] * (K + 1)
        return self.table[combo]

# ---------------------------------------------------------------- training/eval
def train_model(model, steps, seed=0, tag=""):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y, o = gen_echo3(CFG["batch"], CFG["train_len"], rng)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 2000 == 0:
            print(f"  [{tag}] step {step}/{steps} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_full(model, L=4096, reps=2):
    model.eval()
    bs = max(1, min(4, 4096 // L))
    ce = orc = n = 0
    per = {t: [0.0, 0] for t in range(VOCAB)}
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gen_echo3(bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        for t in range(VOCAB):
            m = (y == t)
            per[t][0] += nll[m].sum().item(); per[t][1] += int(m.sum())
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
    echo_toks = [K + 1, K + 2, K + 3, 2 * K + 1]
    echo_n = sum(per[t][1] for t in echo_toks)
    return {"total": round((ce - orc) / n, 4),
            "echo": round(sum(per[t][0] for t in echo_toks) / max(1, echo_n), 4)}

# ---------------------------------------------------------------- experiment
RESULTS = {}
for seed in range(2):
    torch.manual_seed(seed)
    m = Table3K()
    if seed == 0:
        print(f"[arm] table3_k3 params={n_params(m)}", flush=True)
    train_model(m, CFG["steps"], seed, f"tbl3 s{seed}")
    r = eval_full(m, 4096, 2)
    r["params"] = 64
    RESULTS[f"table3_k3_s{seed}"] = r
    print(f"  [tbl3 s{seed}] @4096 {r}", flush=True)
    del m

torch.manual_seed(0)
m = SSMHost()
print(f"[arm] ssm_d16_1 params={n_params(m)}", flush=True)
train_model(m, CFG["steps"], 0, "ssm")
r = eval_full(m, 4096, 2)
r["params"] = n_params(m)
RESULTS["ssm_d16_1"] = r
print(f"  [ssm] @4096 {r}", flush=True)
del m

torch.manual_seed(0)
m = StrongTF()
print(f"[arm] TF_STRONG params={n_params(m)} (10k steps)", flush=True)
train_model(m, CFG["tf_steps"], 0, "tf strong")
r = eval_full(m, 4096, 2)
r["params"] = n_params(m)
RESULTS["TF_STRONG"] = r
print(f"  [tf] @4096 {r}", flush=True)

print("\n" + "=" * 72)
print("DYCK-3-ECHO @4096 (total dCE | echo-dCE)")
print("=" * 72)
for k, v in RESULTS.items():
    print(f"{k:<16} params {v['params']:<7} total {v['total']:<9} echo {v['echo']}", flush=True)
print("=" * 72)
final = {"tag": "ARC2-C11-DYCK3-ECHO", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
