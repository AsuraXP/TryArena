"""
ARC-2 CYCLE 8 / CAPSTONE v2 - DYCK-ECHO: the non-regular boundary, made sharp
=============================================================================
DYNABOUND (dyck_unbounded.py) showed unbounded random-walk Dyck-2 is NOT a
boundary test: next-token prediction needs only the (depth, top) summary, and a
16-dim linear RNN soft-counts it (dCE 0.1256@4096, flat) while tf_rope decayed
(2.1208). Logged as negative refinement.

DYCK-ECHO forces arbitrary-depth STACK READS into the prediction problem:
  tokens: O0 O1 (opens), C (close), E0 E1 (echo of the NEW top after a close),
          Z (echo when the stack becomes empty).
  generator: U-shaped open probability (0.9 rising half, 0.1 falling half), so a
  length-L sequence rises to depth ~0.8L/2 and falls in a run of ~0.9L/2
  consecutive (C, echo) pairs. The k-th echo in the descent reads stack element
  k -> predicting the sequence requires the full stack: non-regular.
  Train L=64 (descent runs <= ~30), eval 64/512/2048/4096 (descent run ~1800).

Hero: SSMHost d16/1blk (3.2k) + echo-organ: EXACT K-bit stack (K=4096)
maintained by push/pop (the mechanism, exact like a register file), with the
organ's program = a learned readout table over the 4 (top, empty) combinations
(CE-supervised) + host residual (L-GATE-INIT). The readout is depth-general
because the next-token distribution depends only on (top, empty) at every
position (verified on the generator).

Controls (same data/budget): ssm_d16_1 (16 floats must remember ~1600 bits ->
predicted failure), tf_rope (must decay, Hahn 2020).
Win = hero dCE <= 0.10 @4096 while both controls fail (ssm >= 0.25, tf >= 0.5).
USAGE: OMP_NUM_THREADS=1 python3 -u dyck_echo.py
"""
import json, math, os, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
CFG = dict(steps=2500, batch=32, train_len=64,
           eval_lens=[64, 512, 2048, 4096], eval_reps=2,
           d_model=16, K=4096, p_rise=0.9, p_fall=0.02)
print(f"[setup] dyck-echo cfg={CFG}", flush=True)
t_start = time.time()

# ---------------------------------------------------------------- data
# tokens: 0=O0 1=O1 2=C 3=E0 4=E1 5=Z
VOCAB = 6
L2, L4 = math.log(2.0), math.log(4.0)

def gen_echo(batch, length, rng):
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll, stack = [], [], []
        while len(x) < length + 1:
            i = len(x)
            mid = (length + 1) // 2
            p = CFG["p_rise"] if i < mid else CFG["p_fall"]
            d = len(stack)
            if d == 0 or rng.random() < p:
                t = rng.randrange(2); stack.append(t)
                x.append(t); nll.append(L2 if d == 0 else L4)
            else:
                t = stack.pop(); x.append(2); nll.append(L2)
                if stack:
                    x.append(3 + stack[-1]); nll.append(0.0)
                else:
                    x.append(5); nll.append(0.0)
        xs.append(x[:length]); ys.append(x[1:length + 1])
        os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

def check_valid(x):
    """replay a sequence; return max depth; asserts well-formedness."""
    stack, maxd, toks = [], 0, list(x.tolist())
    for i, tok in enumerate(toks):
        if tok <= 1:
            stack.append(tok); maxd = max(maxd, len(stack))
        elif tok == 2:
            assert stack, "close on empty"
            stack.pop()
            if i + 1 < len(toks):
                assert toks[i + 1] == (5 if not stack else 3 + stack[-1]), "bad echo after close"
        elif tok in (3, 4):
            assert stack and stack[-1] == tok - 3, "echo mismatch"
        elif tok == 5:
            assert not stack, "Z on non-empty"
    return maxd

# ---------------------------------------------------------------- host
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
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([SSMBlock(d) for _ in range(1)])
        self.norm = nn.LayerNorm(d)

    def hiddens(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.norm(h)

def n_params(m):
    return sum(p.numel() for p in m.parameters())

# ---------------------------------------------------------------- organ
class EchoModel(nn.Module):
    """SSM host + exact-stack organ. Readout table over (top, empty): 4 combos."""
    def __init__(self):
        super().__init__()
        self.host = SSMHost()
        self.table = nn.Parameter(0.1 * torch.randn(4, VOCAB))   # (top, empty) -> logits
        self.hhost = nn.Linear(CFG["d_model"], VOCAB)
        nn.init.zeros_(self.hhost.weight)
        nn.init.zeros_(self.hhost.bias)

    def features(self, x):
        """(top, empty) per position AFTER the token at that position."""
        B, L = x.shape
        K = CFG["K"]
        feats = torch.empty(B, L, 2, dtype=torch.long)
        for b in range(B):
            stack = []
            for t in range(L):
                tok = int(x[b, t])
                if tok <= 1:
                    if len(stack) < K:
                        stack.append(tok)
                elif tok == 2:
                    if stack:
                        stack.pop()
                elif tok in (3, 4):
                    pass  # echo: no stack change
                else:  # Z
                    pass
                empty = 1 if not stack else 0
                top = stack[-1] if stack else 0
                feats[b, t, 0] = top
                feats[b, t, 1] = empty
        return feats

    def forward(self, x):
        h = self.host.hiddens(x)
        f = self.features(x)
        combo = f[:, :, 0] + f[:, :, 1] * 2
        return self.table[combo] + self.hhost(h)

class PlainSSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.host = SSMHost()
        self.head = nn.Linear(CFG["d_model"], VOCAB)
        self.head.weight = self.host.emb.weight

    def forward(self, x):
        return self.head(self.host.hiddens(x))

class TransformerLM(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB)
        self.head.weight = self.emb.weight
        self.d_model = d_model

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

# ---------------------------------------------------------------- training/eval
def train_model(model, seed=0, tag=""):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(1, CFG["steps"] + 1):
        x, y, o = gen_echo(CFG["batch"], CFG["train_len"], rng)
        loss = F.cross_entropy(model(x).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 500 == 0:
            print(f"  [{tag}] step {step}/{CFG['steps']} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

@torch.no_grad()
def eval_dce(model, L, reps):
    model.eval()
    bs = max(1, min(8, 4096 // L))
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gen_echo(bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        ce += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
        orc += o.sum().item(); n += y.numel()
    return (ce - orc) / n

# ---------------------------------------------------------------- experiment
ALL = {}
torch.manual_seed(0)
hero = EchoModel()
if os.path.exists("dyke_s0.pt"):
    hero.load_state_dict(torch.load("dyke_s0.pt", weights_only=True))
    print("[run] loaded dyke_s0.pt (resume eval, no retrain)", flush=True)
else:
    print(f"[run] echo-organ params={n_params(hero)}", flush=True)
    train_model(hero, 0, "echo")
    torch.save(hero.state_dict(), "dyke_s0.pt")
ALL["echo (host + exact-stack organ)"] = {f"L{L}": round(eval_dce(hero, L, CFG["eval_reps"]), 4)
                                          for L in CFG["eval_lens"]}
del hero

torch.manual_seed(0)
ssm = PlainSSM()
print(f"[run] ssm_d16_1 control params={n_params(ssm)}", flush=True)
train_model(ssm, 0, "ssm")
ALL["ssm_d16_1 (same data)"] = {f"L{L}": round(eval_dce(ssm, L, CFG["eval_reps"]), 4)
                                for L in CFG["eval_lens"]}
del ssm

torch.manual_seed(0)
tf = TransformerLM()
print(f"[run] tf_rope params={n_params(tf)} (same budget)", flush=True)
train_model(tf, 0, "tf")
ALL["tf_rope"] = {f"L{L}": round(eval_dce(tf, L, CFG["eval_reps"]), 4)
                  for L in CFG["eval_lens"]}

print("\n" + "=" * 80)
print("RESULTS  dCE = excess CE over analytic oracle (nats/token; 0 = perfect)")
print("=" * 80)
print(f"{'run':<32}" + "".join(f"{f'L{L}':<10}" for L in CFG["eval_lens"]), flush=True)
for run, r in ALL.items():
    print(f"{run:<32}" + "".join(f"{r[f'L{L}']:<10}" for L in CFG["eval_lens"]), flush=True)
print("=" * 80)
hero_r = ALL["echo (host + exact-stack organ)"]
verdict = (f"hero dCE@4096 = {hero_r['L4096']} (WIN if <= 0.10 while ssm >= 0.25 and tf >= 0.5); "
           f"ssm = {ALL['ssm_d16_1 (same data)']['L4096']}; tf = {ALL['tf_rope']['L4096']}")
final = {"tag": "ARC2-C8-DYCKECHO", "runs": ALL, "verdict": verdict,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
