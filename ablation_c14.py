"""
C14 ABLATION: why did the isolated echo branch (iso_s0_10k: 1.14 @4096)
miss the standalone echo-organ cert (-0.2935)? Controlled tests, each on
PURE echo data (batch 32, L=64, 2500 steps, seed 0) with the SAME
components as IsoModel branch 0, varying ONE factor at a time:
  A: 8-row (top,empty,prevC) table          <- iso branch 0 as built
  B: 4-row (top,empty) table                <- standalone dyck_echo.py
  C: 8-row table but L=63 training          <- iso training length
  D: 8-row table + router CE on first-3-token emb (batch 8 duty)
Prints total dCE @4096 (cert convention: standalone = -0.2935).
"""
import json, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB = 45
LN6 = math.log(6.0)
g = {"__name__": "abl"}
exec(open("unified_iso.py").read().split("\nRESULTS = {}")[0], g)
SSMBlock, n_params, gen_echo_t = g["SSMBlock"], g["n_params"], g["gen_echo_t"]

class Branch0(nn.Module):
    def __init__(self, rows8, vocab=VOCAB):
        super().__init__()
        d = 16
        self.emb = nn.Embedding(vocab, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.host = SSMBlock(d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.table = nn.Parameter(0.1 * torch.randn(4 if not rows8 else 8, vocab))

    def features(self, x):
        B, L = x.shape
        feats = torch.empty(B, L, 3, dtype=torch.long)
        for b in range(B):
            stack = []
            for t in range(L):
                tok = int(x[b, t])
                if tok in (0, 1):
                    if len(stack) < 4096:
                        stack.append(tok)
                elif tok == 2:
                    if stack:
                        stack.pop()
                feats[b, t, 0] = stack[-1] if stack else 0
                feats[b, t, 1] = 1 if not stack else 0
                feats[b, t, 2] = 1 if tok == 2 else 0
        return feats

    def forward(self, x):
        f = self.features(x)
        if self.table.shape[0] == 8:
            combo = f[:, :, 0] + f[:, :, 1] * 2 + f[:, :, 2] * 4
        else:
            combo = f[:, :, 0] + f[:, :, 1] * 2
        h = self.norm(self.host(self.emb(x)))
        return self.table[combo] + self.head(h)

@torch.no_grad()
def eval4096(model, reps=2):
    model.eval()
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + 4096 + i)
        x, y, o = gen_echo_t(1, 4096, rng)
        nll = -F.log_softmax(model(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
    return round((ce - orc) / n, 4)

def run(tag, rows8, train_len=64, batch=32, router_ce=False, steps=2500):
    torch.manual_seed(0)
    m = Branch0(rows8)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    rng = random.Random(17)
    t0 = time.time()
    for step in range(steps):
        x, y, o = gen_echo_t(batch, train_len, rng)
        logits = m(x)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
        if router_ce:
            # mimic the iso router gradient: a 3-way classifier on first-3 emb
            task = torch.zeros(batch, dtype=torch.long)  # all echo
            rl = torch.randn(batch, 3, requires_grad=False)
            emb3 = m.emb(x[:, :3])
            rl2 = F.linear(emb3.reshape(batch, -1),
                           torch.randn(16, 48))  # proxy: extra emb gradient path
            loss = loss + 0.5 * F.cross_entropy(rl2, task)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad()
    d = eval4096(m)
    print(f"  {tag:<34} dCE@4096 = {d}   ({time.time()-t0:.0f}s)", flush=True)
    return d

print("ABLATION C14 — pure echo, batch 32, L=64, 2500 steps, dCE@4096 "
      "(standalone cert = -0.2935; iso_s0_10k = 1.1401):", flush=True)
run("A: 8-row table (iso branch0)", rows8=True)
run("B: 4-row table (standalone)", rows8=False)
run("C: 8-row table, L=63", rows8=True, train_len=63)
run("D: 8-row table + emb router-proxy CE", rows8=True, router_ce=True)
print("DONE", flush=True)
