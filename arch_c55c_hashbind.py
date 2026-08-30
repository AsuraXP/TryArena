#!/usr/bin/env python3
"""
C55c HASHBIND — outer-product associative memory (Schmidhuber FWP; Ba et al. 2016;
linear attn Katharopoulos 2020). Cite: not Gated-DeltaNet. Single d×d fast weight.
Hypothesis: 1-fact BIND is just key-value write; if this cannot 1.0 in-dist the
eval/loss is broken. OOD entities test whether keys interpolate.
"""
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
V, PAD, BOS, SEP, QMARK, EQ = 64, 0, 1, 2, 3, 4
REL0, ENT0 = 42, 10

class HASHBIND(nn.Module):
    def __init__(self, V=V, d=24):
        super().__init__()
        self.d, self.V = d, V
        self.emb = nn.Embedding(V, d)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wq = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(d, 1)
        self.out = nn.Linear(d, V)
        nn.init.normal_(self.emb.weight, 0, 0.05)

    def forward(self, x):
        B, T = x.shape
        d = self.d
        e = self.emb(x)
        M = torch.zeros(B, d, d, device=x.device)
        last = torch.zeros(B, d, device=x.device)
        outs = []
        for t in range(T):
            et = e[:, t]
            k = torch.tanh(self.Wk(et))
            v = torch.tanh(self.Wv(et))
            g = torch.sigmoid(self.gate(et))
            M = M + g.unsqueeze(-1) * torch.einsum("bi,bj->bij", v, k)
            q = torch.tanh(self.Wq(et))
            read = torch.einsum("bij,bj->bi", M, q)
            last = 0.5 * last + 0.5 * read
            outs.append(self.out(last))
        return torch.stack(outs, 1)


class TFMicro(nn.Module):
    def __init__(self, V=V, d=16, n_layer=2, n_head=4, max_len=128):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        lyr = nn.TransformerEncoderLayer(d, n_head, 32, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lyr, n_layer)
        self.out = nn.Linear(d, V)

    def forward(self, x):
        h = self.emb(x) + self.pe[:x.size(1)]
        m = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.out(self.enc(h, mask=m, is_causal=True))


def batch(B, L, rng, ood=False):
    lo, hi = (26, 42) if ood else (10, 26)
    xs, ys = [], []
    for _ in range(B):
        e = int(rng.integers(lo, hi)); f = int(rng.integers(lo, hi))
        seq = [BOS, e, REL0, f, SEP, QMARK, e, REL0, EQ, f]
        seq = (seq + [PAD] * L)[:L]
        xs.append(seq); ys.append(seq[1:] + [PAD])
    return torch.tensor(xs), torch.tensor(ys)


def acc(model, n, ood, seed):
    rng = np.random.default_rng(seed)
    model.eval(); ok = 0
    with torch.no_grad():
        x, y = batch(n, 16, rng, ood)
        p = model(x).argmax(-1)
        for i in range(n):
            j = x[i].tolist().index(EQ)
            ok += int(p[i, j] == y[i, j])
    model.train(); return ok / n


def run(m, steps=600, seed=0):
    opt = torch.optim.Adam(m.parameters(), lr=4e-3)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    last = 0
    t0 = time.time()
    for st in range(1, steps + 1):
        x, y = batch(8, 16, rng, False)
        logits = m(x)
        loss = 0; c = 0
        for i in range(x.size(0)):
            j = x[i].tolist().index(EQ)
            loss = loss + F.cross_entropy(logits[i, j], y[i, j]); c += 1
        loss = loss / c
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if st % 150 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f}", flush=True)
    return last, time.time() - t0

if __name__ == "__main__":
    rec = {"tag": "ARCH-C55C-HASHBIND"}
    for name, m in [("HASHBIND", HASHBIND()), ("TFMicro", TFMicro())]:
        p = sum(x.numel() for x in m.parameters())
        print(f"=== {name} p={p} ===", flush=True)
        loss, wall = run(m)
        rec[name] = {"params": p, "loss": loss, "wall": wall,
                     "id": acc(m, 80, False, 1), "ood": acc(m, 80, True, 2)}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl", "a").write(json.dumps(rec) + "\n")
    print("RESULT", rec, flush=True)
