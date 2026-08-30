#!/usr/bin/env python3
"""
C55d COPYBIND — pointer/copy (Vinyals Pointer-Net 2015; See et al. 2017 pointer-gen).
Mutation vs C55c: answer is a POSITION copy, not a class over V. OOD entities
unseen in train must still be copyable. TF-vocab cannot; TF+pointer and GRU+pointer can.
Hypothesis: copy inductive bias, not delta-memory, is the entity-OOD mechanism.
"""
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
V, PAD, BOS, SEP, QMARK, EQ = 64, 0, 1, 2, 3, 4
REL0 = 42

def batch(B, L, rng, ood=False):
    lo, hi = (26, 42) if ood else (10, 26)
    xs, ptr = [], []
    for _ in range(B):
        e = int(rng.integers(lo, hi)); f = int(rng.integers(lo, hi))
        seq = [BOS, e, REL0, f, SEP, QMARK, e, REL0, EQ]  # answer copies pos of f (=3)
        seq = (seq + [PAD] * L)[:L]
        xs.append(seq); ptr.append(3)  # index of value token
    return torch.tensor(xs), torch.tensor(ptr)


class CopyGRU(nn.Module):
    def __init__(self, V=V, d=24):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.q = nn.Linear(d, d, bias=False)

    def ptr_logits(self, x):
        h, _ = self.gru(self.emb(x))
        # query = state at EQ
        B, T, d = h.shape
        q = []
        for i in range(B):
            j = int((x[i] == EQ).nonzero()[0])
            q.append(h[i, j])
        q = torch.stack(q)
        return torch.einsum("bd,btd->bt", self.q(q), h)

    def forward(self, x):
        return self.ptr_logits(x)


class TFCopy(nn.Module):
    def __init__(self, V=V, d=16, max_len=32):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        lyr = nn.TransformerEncoderLayer(d, 4, 32, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lyr, 2)
        self.q = nn.Linear(d, d, bias=False)

    def ptr_logits(self, x):
        h = self.emb(x) + self.pe[:x.size(1)]
        m = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.enc(h, mask=m, is_causal=True)
        B = x.size(0)
        q = []
        for i in range(B):
            j = int((x[i] == EQ).nonzero()[0])
            q.append(h[i, j])
        q = torch.stack(q)
        return torch.einsum("bd,btd->bt", self.q(q), h)

    def forward(self, x):
        return self.ptr_logits(x)


def acc(model, n, ood, seed):
    rng = np.random.default_rng(seed)
    model.eval(); ok = 0
    with torch.no_grad():
        x, ptr = batch(n, 16, rng, ood)
        logits = model(x)
        pred = logits.argmax(-1)
        # predicted position -> copied token vs gold f = x[:,3]
        for i in range(n):
            pos = int(pred[i])
            ok += int(x[i, pos] == x[i, 3])
    model.train(); return ok / n


def run(m, steps=400, seed=0):
    opt = torch.optim.Adam(m.parameters(), lr=4e-3)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    last = 0; t0 = time.time()
    for st in range(1, steps + 1):
        x, ptr = batch(8, 16, rng, False)
        loss = F.cross_entropy(m(x), ptr)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if st % 100 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f}", flush=True)
    return last, time.time() - t0

if __name__ == "__main__":
    rec = {"tag": "ARCH-C55D-COPYBIND"}
    for name, m in [("CopyGRU", CopyGRU()), ("TFCopy", TFCopy())]:
        p = sum(x.numel() for x in m.parameters())
        print(f"=== {name} p={p} ===", flush=True)
        loss, wall = run(m)
        rec[name] = {"params": p, "loss": loss, "wall": wall,
                     "id": acc(m, 80, False, 1), "ood": acc(m, 80, True, 2)}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl", "a").write(json.dumps(rec) + "\n")
    print("RESULT", rec, flush=True)
