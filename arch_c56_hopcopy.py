#!/usr/bin/env python3
"""
C56 HOPCOPY — latent multi-hop pointer (not token CoT).
Prior: Pointer-Net 2015; C55e L-COPY-OOD-BIND (1-hop).
Two-hop LM failure: arXiv:2608.07261 — atomic facts stored, composition fails
especially 2nd-hop OOD. CRQs arXiv:2503.01544 — depth vs RNN tradeoff.
Mutation: after 1st content-read, RE-QUERY memory with the *read value vector*
(internal hop) then pointer-copy. No O(N^2) token attn. TFCopy = 2L causal TF pointer.
"""
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
V = 80
PAD, BOS, SEP, QMARK, EQ = 0, 1, 2, 3, 4
REL1, REL2, REL3 = 50, 51, 52

def make(B, L, rng, hops=2, ood=False, nf=1):
    """facts shuffled so last token is NOT the answer. Query a then rels; copy final value."""
    lo, hi = (40, 56) if ood else (10, 26)
    xs, ptrs = [], []
    for _ in range(B):
        seq = [BOS]
        chain = [int(rng.integers(lo, hi)) for _ in range(hops + 1)]
        rels = [REL1, REL2, REL3][:hops]
        triples = []
        for i in range(hops):
            triples.append((chain[i], rels[i], chain[i + 1]))
        # distractor: same last-rel, wrong value — blocks r2-lookup shortcut
        dx = int(rng.integers(lo, hi)); dy = int(rng.integers(lo, hi))
        triples.append((dx, rels[-1], dy))
        rng.shuffle(triples)
        val_pos = None
        target = chain[-1]
        for e, r, f in triples:
            seq += [e, r, f]
            if f == target and e == chain[-2] and r == rels[-1]:
                val_pos = len(seq) - 1
        seq += [SEP, QMARK, chain[0]] + list(rels) + [EQ]
        seq = (seq + [PAD] * L)[:L]
        if val_pos is None:
            val_pos = 3
        xs.append(seq); ptrs.append(val_pos)
    return torch.tensor(xs), torch.tensor(ptrs)


class HopCopy(nn.Module):
    def __init__(self, V=V, d=24, hops=2):
        super().__init__()
        self.hops = hops
        self.emb = nn.Embedding(V, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.Wq = nn.Linear(d, d)
        self.mix = nn.Linear(2 * d, d)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)
        B, T, d = h.shape
        # start query = hidden at EQ
        qs = []
        for i in range(B):
            j = int((x[i] == EQ).nonzero()[0])
            qs.append(h[i, j])
        q = self.Wq(torch.stack(qs))
        ptr = None
        for _hop in range(self.hops):
            logits = torch.einsum("bd,btd->bt", q, h)
            ptr = logits
            w = F.softmax(logits, dim=-1)
            read = torch.einsum("bt,btd->bd", w, h)
            q = self.Wq(self.mix(torch.cat([q, read], -1)))
        return ptr


class TFCopy(nn.Module):
    def __init__(self, V=V, d=24, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        lyr = nn.TransformerEncoderLayer(d, 4, 48, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(lyr, 2)
        self.q = nn.Linear(d, d)

    def forward(self, x):
        h = self.emb(x) + self.pe[: x.size(1)]
        m = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.enc(h, mask=m, is_causal=True)
        B = x.size(0)
        qs = []
        for i in range(B):
            j = int((x[i] == EQ).nonzero()[0])
            qs.append(h[i, j])
        q = self.q(torch.stack(qs))
        return torch.einsum("bd,btd->bt", q, h)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def acc(model, n, hops, ood, seed, L=40):
    rng = np.random.default_rng(seed)
    model.eval(); ok = 0
    with torch.no_grad():
        x, ptr = make(n, L, rng, hops=hops, ood=ood)
        pred = model(x).argmax(-1)
        for i in range(n):
            pos = int(pred[i])
            ok += int(x[i, pos].item() == x[i, int(ptr[i])].item())
    model.train()
    return ok / n


def train(m, hops=2, steps=900, seed=0, L=40):
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    last = 0.0
    t0 = time.time()
    for st in range(1, steps + 1):
        x, ptr = make(8, L, rng, hops=hops, ood=False)
        loss = F.cross_entropy(m(x), ptr)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if st % 300 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f} p={nparams(m)}", flush=True)
    return last, time.time() - t0


if __name__ == "__main__":
    rec = {"tag": "ARCH-C56C-HOPDIST"}
    for name, m in [("HopCopy", HopCopy(hops=2)), ("TFCopy", TFCopy())]:
        print(f"=== {name} p={nparams(m)} ===", flush=True)
        loss, wall = train(m, hops=2, steps=1200)
        rec[name] = {
            "params": nparams(m), "loss": loss, "wall": wall,
            "h2_id": acc(m, 80, 2, False, 1),
            "h2_ood": acc(m, 80, 2, True, 2),
            "h3_zs": acc(m, 80, 3, False, 3),  # extra hop zero-shot, in-ent
        }
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl", "a").write(json.dumps(rec) + "\n")
    print("RESULT", rec, flush=True)
