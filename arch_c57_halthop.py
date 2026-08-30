#!/usr/bin/env python3
"""
C57 HALTHOP — HopCopy + learned inner-hop halt (not Universal Transformer).
Prior: Graves ACT 1603.08983; UT 1807.03819 (still O(N^2) per step);
looped TF 2402.00976 / 2509.23314 / 2608.18171 — halt on TF blocks.
C56c: 2-hop distractor HopCopy 0.71 vs TF 0.50; 3-hop ZS ~0.
Mutation: INNER pointer hops with halt gate on query; TRAIN hops~U{1,2,3};
eval 3-hop ID/OOD and 4-hop ZS. No token self-attention.
"""
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
V = 80
PAD, BOS, SEP, QMARK, EQ = 0, 1, 2, 3, 4
RELS = [50, 51, 52, 53]

def make(B, L, rng, hops, ood=False):
    lo, hi = (40, 56) if ood else (10, 26)
    xs, ptrs = [], []
    for _ in range(B):
        seq = [BOS]
        chain = [int(rng.integers(lo, hi)) for _ in range(hops + 1)]
        rels = RELS[:hops]
        triples = []
        for i in range(hops):
            triples.append((chain[i], rels[i], chain[i + 1]))
        dx = int(rng.integers(lo, hi)); dy = int(rng.integers(lo, hi))
        triples.append((dx, rels[-1], dy))
        rng.shuffle(triples)
        val_pos = 3
        tgt, src, lr = chain[-1], chain[-2], rels[-1]
        for e, r, f in triples:
            seq += [e, r, f]
            if f == tgt and e == src and r == lr:
                val_pos = len(seq) - 1
        seq += [SEP, QMARK, chain[0]] + list(rels) + [EQ]
        seq = (seq + [PAD] * L)[:L]
        xs.append(seq); ptrs.append(val_pos)
    return torch.tensor(xs), torch.tensor(ptrs)


class HaltHop(nn.Module):
    def __init__(self, V=V, d=24, maxh=4):
        super().__init__()
        self.maxh = maxh
        self.emb = nn.Embedding(V, d)
        self.rnn = nn.GRU(d, d, batch_first=True)
        self.Wq = nn.Linear(d, d)
        self.mix = nn.Linear(2 * d, d)
        self.halt = nn.Linear(d, 1)

    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        B = x.size(0)
        qs = []
        for i in range(B):
            j = int((x[i] == EQ).nonzero()[0])
            qs.append(h[i, j])
        q = self.Wq(torch.stack(qs))
        acc_ptr = 0
        remain = torch.ones(B, 1, device=x.device)
        ponder = 0.0
        last = None
        for k in range(self.maxh):
            logits = torch.einsum("bd,btd->bt", q, h)
            last = logits
            p_halt = torch.sigmoid(self.halt(q))
            w_k = remain * p_halt if k < self.maxh - 1 else remain
            acc_ptr = acc_ptr + w_k * logits
            ponder = ponder + remain.mean()
            remain = remain * (1 - p_halt)
            rd = torch.einsum("bt,btd->bd", F.softmax(logits, -1), h)
            q = self.Wq(self.mix(torch.cat([q, rd], -1)))
        self._ponder = ponder
        return acc_ptr


class TFCopy(nn.Module):
    def __init__(self, V=V, d=24, max_len=80):
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
        hh = self.emb(x) + self.pe[: x.size(1)]
        m = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        hh = self.enc(hh, mask=m, is_causal=True)
        B = x.size(0)
        qs = []
        for i in range(B):
            j = int((x[i] == EQ).nonzero()[0])
            qs.append(hh[i, j])
        q = self.q(torch.stack(qs))
        return torch.einsum("bd,btd->bt", q, hh)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


def acc(model, n, hops, ood, seed, L=56):
    rng = np.random.default_rng(seed)
    model.eval(); ok = 0
    with torch.no_grad():
        x, ptr = make(n, L, rng, hops, ood)
        pred = model(x).argmax(-1)
        for i in range(n):
            ok += int(x[i, int(pred[i])].item() == x[i, int(ptr[i])].item())
    model.train()
    return ok / n


def train(m, steps=1500, seed=0, L=56, halt=False):
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    last = 0.0
    t0 = time.time()
    for st in range(1, steps + 1):
        hops = int(rng.integers(1, 4))  # 1..3
        x, ptr = make(8, L, rng, hops, False)
        logits = m(x)
        loss = F.cross_entropy(logits, ptr)
        if halt and hasattr(m, "_ponder"):
            loss = loss + 0.01 * m._ponder
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if st % 500 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f} p={nparams(m)}", flush=True)
    return last, time.time() - t0


if __name__ == "__main__":
    rec = {"tag": "ARCH-C57-HALTHOP"}
    arms = [("HaltHop", HaltHop(), True), ("TFCopy", TFCopy(), False)]
    for name, m, h in arms:
        print(f"=== {name} p={nparams(m)} ===", flush=True)
        loss, wall = train(m, halt=h)
        rec[name] = {
            "params": nparams(m), "loss": loss, "wall": wall,
            "h2_id": acc(m, 64, 2, False, 1),
            "h2_ood": acc(m, 64, 2, True, 2),
            "h3_id": acc(m, 64, 3, False, 3),
            "h3_ood": acc(m, 64, 3, True, 4),
            "h4_zs": acc(m, 64, 4, False, 5),
        }
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl", "a").write(json.dumps(rec) + "\n")
    print("RESULT", rec, flush=True)
