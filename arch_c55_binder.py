#!/usr/bin/env python3
"""
C55 BINDER — Binding-Indexed Neural Delta with Explicit Relations.
Prior art (do not copy):
  Gated DeltaNet / DeltaNet-2 (Yang ICLR'25; arXiv:2412.06464, 2605.22791)
    — delta overwrite + decoupled erase/write; still rank-1 token mix, no typed slots.
  Titans LMM (Behrouz et al. 2025 arXiv:2501.00663) — surprise-gated test-time memory;
    reimpls show memory-alone weak if backbone frozen (arXiv:2510.09551).
  LM2 / RASA — slot memory still sits on Transformer attn (quadratic).
Mutation: O(N) token scan into S typed slots; per-slot gated-delta with INDEPENDENT
erase b and write w; surprise scales w; ONE inner Hopfield-style mix over S (not N)
for relational compose. Zero token self-attention.
"""
from __future__ import annotations
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

torch.set_num_threads(1)
V = 64
PAD, BOS, SEP, QMARK, EQ = 0, 1, 2, 3, 4
# entities 10-41, relations 42-47, digits 48-57
ENT0, NENT, REL0, NDIG = 10, 32, 42, 10

def nparams(m):
    return sum(p.numel() for p in m.parameters())


class BINDER(nn.Module):
    def __init__(self, V=V, d=16, S=8):
        super().__init__()
        self.d, self.S, self.V = d, S, V
        self.emb = nn.Embedding(V, d)
        self.Wk = nn.Linear(d, d, bias=False)
        self.Wv = nn.Linear(d, d, bias=False)
        self.Wb = nn.Linear(d, S)   # erase per slot
        self.Ww = nn.Linear(d, S)   # write per slot
        self.surprise = nn.Linear(d, 1)
        # slot-slot compose (relation mix), not token attn
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wkc = nn.Linear(d, d, bias=False)
        self.Wvc = nn.Linear(d, d, bias=False)
        self.ctrl = nn.GRUCell(d, d)
        self.out = nn.Linear(d + d, V)
        nn.init.normal_(self.emb.weight, 0, 0.04)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        B, T = x.shape
        d, S = self.d, self.S
        e = self.emb(x)
        slots = torch.zeros(B, S, d, device=x.device)
        h = torch.zeros(B, d, device=x.device)
        logits = []
        for t in range(T):
            et = e[:, t]
            k, v = self.Wk(et), self.Wv(et)
            b = torch.sigmoid(self.Wb(et))          # [B,S]
            w = torch.sigmoid(self.Ww(et))
            sur = torch.sigmoid(self.surprise(et - h))
            w = w * sur
            # address: cos-sim to slots
            sn = F.normalize(slots, dim=-1)
            kn = F.normalize(k, dim=-1)
            addr = F.softmax(8.0 * torch.einsum("bsd,bd->bs", sn, kn), dim=-1)
            # gated delta per slot: erase then write residual vs current read
            read = torch.einsum("bs,bsd->bd", addr, slots)
            delta = v - read
            slots = slots * (1.0 - b.unsqueeze(-1)) + (w * addr).unsqueeze(-1) * delta.unsqueeze(1)
            # O(S^2) relational compose
            q = self.Wq(slots)
            kc = self.Wkc(slots)
            vc = self.Wvc(slots)
            att = F.softmax(torch.einsum("bid,bjd->bij", q, kc) / math.sqrt(d), dim=-1)
            slots = slots + torch.einsum("bij,bjd->bid", att, vc)
            h = self.ctrl(et, h)
            pooled = slots.mean(1)
            logits.append(self.out(torch.cat([h, pooled], -1)))
        return torch.stack(logits, 1)


class TFMicro(nn.Module):
    def __init__(self, V=V, d=16, n_layer=2, n_head=4, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, n_head, dim_feedforward=32,
                                           batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.out = nn.Linear(d, V)

    def forward(self, x):
        h = self.emb(x) + self.pe[: x.size(1)]
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        return self.out(self.enc(h, mask=mask, is_causal=True))


def make_batch(task, B, L, rng, ood=False):
    """BIND: fact e R f ... Q e R -> f
    HOP2: e R1 m, m R2 f ... Q e R1 R2 -> f  (compose)
    ADD1: d1 + d2 = (d1+d2)%10  [carry ignored micro]
    """
    xs, ys = [], []
    ent_lo, ent_hi = (26, 42) if ood else (10, 26)  # OOD entities
    for _ in range(B):
        seq = [BOS]
        if task == "BIND":
            nfact = 1 if not ood else int(rng.integers(1, 3))
            facts = []
            for _i in range(nfact):
                e = int(rng.integers(ent_lo, ent_hi))
                r = REL0
                f = int(rng.integers(ent_lo, ent_hi))
                facts.append((e, r, f))
                seq += [e, r, f]
            e, r, f = facts[int(rng.integers(0, len(facts)))]
            seq += [SEP, QMARK, e, r, EQ, f]
        elif task == "HOP2":
            a = int(rng.integers(ent_lo, ent_hi))
            b = int(rng.integers(ent_lo, ent_hi))
            c = int(rng.integers(ent_lo, ent_hi))
            r1, r2 = REL0, REL0 + 1
            seq += [a, r1, b, b, r2, c, SEP, QMARK, a, r1, r2, EQ, c]
        elif task == "ADD1":
            d1 = int(rng.integers(0, 10))
            d2 = int(rng.integers(0, 10))
            seq += [48 + d1, 48 + d2, EQ, 48 + ((d1 + d2) % 10)]
        else:
            seq += [10, EQ, 10]
        seq = seq[:L] + [PAD] * (L - len(seq))
        tgt = seq[1:] + [PAD]
        xs.append(seq); ys.append(tgt)
    return torch.tensor(xs), torch.tensor(ys)


def answer_acc(model, task, n=64, ood=False, L=48, seed=0):
    rng = np.random.default_rng(seed)
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        x, y = make_batch(task, n, L, rng, ood=ood)
        pred = model(x).argmax(-1)
        for i in range(n):
            seq = x[i].tolist()
            if EQ not in seq:
                continue
            j = seq.index(EQ)  # predict token at EQ position -> next is answer in y[j]
            # model at index of EQ predicts answer (y aligned as shift)
            gold = y[i, seq.index(EQ)]
            pr = pred[i, seq.index(EQ)]
            ok += int(pr == gold)
            tot += 1
    model.train()
    return ok / max(tot, 1)


def answer_loss(logits, x, y):
    # supervise only the token after EQ (answer)
    B, T, C = logits.shape
    losses = []
    for i in range(B):
        seq = x[i].tolist()
        if EQ not in seq:
            continue
        j = seq.index(EQ)
        losses.append(F.cross_entropy(logits[i, j], y[i, j]))
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def train(model, tasks, steps, L, B, seed, lr=3e-3):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    last = 0.0
    t0 = time.time()
    for st in range(1, steps + 1):
        task = tasks[(st - 1) % len(tasks)]
        x, y = make_batch(task, B, L, rng, ood=False)
        logits = model(x)
        loss = answer_loss(logits, x, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.detach())
        if st % 200 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f} p={nparams(model)}", flush=True)
    return last, time.time() - t0


if __name__ == "__main__":
    tasks = ["BIND", "HOP2", "ADD1"]
    L, B, STEPS = 40, 8, 800
    arms = {
        "BINDER": BINDER(),
        "TFMicro": TFMicro(),
    }
    rec = {"tag": "ARCH-C55B-BINDER-ANSONLY"}
    for name, m in arms.items():
        print(f"=== {name} p={nparams(m)} ===", flush=True)
        loss, wall = train(m, tasks, STEPS, L, B, seed=0)
        acc_id = {t: answer_acc(m, t, n=80, ood=False, L=L, seed=1) for t in tasks}
        acc_ood = {t: answer_acc(m, t, n=80, ood=True, L=L, seed=2) for t in tasks}
        rec[name] = {"params": nparams(m), "loss": loss, "wall": wall,
                     "acc_id": acc_id, "acc_ood": acc_ood}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl", "a").write(json.dumps(rec) + "\n")
    print("RESULT", rec, flush=True)
