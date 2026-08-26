#!/usr/bin/env python3
"""ARCH-VET (cycle 51, Phase 1-4) — VET-LM: native discrete Mealy
controller x value register x exact LIFO side channel, as a token-
prediction architecture, vs Mamba-micro (continuous selective SSM)
vs micro-Transformer, on a synthetic 4-task reasoning stream.

PHASE-1 PRIOR ART (searched 2026-08-26):
  - Mamba-3 (ICLR 2026, arXiv 2603.15569): the SSM line's own
    statement of the limitation — real non-negative-eigenvalue
    transitions collapse to TC0-class state tracking (parity /
    permutation composition; Merrill et al. 2025, Grazzi et al.
    2025, Sarrof et al. 2024); single-layer Mamba not a universal
    approximator (Yu & Erichson 2025); copying failures (Jelassi
    et al. 2024); SSMs underperform attention on copying / ICL /
    long-context reasoning (survey arXiv 2408.01129 §7.5).
    Mamba-3's fixes (complex state, block-bias, MIMO) remain
    CONTINUOUS-selective.
  - Finite-state controller line (arXiv 2602.08734; ETH HRNN-LM
    equivalence; OpenReview S1gOpsCctm QBN/MMN insertion): all
    post-hoc extraction/quantization of continuous RNNs. Nothing
    implements the native learned k-state Mealy controller x value
    register x exact LIFO with state x query bilinear readout as an
    LM architecture.
HYPOTHESIS (H1): discrete control REGIMES (k=5) selecting per-token
read/write/decay action rows, with the d-dim register as the content
memory (L-VALUE-CHANNEL: the joint flag x value budget lives in the
value channel, not the state) plus an exact top-K LIFO side channel
read out by a controller-state x query bilinear (L-QUERY-READOUT),
beats continuous-selective SSM and micro-attention at equal params
on algorithmic reasoning: symbol tracking (long gaps), mod-3
counting (n beyond train), Dyck depth beyond train, key-value
retrieval (long gaps) — especially at length generalization.

ARCHITECTURES (V=48, d=16, single block + head, ~8k params each):
  VET : E(V,d); controller s_t = softmax(W_s x_t + W_ss s_{t-1} + b)
        (k=5, soft one-hot); register R_t = a(s_t) R_{t-1} +
        (sum_s s_t Ww[s] x_t), a(s) = per-state decay rows
        (exp(-softplus(Alog[s]))); hard push gate
        g = sigmoid(Wg [s;x]) (STE) -> exact top-4 buffer of pushed
        token embeddings; output = W_out(R + x) + T[s, j] additive
        stack table (0.1-randn) + M[s, :, :] x query bilinear
        (ZERO-init, L-GATE-INIT + L-QUERY-READOUT).
  MAMBA: depth-2 selective S4-style, d_state=48:
        h_t = exp(-softplus(A)-softplus(Wd x_t)) h + (WB x_t) x_t;
        y = Wy (h (WC x_t)) content-read (Mamba-1 fix for
        content-based reasoning); head.
  TF  : 2-layer causal, d16, 2 heads, pre-LN, MLP 4x, head.
DATA: 4-task auto-regressive stream, L=256 train; VAL = same
families at HELD-OUT long intervals (gaps 32-64, n<=30, depth
3-4, gaps 24-48); probes: task accuracy at train-interval vs
eval-interval + CE @256/512/1024 (length invariance).
Protocol: AdamW 3e-3, batch 8, 2000 steps/arm on a SHARED 512-stream
train pool (fresh minibatch/step; no single-batch memorization), seed 0,
1 thread. TF baseline carries sinusoidal PE (legit, not a strawman).
Tag ARCH-VET-LM-1.  Usage: OMP_NUM_THREADS=1 python3 -u arch_vet_lm.py
"""
import json
import math
import os
import random
import resource
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
PEAK = 0.0


def _peak():
    global PEAK
    try:
        mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        if mb > PEAK:
            PEAK = mb
    except Exception:
        pass


# ============================================================ vocab
V = 48
# 0 BOS, 1 EOS, 2 U (turn start), 3 A (answer marker)
BOS, EOS, U, A = 0, 1, 2, 3
T_TASK = 4                       # task start
MODS = 5 + 8                     # filler f0..f7 = 5..12
TRACK = 13 + 8                   # symbols x0..x7 = 13..20
ONE = 21                         # counter token
MANS = 22 + 3                    # mod answers m0..m2 = 22..24
BRK = 25 + 4                     # brackets (a,a),(b,b) = 25..28
KEYS = 29 + 4                    # pair keys k0..k3 = 29..32
VALS = 33 + 4                    # pair vals p0..p3 = 33..36
assert KEYS + 4 <= V


def fill_tok(rng):
    return rng.randrange(8) + MODS


# ============================================================ data
def gen_stream(rng, L=256, hard=False):
    """One 4-task mixed stream. hard = held-out long-interval."""
    # interval regimes: train vs eval (the generalization axis)
    if hard:
        track_gap_lo, track_gap_hi = 32, 64
        count_lo, count_hi = 13, 30
        dyck_depth = 3                     # train was <=2
        pair_gap_lo, pair_gap_hi = 24, 48
    else:
        track_gap_lo, track_gap_hi = 4, 16
        count_lo, count_hi = 2, 12
        dyck_depth = 2
        pair_gap_lo, pair_gap_hi = 4, 12
    def emit(d):
        if d == 0:
            return []
        t = rng.randrange(2)
        if rng.random() < 0.7:
            return [BRK + t] + emit(d - 1) + [BRK + 2 + t]
        return ([BRK + t] + emit(d - 1) + [BRK + 2 + t]
                + [BRK + 1 - t] + emit(d - 1) + [BRK + 3 - t])

    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        cand = None
        for _attempt in range(6):         # task must fit the remaining room
            task = rng.randrange(4)
            if task == 0:                 # TRACK: T x <gap fills> A x
                sym = rng.randrange(8) + TRACK
                gap = min(rng.randrange(track_gap_lo,
                                        track_gap_hi + 1), max(0, room - 4))
                cand = ([T_TASK, sym]
                        + [fill_tok(rng) for _ in range(gap)]
                        + [A, sym])
            elif task == 1:               # MODK: T T 1*1 ... A m
                n = min(rng.randrange(count_lo, count_hi + 1),
                        max(0, room - 4))
                if n < max(1, count_lo):
                    continue
                cand = ([T_TASK, T_TASK] + [ONE] * n
                        + [A, MANS + (n % 3)])
            elif task == 2:               # DYCK: balanced brackets
                for _dd in range(6):
                    d = min(dyck_depth, max(1, dyck_depth - _dd))
                    seg = emit(d)
                    if 3 + len(seg) <= room:
                        cand = [T_TASK, T_TASK, T_TASK] + seg
                        break
            else:                         # PAIR: T T k v <gap> A k v
                i = rng.randrange(4)
                j = rng.randrange(4)
                gap = min(rng.randrange(pair_gap_lo, pair_gap_hi + 1),
                          max(0, room - 8))
                cand = ([T_TASK, T_TASK, KEYS + i, VALS + j]
                        + [fill_tok(rng) for _ in range(gap)]
                        + [A, KEYS + i, VALS + j])
            if cand is not None and len(cand) <= room:
                break
            cand = None
        if cand is None:                  # filler pad (rare)
            x += [fill_tok(rng)] * room
            break
        x += cand
    x.append(EOS)
    assert len(x) == L + 1, len(x)
    return x, {}


def make_batches(n, L, rng, hard=False):
    xs = []
    for _ in range(n):
        x, _ = gen_stream(rng, L, hard)
        xs.append(torch.tensor(x[:L + 1]))
    return torch.stack(xs)


# ============================================================ models
class VETLM(nn.Module):
    """Discrete Mealy controller (k) x value register (d) x exact
    top-K LIFO (STE push) + state x query bilinear stack readout.

    Per token:  s_t = softmax(Ws x_t + Wss s_{t-1})        (k-state)
                a_t = sum_k s_t,k exp(-softplus(Alog[k]))  (regime decay)
                R_t = a_t R_{t-1} + sum_k s_t,k Ww[k] x_t  (register)
                push gate g_t (hard, STE) -> exact top-K buffer
                logits = head(Wo(R + x)) + s x feat bilinear (M,
                          zero-init) + sum_j sel_j s T[s, j]
    """

    def __init__(self, V, d, k=5, K=4):
        super().__init__()
        self.d, self.k, self.K = d, k, K
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, std=0.02)
        self.Ws = nn.Linear(d, k)
        self.Wss = nn.Linear(k, k, bias=False)
        self.Alog = nn.Parameter(torch.full((k, d), -3.0))
        self.Ww = nn.Parameter(0.1 * torch.randn(k, d, d))
        self.Wg = nn.Linear(d + k, 1)
        nn.init.constant_(self.Wg.bias, -1.0)   # sparse init: no push
        self.Wo = nn.Linear(d, d, bias=False)
        self.T = nn.Parameter(0.1 * torch.randn(k, K + 1, V))
        self.M = nn.Parameter(torch.zeros(k, d, V))      # zero-init
        self.head = nn.Linear(d, V)

    def forward(self, x):
        B, L = x.shape
        e = self.E(x)                                    # (B, L, d)
        R = torch.zeros(B, self.d, device=x.device)
        s = torch.full((B, self.k), 1.0 / self.k,
                       device=x.device)
        buf = torch.zeros(B, self.K, self.d, device=x.device)
        valid = torch.zeros(B, self.K, dtype=torch.bool,
                            device=x.device)
        lg = torch.empty(B, L, V, device=x.device)
        for t in range(L):
            xt = e[:, t]
            s = F.softmax(self.Ws(xt) + self.Wss(s), -1)
            a = (s.unsqueeze(-1)
                 * torch.exp(-F.softplus(self.Alog))).sum(1)  # (B,d)
            w = torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            R = a * R + w
            g = torch.sigmoid(self.Wg(torch.cat([s, xt], -1)))
            push = (g > 0.5) + (g - g.detach())          # STE: hard fwd
            buf = torch.roll(buf, 1, dims=1)
            buf[:, 0] = xt * push                              # (B,d)
            valid = torch.roll(valid, 1, dims=1)
            valid[:, 0] = (g > 0.5).squeeze(-1)
            y = self.Wo(R + xt)
            feat = torch.stack(
                [buf[:, j] for j in range(self.K)] + [xt], 1)
            logits = self.head(y)
            logits = logits + torch.einsum(
                "bk,bjd,kdv->bv", s, feat, self.M)       # bilinear
            # exact top-of-stack: slot j selected iff valid and no
            # newer slot (lower index) is valid (LIFO)
            sel = torch.zeros(B, self.K + 1, device=x.device)
            for j in range(self.K):
                newer = torch.zeros(B, device=x.device) + sum(
                    valid[:, i].float() for i in range(j))
                sel[:, j] = valid[:, j].float() * (newer == 0).float()
            sel[:, self.K] = 1.0
            logits = logits + torch.einsum("bs,ksv->bv", sel, self.T)
            lg[:, t] = logits
        return lg


class _MambaBlock(nn.Module):
    """Selective S4 step: h <- exp(-softplus(A)-softplus(Wd x)) h +
    (WB x) x ;  y = Wy (h * (WC x))  (content read, Mamba-1 fix)."""

    def __init__(self, d, dd):
        super().__init__()
        self.Win = nn.Linear(d, dd, bias=False)
        self.Wd = nn.Linear(d, dd, bias=False)
        self.WB = nn.Linear(d, dd, bias=False)
        self.WC = nn.Linear(d, dd, bias=False)
        self.Wy = nn.Linear(dd, d, bias=False)
        self.Alog = nn.Parameter(torch.full((dd,), -3.0))

    def forward(self, h, xt):
        xi = self.Win(xt)
        a = torch.exp(-F.softplus(self.Alog)
                      - F.softplus(self.Wd(xt)))
        h = a * h + self.WB(xt) * xi
        return h, self.Wy(h * self.WC(xt))


class MambaMicro(nn.Module):
    """Depth-2 continuous selective SSM (Mamba-1-style), d_state=48,
    content read — the baseline's strongest known configuration for
    content-based reasoning."""

    def __init__(self, V, d, dd=48):
        super().__init__()
        self.d = d
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, std=0.02)
        self.blocks = nn.ModuleList([_MambaBlock(d, dd)
                                     for _ in range(2)])
        self.head = nn.Linear(d, V)

    def forward(self, x):
        B, L = x.shape
        e = self.E(x)
        dd = self.blocks[0].Alog.numel()
        h = torch.zeros(B, dd, device=x.device)
        out = torch.empty(B, L, self.d, device=x.device)
        for t in range(L):
            xt = e[:, t]
            y = torch.zeros_like(xt)
            for b in self.blocks:
                h, yb = b(h, xt)
                y = y + yb
            out[:, t] = y
        return self.head(out)


def sinusoidal_pe(L, d):
    pos = torch.arange(L).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float()
                    * (-math.log(10000.0) / d))
    pe = torch.zeros(L, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class TFMicro(nn.Module):
    """2-layer causal Transformer, d16, 2 heads, pre-LN, sinusoidal PE
    (extrapolates to L>train — a legitimate, non-strawman baseline)."""

    def __init__(self, V, d, nh=2, depth=2, mlp=4, max_len=2048):
        super().__init__()
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nh, d * mlp,
                                       batch_first=True,
                                       norm_first=True,
                                       dropout=0.0)
            for _ in range(depth)])
        self.head = nn.Linear(d, V)
        self.d = d
        self.register_buffer("pe", sinusoidal_pe(max_len, d))

    def forward(self, x):
        B, L = x.shape
        e = self.E(x) + self.pe[:L].unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(
            L, device=x.device)
        h = e
        for b in self.blocks:
            h = b(h, src_mask=mask, is_causal=True)
        return self.head(h)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ============================================================ eval
@torch.no_grad()
def val_ce(model, xs, L):
    model.eval()
    B = xs.shape[0]
    lg = model(xs[:, :L + 1])
    nll = -F.log_softmax(lg, -1).gather(-1,
                                        xs[:, 1:L + 1].unsqueeze(-1)
                                        ).squeeze(-1)
    return round(float(nll.mean()), 4)


@torch.no_grad()
def task_acc(model, n, L, rng, hard):
    """Per-family exact-match accuracy at the answer positions
    (teacher-forced argmax; gold tokens read from the stream).
    track: symbol after A; modk: m after A; dyck: full balanced
    segment; pair: key AND value after A."""
    model.eval()
    fam_ok = {"track": [0, 0], "modk": [0, 0], "dyck": [0, 0],
              "pair": [0, 0]}
    for _ in range(n):
        x, _ = gen_stream(rng, L, hard)
        x2 = x[:L + 1]
        lg = model(torch.tensor(x2).unsqueeze(0))[:, :L, :]
        pred = lg.argmax(-1).squeeze(0)
        i = 0
        while i < len(x2) - 4:
            if x2[i] != T_TASK:
                i += 1
                continue
            nxt1, nxt2 = x2[i + 1], (x2[i + 2] if i + 2 < len(x2) else -1)
            if TRACK <= nxt1 < TRACK + 8:           # track
                j = i
                while j < len(x2) - 1 and x2[j] != A:
                    j += 1
                gold = [j + 1]
            elif nxt1 == T_TASK and nxt2 == T_TASK:  # dyck
                i2, depth, seg = i + 3, 0, []
                while i2 < len(x2):
                    t2 = x2[i2]
                    if BRK <= t2 < BRK + 4:
                        depth += 1 if t2 < BRK + 2 else -1
                        seg.append(i2)
                        if depth == 0:
                            i2 += 1
                            break
                    i2 += 1
                gold = seg
            elif nxt1 == T_TASK and nxt2 == ONE:     # modk
                j = i
                while j < len(x2) - 1 and x2[j] != A:
                    j += 1
                gold = [j + 1]
            elif nxt1 == T_TASK:                     # pair
                j = i
                while j < len(x2) - 1 and x2[j] != A:
                    j += 1
                gold = [j + 1, j + 2]
            else:
                i += 1
                continue
            # family by shape
            if TRACK <= nxt1 < TRACK + 8:
                fam = "track"
            elif nxt1 == T_TASK and nxt2 == T_TASK:
                fam = "dyck"
            elif nxt1 == T_TASK and nxt2 == ONE:
                fam = "modk"
            else:
                fam = "pair"
            # pred[t] guesses token t+1 -> answer at g checked via pred[g-1]
            ok_ = int(bool(gold) and max(gold) <= L
                      and all(int(pred[g - 1]) == x2[g] for g in gold))
            fam_ok[fam][0] += ok_
            fam_ok[fam][1] += 1
            i = max(gold) + 1 if gold else i + 3
    return {f: (fam_ok[f][0] / fam_ok[f][1]
                if fam_ok[f][1] else float("nan"))
            for f in fam_ok}


# ============================================================ train
def make_pool(n, L, seed, hard=False):
    prng = random.Random(seed)
    return [torch.tensor(gen_stream(prng, L, hard)[0]) for _ in range(n)]


def train_arm(name, model, pool, steps=2000, batch=8, lr=3e-3):
    """Train on a shared diverse pool (fresh minibatch each step)."""
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    hist = []
    n_pool = len(pool)
    for step in range(1, steps + 1):
        sel = [(step * batch + i) % n_pool for i in range(batch)]
        x = torch.stack([pool[i] for i in sel])
        y = x[:, 1:]
        lg = model(x[:, :256])
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            hist.append((step, round(float(loss), 4)))
            _peak()
            print(f"  [{name}] step {step}/{steps} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return hist


# ============================================================ main
def main():
    result = {"tag": "ARCH-VET-LM-1", "architectures": {}, "data":
              "4-task reasoning stream V=48 L=256: TRACK (symbol "
              "recall, gap 4-16 train / 32-64 eval), MODK (mod-3 "
              "count, n 2-12 / 13-30), DYCK (depth 2 / 3-4), PAIR "
              "(kv recall, gap 4-12 / 24-48); val = held-out long "
              "intervals; probes CE @256/512/1024 + per-task acc",
              "prior_art": ["Mamba-3 ICLR2026 arXiv 2603.15569 "
                            "(state-tracking TC0 collapse; "
                            "Merrill 2025; Grazzi 2025; Sarrof 2024; "
                            "Yu+Erichson 2025; Jelassi 2024; survey "
                            "2408.01129 7.5)",
                            "FSC line post-hoc only: arXiv "
                            "2602.08734, ETH HRNN-LM equivalence, "
                            "OpenReview S1gOpsCctm (QBN/MMN)"]}
    arms = {}
    pool = make_pool(512, 256, 12345)   # shared diverse train pool
    print(f"[data] train pool {len(pool)} x 256 generated", flush=True)
    for name, cls in (("VET", VETLM), ("MAMBA", MambaMicro),
                      ("TF", TFMicro)):
        if cls is VETLM:
            m = cls(V, 16, k=5, K=4)
        elif cls is MambaMicro:
            m = cls(V, 16)
        else:
            m = cls(V, 16)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_arm(name, m, pool, 2000, 8)
        rng = random.Random(999)
        vtr = make_batches(8, 256, rng)
        rh = random.Random(777)
        vha = make_batches(8, 256, rh, hard=True)
        r512 = random.Random(31337)
        v512 = make_batches(2, 512, r512)
        r1024 = random.Random(31415)
        v1024 = make_batches(2, 1024, r1024)
        m.eval()
        ce = {"256_train": val_ce(m, vtr, 256),
              "256_hard": val_ce(m, vha, 256),
              "512": val_ce(m, v512, 512),
              "1024": val_ce(m, v1024, 1024)}
        acc_tr = task_acc(m, 24, 256, random.Random(555), hard=False)
        acc_ev = task_acc(m, 24, 256, random.Random(666), hard=True)
        arms[name] = {"params": n_params(m), "loss_curve": hist,
                      "ce": ce, "acc_train_interval": acc_tr,
                      "acc_eval_interval": acc_ev}
        print(f"[{name}] ce={ce}", flush=True)
        print(f"[{name}] acc(train)={acc_tr}", flush=True)
        print(f"[{name}] acc(eval)={acc_ev}", flush=True)
        _peak()
    result["architectures"] = arms
    # Phase-4 verdict: VET vs the best of (MAMBA, TF) on the eval
    # interval accuracy (the reasoning axis) and length invariance
    def axis(a):
        return {f: a["acc_eval_interval"][f] for f in a["acc_eval_interval"]}
    vet, mam, tf = arms["VET"], arms["MAMBA"], arms["TF"]
    wins = {f: (axis(vet)[f] >= axis(mam)[f],
                axis(vet)[f] >= axis(tf)[f]) for f in
            ("track", "modk", "dyck", "pair")}
    ratio = {n: round(arms[n]["ce"]["1024"] /
                      max(1e-9, arms[n]["ce"]["256_hard"]), 3)
             for n in arms}
    result["phase4"] = {
        "acc_eval": {n: axis(arms[n]) for n in arms},
        "len_ratio_1024_over_256hard": ratio,
        "vet_wins_vs": {f: {"mamba": bool(wins[f][0]),
                            "tf": bool(wins[f][1])} for f in wins}}
    result["wall_s"] = round(time.time() - T0, 1)
    result["peak_mb"] = round(PEAK, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
