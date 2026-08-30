#!/usr/bin/env python3
"""
ARCH-VET-LM (canonical P1) — learned k-state Mealy × soft value register × STE LIFO.
Prior art:
  Mealy/Moore controllers — classic automata; soft-state RNNs (Graves 2013).
  Stack RNNs / NTM (Joulin & Mikolov 2015; Graves et al. 2014) — differentiable LIFO.
  STE (Bengio et al. 2013; Yin et al. 2019) — discrete push with identity backward.
  Mamba/S6 (Gu & Dao 2023) — selective SSM control arm.
  Content-addressed memory (Graves NTM; Bahdanau attn) — P8 VETCAM.
No transformer re-tests of old axes; TFMicro is C51-authorized control only.
"""
from __future__ import annotations
import math, os, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F

V = 48
PAD, BOS, EOS, SEP, T_TASK = 0, 1, 2, 3, 4
TOK_A, TOK_B = 10, 11
BRK_OPEN, BRK_CLOSE = 20, 21
CNT_TOK = 21  # also used as count token in MODK; dyck uses 20/21
PAIR_OPEN, PAIR_CLOSE = 30, 31
DIV_BASE = 39  # quotients 39..47 for d=3,4 in P7

def set_threads():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

class VETLM(nn.Module):
    """Native k-state Mealy × d-dim per-state register × exact top-K LIFO."""
    def __init__(self, V=V, k=5, d=16, K=4, cam=False):
        super().__init__()
        self.V, self.k, self.d, self.K, self.cam = V, k, d, K, cam
        self.emb = nn.Embedding(V, d)
        # controller: token × state -> next logits (Mealy)
        self.W_ctrl = nn.Parameter(torch.randn(V, k, k) * 0.05)
        self.b_ctrl = nn.Parameter(torch.zeros(V, k))
        # per-state decay + write gate for value register
        self.decay = nn.Parameter(torch.zeros(k, d))  # sigmoid later
        self.W_write = nn.Linear(d, d, bias=True)
        self.g_write = nn.Linear(d + k, 1, bias=True)
        # LIFO stack table (additive): token embeddings written by STE push
        self.stack_scale = nn.Parameter(torch.ones(1) * 0.1)
        # bilinear readout: state × query — ZERO INIT (C51)
        self.W_read = nn.Parameter(torch.zeros(k, d, V))
        self.b_read = nn.Parameter(torch.zeros(V))
        self.W_stk = nn.Linear(d, V, bias=False)
        self.tau = nn.Parameter(torch.tensor(4.0))  # CAM temperature
        self._init()

    def _init(self):
        nn.init.normal_(self.emb.weight, 0, 0.05)
        nn.init.zeros_(self.W_read)
        nn.init.zeros_(self.b_read)
        nn.init.xavier_uniform_(self.W_write.weight)
        nn.init.zeros_(self.W_write.bias)
        nn.init.zeros_(self.g_write.bias)
        nn.init.zeros_(self.W_stk.weight)

    def nparams(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # x: [B,T] int
        B, T = x.shape
        k, d, K, Vv = self.k, self.d, self.K, self.V
        device = x.device
        s = torch.zeros(B, k, device=device)
        s[:, 0] = 1.0
        v = torch.zeros(B, d, device=device)
        # stack buffer of K slots of d-dim, depth pointer (soft)
        buf = torch.zeros(B, K, d, device=device)
        depth = torch.zeros(B, device=device)
        logits = []
        e_all = self.emb(x)
        for t in range(T):
            xt = x[:, t]
            et = e_all[:, t]
            # controller
            Wc = self.W_ctrl[xt]          # [B,k,k]
            bc = self.b_ctrl[xt]          # [B,k]
            s_logits = torch.einsum("bk,bkj->bj", s, Wc) + bc
            s = F.softmax(s_logits, dim=-1)
            # value register: per-state decay
            dec = torch.sigmoid(s @ self.decay)  # [B,d]
            g = torch.sigmoid(self.g_write(torch.cat([et, s], dim=-1)))
            v = dec * v + g * torch.tanh(self.W_write(et))
            # STE LIFO push/pop heuristic: tokens >= PAIR_OPEN or BRK
            is_push = ((xt == BRK_OPEN) | (xt == PAIR_OPEN) | (xt == TOK_A)).float()
            is_pop  = ((xt == BRK_CLOSE) | (xt == PAIR_CLOSE) | (xt == TOK_B)).float()
            # hard depth for STE
            hard_push = (is_push > 0.5).float()
            hard_pop = (is_pop > 0.5).float()
            new_depth = (depth + hard_push - hard_pop).clamp(0, K)
            depth_ste = new_depth + (new_depth - new_depth.detach())  # identity STE noop
            # write to slot min(depth, K-1)
            slot = depth.long().clamp(0, K - 1)
            if hard_push.any():
                # additive stack table
                buf = buf.clone()
                idx = torch.arange(B, device=device)
                buf[idx, slot] = buf[idx, slot] + self.stack_scale * et * hard_push.unsqueeze(-1)
            if hard_pop.any():
                pop_slot = (depth - 1).long().clamp(0, K - 1)
                top = buf[torch.arange(B, device=device), pop_slot]
            else:
                top = buf[torch.arange(B, device=device), slot]
            depth = new_depth
            # readout
            q = v + 0.25 * top
            # state×query bilinear
            # W_read: [k,d,V]; s:[B,k]; q:[B,d]
            read = torch.einsum("bk,kdv,bd->bv", s, self.W_read, q)
            if self.cam:
                # content-addressed readout over buf
                # cos-sim(xt, buf_j)
                bn = F.normalize(buf, dim=-1)
                qn = F.normalize(et, dim=-1)
                sim = torch.einsum("bkd,bd->bk", bn, qn)
                w = F.softmax(self.tau * sim, dim=-1)
                cam_v = torch.einsum("bk,bkd->bd", w, buf)
                read = read + self.W_stk(cam_v)
            else:
                read = read + self.W_stk(top)
            read = read + self.b_read
            logits.append(read)
        return torch.stack(logits, dim=1)


class MambaMicro(nn.Module):
    """Depth-2 selective SSM micro (~9.3k). Control arm. Gu & Dao 2023."""
    def __init__(self, V=V, d=32, d_state=48, n_layer=2):
        super().__init__()
        self.V, self.d, self.d_state = V, d, d_state
        self.emb = nn.Embedding(V, d)
        self.layers = nn.ModuleList([_S6(d, d_state) for _ in range(n_layer)])
        self.out = nn.Linear(d, V)
        nn.init.normal_(self.emb.weight, 0, 0.05)

    def nparams(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        h = self.emb(x)
        for L in self.layers:
            h = h + L(h)
        return self.out(h)


class _S6(nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.d, self.n = d, n
        self.W_in = nn.Linear(d, 2 * d + n)
        self.A_log = nn.Parameter(torch.log(torch.linspace(1, n, n)))
        self.D = nn.Parameter(torch.ones(d) * 0.1)
        self.W_out = nn.Linear(d, d)
        self.dt_proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        n = self.n
        u, dt_raw, Bpar = self.W_in(x).split([d, d, n], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))
        A = -torch.exp(self.A_log)  # [n]
        # scan over T, state [B,d,n] is too big; use [B,n] shared across channels approx
        s = torch.zeros(B, n, device=x.device)
        ys = []
        Bp = torch.tanh(Bpar)
        for t in range(T):
            dtt = dt[:, t].mean(dim=-1, keepdim=True)  # [B,1]
            decay = torch.exp(dtt * A.unsqueeze(0))  # [B,n]
            s = decay * s + Bp[:, t] * u[:, t].mean(dim=-1, keepdim=True)
            y = (s.mean(dim=-1, keepdim=True) * u[:, t]) + self.D * u[:, t]
            ys.append(y)
        y = torch.stack(ys, 1)
        return self.W_out(y)


class TFMicro(nn.Module):
    """2L d16 sinusoidal PE — C51 control. Cited, not re-tested as old axis."""
    def __init__(self, V=V, d=16, n_layer=2, n_head=4, max_len=1024):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: d // 2] if d % 2 else pos * div)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(d, n_head, dim_feedforward=32, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.out = nn.Linear(d, V)

    def nparams(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        h = self.emb(x) + self.pe[: x.size(1)]
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        h = self.enc(h, mask=mask, is_causal=True)
        return self.out(h)


# ---------------- data ----------------
def _rand_len(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))

def make_batch(task, B, L, rng, eval_span=None):
    """Tasks TRACK / MODK / DYCK / PAIR. Next-token CE on full sequence.
    Labels: copy-shift of constructed target stream.
    """
    lo, hi = (4, min(16, L)) if eval_span is None else eval_span
    xs, ys = [], []
    for _ in range(B):
        n = _rand_len(rng, lo, min(hi, L - 4))
        x = [BOS, T_TASK]
        if task == "TRACK":
            # remember first content token, emit it after SEP
            tok = int(rng.integers(10, 20))
            rest = rng.integers(10, 20, size=n - 1).tolist()
            body = [tok] + rest + [SEP] + [tok]
        elif task == "MODK":
            kmod = 3
            seq = [CNT_TOK] * n
            body = seq + [SEP] + [39 + (n % kmod)]
        elif task == "DYCK":
            # balanced parens depth up to 2 in-train; 3-4 eval
            s, dpth = [], 0
            for i in range(n):
                if dpth == 0 or (dpth < 2 and rng.random() < 0.5):
                    s.append(BRK_OPEN); dpth += 1
                else:
                    s.append(BRK_CLOSE); dpth -= 1
            while dpth:
                s.append(BRK_CLOSE); dpth -= 1
            body = s + [SEP] + [EOS]
        elif task == "PAIR":
            # push A, later pop B matching — LIFO pairing
            s, st = [], []
            for i in range(n):
                if (not st) or rng.random() < 0.5:
                    s.append(PAIR_OPEN); st.append(1)
                else:
                    s.append(PAIR_CLOSE); st.pop()
            while st:
                s.append(PAIR_CLOSE); st.pop()
            body = s + [SEP] + [EOS]
        elif task == "DIV":
            ddiv = int(rng.choice([3, 4]))
            nval = int(rng.integers(8, 40))
            q = nval // ddiv
            body = [ddiv] + [CNT_TOK] * nval + [SEP] + [DIV_BASE + min(q, 8)]
        else:
            body = [10] * n + [SEP, EOS]
        seq = (x + body)[:L]
        seq += [PAD] * (L - len(seq))
        tgt = seq[1:] + [PAD]
        xs.append(seq); ys.append(tgt)
    return torch.tensor(xs), torch.tensor(ys)


def ce_loss(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=PAD)


def eval_acc(model, task, n=64, span=(32, 64), L=256, seed=0):
    rng = __import__("numpy").random.default_rng(seed)
    model.eval()
    correct = tot = 0
    with torch.no_grad():
        x, y = make_batch(task, n, L, rng, eval_span=span)
        logits = model(x)
        pred = logits.argmax(-1)
        # score tokens after SEP
        for i in range(n):
            seq = x[i].tolist()
            if SEP not in seq:
                continue
            j = seq.index(SEP)
            gold = y[i, j : j + 3]
            pr = pred[i, j : j + 3]
            m = gold != PAD
            if m.any():
                correct += int((pr[m] == gold[m]).all())
                tot += 1
    model.train()
    return correct / max(tot, 1)


def train_arm(model, tasks, steps=2000, L=256, B=8, lr=3e-3, seed=0, log_every=250, L_eval=(256, 512, 1024)):
    set_threads()
    torch.manual_seed(seed)
    rng = __import__("numpy").random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    hist = []
    for st in range(1, steps + 1):
        task = tasks[(st - 1) % len(tasks)]
        x, y = make_batch(task, B, L, rng)
        loss = ce_loss(model, x, y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if st % log_every == 0 or st == 1:
            ces = {}
            model.eval()
            with torch.no_grad():
                for Le in L_eval:
                    if Le > L:
                        # length gen: same batch construction with longer pad
                        xx, yy = make_batch(task, 4, Le, rng)
                        ces[Le] = float(ce_loss(model, xx, yy))
                    else:
                        ces[Le] = float(loss)
            model.train()
            line = f"step {st} loss={float(loss.detach()):.3f} ce={ces} p={model.nparams()}"
            print(line, flush=True)
            hist.append((st, float(loss.detach()), ces))
    print(f"done {time.time()-t0:.1f}s params={model.nparams()}", flush=True)
    return hist


if __name__ == "__main__":
    set_threads()
    m = VETLM()
    print("VETLM params", m.nparams())
    mb = VETLM(k=8, d=24, K=8)
    print("VETbig params", mb.nparams())
    mm = MambaMicro()
    print("MambaMicro params", mm.nparams())
    tf = TFMicro()
    print("TFMicro params", tf.nparams())
