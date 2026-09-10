# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 8 (cycle 52) — VETCAM: CONTENT-ADDRESSED LIFO
READOUT. Phase-1 prior art (searched 2026-08-27):
  - Differentiable LIFO line: Stack neural module networks
    (Wiley 10.1002/ail2.39, 2021) = LIFO pool + one-hot top pointer,
    read = TOP ONLY (order); Joulin & Mikolov 2015, Grefenstette
    2015 (NTM push/pop); Stogin et al. 2020 (arXiv 2006.03651,
    stable TM): SOFT stacks unstable when push/pop intensities
    mismatch.
  - Content-addressed line: RAM-Net (arXiv 2602.11958) = sparse
    content addresses + temperature-scaled top-K read (NO LIFO
    structure; addresses are high-dim sparse codes, not stored
    content); survey arXiv 2607.25380 (attention = content-
    addressable transient memory; explicit slots LM2/Engram);
    Mamba line (2312.00752 / 2405.21060 / 2603.15569) state-internal
    content read only, no cross-slot addressing (2411.02941 flags
    SSM content-reasoning weakness).
  GAP: no native LM architecture with EXACT LIFO write structure
  + soft content-addressed read across the whole buffer + top-of-
  stack fallback. The buffer stays EXACT (STE push — avoids the
  Stogin soft-stack instability); only the READOUT is soft.
MUTATION of VETLM (P2/P3 showed the order-based top-of-stack read
is init-fragile on PAIR, the query-by-content task):
  gate_j = valid_j exp(tau sim(xt, buf_j)) / sum_l valid_l
           exp(tau sim(xt, buf_j)),  sim = cos, tau = exp(softplus
           (tlog)) LEARNABLE (init ~1.31 soft);
  gate_K = max(0, 1 - sum_j gate_j)  (fallback: current-token row);
  logits += einsum(gate, T)          (same T table as VETLM, now
                                      content-gated instead of
                                      order-gated). Everything else
  (Mealy s, register R, STE push, M bilinear) identical.
  Params: 8,372 + 1 (tlog) = 8,373.
Protocol: P1 4-task stream, 2000 steps, 2 seeds (0, 111) — tests
whether content addressing stabilizes the pair basin (P3 found
.604 in 1/3 base inits) — + PAIR-gap frontier 32-64/64-96/96-144
(20 streams each). Tag ARCH-VET-LM-P8.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

_src = open("arch_vet_lm.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns)
V = _ns["V"]; VETLM = _ns["VETLM"]
make_pool = _ns["make_pool"]; make_batches = _ns["make_batches"]
train_arm = _ns["train_arm"]; val_ce = _ns["val_ce"]
task_acc = _ns["task_acc"]; n_params = _ns["n_params"]
BOS = _ns["BOS"]; T_TASK = _ns["T_TASK"]; A = _ns["A"]
KEYS = _ns["KEYS"]; VALS = _ns["VALS"]; MODS = _ns["MODS"]
EOS = _ns["EOS"]


class VETCAM(VETLM):
    """VETLM with content-addressed (soft, learned-temperature)
    readout gating over the exact LIFO buffer."""
    def __init__(self, V, d, k=5, K=4):
        super().__init__(V, d, k=k, K=K)
        self.tlog = nn.Parameter(torch.zeros(1))     # tau = softplus

    def forward(self, x):
        B, L = x.shape
        e = self.E(x)
        R = torch.zeros(B, self.d, device=x.device)
        s = torch.full((B, self.k), 1.0 / self.k, device=x.device)
        buf = torch.zeros(B, self.K, self.d, device=x.device)
        valid = torch.zeros(B, self.K, dtype=torch.bool, device=x.device)
        tau = torch.exp(F.softplus(self.tlog))
        lg = torch.empty(B, L, V, device=x.device)
        for t in range(L):
            xt = e[:, t]
            s = F.softmax(self.Ws(xt) + self.Wss(s), -1)
            a = (s.unsqueeze(-1)
                 * torch.exp(-F.softplus(self.Alog))).sum(1)
            w = torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            R = a * R + w
            g = torch.sigmoid(self.Wg(torch.cat([s, xt], -1)))
            push = (g > 0.5) + (g - g.detach())
            buf = torch.roll(buf, 1, dims=1)
            buf[:, 0] = xt * push
            valid = torch.roll(valid, 1, dims=1)
            valid[:, 0] = (g > 0.5).squeeze(-1)
            y = self.Wo(R + xt)
            feat = torch.stack(
                [buf[:, j] for j in range(self.K)] + [xt], 1)
            logits = self.head(y)
            logits = logits + torch.einsum(
                "bk,bjd,kdv->bv", s, feat, self.M)
            # content-addressed gate over buffer slots
            num = torch.einsum("bd,bkd->bk", xt, buf)          # dot
            den = (xt.norm(dim=-1, keepdim=True)
                   * buf.norm(dim=-1) + 1e-6)
            sim = (num / den).clamp(-1.0, 1.0)
            score = torch.where(valid, tau * sim,
                                torch.full_like(sim, -1e4))
            wts = torch.softmax(score, dim=-1)
            wts = wts * valid.any(dim=-1, keepdim=True).float()
            gateK = wts.sum(dim=-1, keepdim=True)
            gate = torch.cat([wts, (1.0 - gateK).clamp(min=0.0)], -1)
            logits = logits + torch.einsum("bs,ksv->bv", gate, self.T)
            lg[:, t] = logits
        return lg


def gen_pair(rng, L, gap_lo, gap_hi):
    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        if room < 10:
            x += [rng.randrange(8) + MODS] * room
            break
        gap = min(rng.randrange(gap_lo, gap_hi + 1), max(1, room - 8))
        i, j = rng.randrange(4), rng.randrange(4)
        x += ([T_TASK, T_TASK, KEYS + i, VALS + j]
              + [rng.randrange(8) + MODS for _ in range(gap)]
              + [A, KEYS + i, VALS + j])
    x = x[:L]
    x.append(EOS)
    return x


def pair_acc(model, n, gap_lo, gap_hi):
    model.eval()
    L = gap_hi + 10
    ok = tot = 0
    with torch.no_grad():
        for i in range(n):
            x = torch.tensor(gen_pair(random.Random(666 + i), L,
                                      gap_lo, gap_hi)).unsqueeze(0)
            lg = model(x)[:, :L, :]
            pred = lg.argmax(-1).squeeze(0)
            xl = x.squeeze(0).tolist()
            j = 0
            while j < L - 1:
                if (xl[j] == T_TASK and xl[j + 1] == T_TASK
                        and KEYS <= xl[j + 2] < KEYS + 4):
                    k = j
                    while k < L - 1 and xl[k] != A:
                        k += 1
                    if k < L - 2:
                        tot += 1
                        ok += int(int(pred[k]) == xl[k + 1]
                                  and int(pred[k + 1]) == xl[k + 2])
                    j = k + 3
                else:
                    j += 1
    return (ok / tot) if tot else float("nan"), tot


if __name__ == "__main__":
    pool = make_pool(512, 256, 12345)
    result = {"tag": "ARCH-VET-LM-P8",
              "protocol": "VETCAM (content-addressed LIFO readout, "
                          "learned tau), P1 4-task stream, 2000 "
                          "steps, seeds 0/111; PAIR-gap frontier "
                          "32-64/64-96/96-144 (20 streams); "
                          "baselines: cite P1/P3 (not re-run)",
              "seeds": {}}
    for seed in (0, 111):
        torch.manual_seed(seed)
        m = VETCAM(V, 16, k=5, K=4)
        print(f"[P8 seed {seed}] params={n_params(m)}", flush=True)
        hist = train_arm(f"P8-{seed}", m, pool, 2000, 8)
        m.eval()
        rng = random.Random(999);  vtr = make_batches(8, 256, rng)
        rh  = random.Random(777);  vha = make_batches(8, 256, rh, hard=True)
        r5  = random.Random(31337); v512 = make_batches(2, 512, r5)
        r10 = random.Random(31415); v1024 = make_batches(2, 1024, r10)
        ce = {"256_train": val_ce(m, vtr, 256),
              "256_hard": val_ce(m, vha, 256),
              "512": val_ce(m, v512, 512),
              "1024": val_ce(m, v1024, 1024)}
        at = task_acc(m, 24, 256, random.Random(555), hard=False)
        ae = task_acc(m, 24, 256, random.Random(666), hard=True)
        fr = {}
        for (lo, hi) in ((32, 64), (64, 96), (96, 144)):
            a, tot = pair_acc(m, 20, lo, hi)
            fr[f"gap{lo}_{hi}"] = round(a, 4)
        tau = float(torch.exp(F.softplus(m.tlog)))
        print(f"[P8 seed {seed}] ce={ce} tau={tau:.3f}", flush=True)
        print(f"[P8 seed {seed}] acc(eval)={ {k: round(v,3) for k,v in ae.items()} }", flush=True)
        print(f"[P8 seed {seed}] pair_frontier={fr}", flush=True)
        result["seeds"][str(seed)] = {
            "tau": round(tau, 4), "loss_curve": hist, "ce": ce,
            "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
            "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()},
            "pair_frontier": fr}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
