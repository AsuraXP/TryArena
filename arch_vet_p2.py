# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 2 — additive component ablations of the VET architecture
(cycle 51). Same 4-task stream / pool / 2000 steps / probes as the Phase-1
base (arch_vet_lm.py). Components (subset chain):
  A1  Mealy controller (k=5) + state x query bilinear readout only
      (no register, no LIFO)          — 'control + query'
  A2  A1 + soft value register R_t = a(s) R_{t-1} + sum s_k Ww[k] x_t
      (per-state decay rows)          — + 'value channel'
  A3  A2 + exact top-4 LIFO (STE push, stack table T, full bilinear)
      = Phase-1 base VETLM            — + 'exact LIFO side channel'
Model/data/probe code is exec'd from the committed arch_vet_lm.py (trailing
__main__ guard stripped); A1/A2 are minimal re-implementations of the subset
paths on the same embedding/controller, zero-init readout (L-GATE-INIT).
Tag ARCH-VET-LM-P2.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.manual_seed(0)
T0 = time.time()

_src = open("arch_vet_lm.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns)
V = _ns["V"]; VETLM = _ns["VETLM"]
make_pool = _ns["make_pool"]; make_batches = _ns["make_batches"]
train_arm = _ns["train_arm"]; val_ce = _ns["val_ce"]
task_acc = _ns["task_acc"]; n_params = _ns["n_params"]


class VETA1(nn.Module):
    """A1: Mealy controller + state x query bilinear readout only."""
    def __init__(self, V, d, k=5):
        super().__init__()
        self.d, self.k = d, k
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, std=0.02)
        self.Ws = nn.Linear(d, k)
        self.Wss = nn.Linear(k, k, bias=False)
        self.M = nn.Parameter(torch.zeros(k, d, V))      # zero-init
        self.head = nn.Linear(d, V)

    def forward(self, x):
        B, L = x.shape
        e = self.E(x)
        s = torch.full((B, self.k), 1.0 / self.k, device=x.device)
        lg = torch.empty(B, L, V, device=x.device)
        for t in range(L):
            xt = e[:, t]
            s = F.softmax(self.Ws(xt) + self.Wss(s), -1)
            logits = self.head(xt)
            logits = logits + torch.einsum(
                "bk,bd,kdv->bv", s, xt, self.M)
            lg[:, t] = logits
        return lg


class VETA2(nn.Module):
    """A2: A1 + soft value register (no exact LIFO)."""
    def __init__(self, V, d, k=5):
        super().__init__()
        self.d, self.k = d, k
        self.E = nn.Embedding(V, d)
        nn.init.normal_(self.E.weight, std=0.02)
        self.Ws = nn.Linear(d, k)
        self.Wss = nn.Linear(k, k, bias=False)
        self.Alog = nn.Parameter(torch.full((k, d), -3.0))
        self.Ww = nn.Parameter(0.1 * torch.randn(k, d, d))
        self.Wo = nn.Linear(d, d, bias=False)
        self.M = nn.Parameter(torch.zeros(k, d, V))      # zero-init
        self.head = nn.Linear(d, V)

    def forward(self, x):
        B, L = x.shape
        e = self.E(x)
        R = torch.zeros(B, self.d, device=x.device)
        s = torch.full((B, self.k), 1.0 / self.k, device=x.device)
        lg = torch.empty(B, L, V, device=x.device)
        for t in range(L):
            xt = e[:, t]
            s = F.softmax(self.Ws(xt) + self.Wss(s), -1)
            a = (s.unsqueeze(-1)
                 * torch.exp(-F.softplus(self.Alog))).sum(1)
            w = torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            R = a * R + w
            y = self.Wo(R + xt)
            logits = self.head(y)
            logits = logits + torch.einsum(
                "bk,bd,kdv->bv", s, xt, self.M)
            lg[:, t] = logits
        return lg


def eval_arm(m, tag):
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
    print(f"[{tag}] ce={ce}", flush=True)
    print(f"[{tag}] acc(train)={{ {', '.join(f'{k}: {v:.3f}' for k, v in at.items())} }}", flush=True)
    print(f"[{tag}] acc(eval)={{ {', '.join(f'{k}: {v:.3f}' for k, v in ae.items())} }}", flush=True)
    return ce, at, ae


if __name__ == "__main__":
    pool = make_pool(512, 256, 12345)          # identical shared pool
    print(f"[data] pool {len(pool)} x 256", flush=True)
    arms = [("A1-control+query", VETA1(V, 16, k=5)),
            ("A2+register", VETA2(V, 16, k=5)),
            ("A3+LIFO(base)", VETLM(V, 16, k=5, K=4))]
    result = {"tag": "ARCH-VET-LM-P2",
              "protocol": "same 4-task stream / 512-pool(seed 12345) / "
                          "2000 steps / seed 0 / probes as Phase-1; "
                          "A1=no register no LIFO, A2=+soft register, "
                          "A3=full base (re-run same-session)",
              "ablations": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_arm(name, m, pool, 2000, 8)
        ce, at, ae = eval_arm(m, name)
        result["ablations"][name] = {
            "params": n_params(m), "loss_curve": hist,
            "ce": ce,
            "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
            "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()},
            "len_ratio_1024_over_256hard":
                round(ce["1024"] / max(1e-9, ce["256_hard"]), 3)}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
