# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 7 (cycle 52) — DIVIDE task: the next algorithmic
frontier (C49 NEXT item: division/GCD) as a token-prediction probe.
Single-task DIV stream (same V=48 vocab; new tokens, no collision):
  T T d 1*1*...1 A q      d in {3,4} = tok 37,38 (free slots 37..47),
  q = n // d in 0..8 = tok 39..47, n = count of ONE (tok 21).
  Shape-disjoint from all P1 families (pair is T T KEYS, modk is
  T T ONE, dyck is T T T, track is T sym).
Train: n 4..12, d {3,4}, L=256, 2000 steps, pool 256 (seed 12345),
seed 0, 1 thread — identical budget to P4. Frontier: eval n
13..16 / 17..20 / 21..24 (q up to 8; beyond 24, d=3 needs q=9 which
has no token — frontier stops where the vocab stops). Arms:
VETbase 8,372p / VETbig k8/d24/K8 20,697p / MAMBA 9,360p.
Question: does the structure (value channel holding n + controller
doing the count->quotient map) extend division beyond the training
count range, and does VETbig extend it further? Tag ARCH-VET-LM-P7.
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
V = _ns["V"]; BOS = _ns["BOS"]; EOS = _ns["EOS"]; T_TASK = _ns["T_TASK"]
A = _ns["A"]; ONE = _ns["ONE"]; MODS = _ns["MODS"]
VETLM = _ns["VETLM"]; MambaMicro = _ns["MambaMicro"]
val_ce = _ns["val_ce"]; n_params = _ns["n_params"]

D3, D4 = 37, 38          # divisor tokens (3, 4)
Q0 = 39                  # q = n//d -> tok Q0 + q (q 0..8)
assert Q0 + 9 <= V


def gen_div(rng, L, n_lo, n_hi):
    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        if room < 6:
            x += [rng.randrange(8) + MODS] * room
            break
        n = min(rng.randrange(n_lo, n_hi + 1), max(1, room - 5))
        d = (3, 4)[rng.randrange(2)]
        dtok = D3 if d == 3 else D4
        q = n // d
        assert q <= 8
        x += [T_TASK, T_TASK, dtok] + [ONE] * n + [A, Q0 + q]
    x = x[:L]
    x.append(EOS)
    return x


def div_acc(model, n, n_lo, n_hi):
    model.eval()
    L = n_hi + 8
    ok = tot = 0
    with torch.no_grad():
        for i in range(n):
            x = torch.tensor(gen_div(random.Random(888 + i), L,
                                     n_lo, n_hi)).unsqueeze(0)
            lg = model(x)[:, :L, :]
            pred = lg.argmax(-1).squeeze(0)
            xl = x.squeeze(0).tolist()
            j = 0
            while j < L - 1:
                if (xl[j] == T_TASK and xl[j + 1] == T_TASK
                        and xl[j + 2] in (D3, D4)):
                    k = j
                    while k < L - 1 and xl[k] != A:
                        k += 1
                    if k < L - 1:
                        tot += 1
                        ok += int(int(pred[k]) == xl[k + 1])
                    j = k + 2
                else:
                    j += 1
    return (ok / tot) if tot else float("nan"), tot


def train_div(name, model, steps=2000, B=8, lr=3e-3, L=256, pool_n=256):
    prng = random.Random(12345)
    pool = [torch.tensor(gen_div(prng, L, 4, 12)) for _ in range(pool_n)]
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        sel = [(step * B + i) % len(pool) for i in range(B)]
        x = torch.stack([pool[i] for i in sel])
        y = x[:, 1:]
        lg = model(x[:, :L])
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            hist.append((step, round(float(loss), 4)))
            print(f"  [{name}] step {step}/{steps} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return hist


FRONTIER = [(13, 16), (17, 20), (21, 24)]


def eval_div(m, name):
    m.eval()
    fr = {}
    for (lo, hi) in FRONTIER:
        a, tot = div_acc(m, 20, lo, hi)
        fr[f"n{lo}_{hi}"] = round(a, 4)
        print(f"  [{name}] n {lo}-{hi}: acc {a:.3f} ({tot} blocks)",
              flush=True)
    v256 = torch.stack([torch.tensor(gen_div(random.Random(999 + i),
                                             256, 4, 12))
                        for i in range(8)])
    v1024 = torch.stack([torch.tensor(gen_div(random.Random(31415),
                                              1024, 13, 24))
                         for _ in range(2)])
    ce = {"256_trainn": val_ce(m, v256, 256),
          "1024_evaln": val_ce(m, v1024, 1024)}
    print(f"  [{name}] ce={ce}", flush=True)
    return fr, ce


if __name__ == "__main__":
    arms = [("VETbase", VETLM(V, 16, k=5, K=4)),
            ("VETbig", VETLM(V, 24, k=8, K=8)),
            ("MAMBA", MambaMicro(V, 16))]
    result = {"tag": "ARCH-VET-LM-P7",
              "protocol": "single-task DIV stream (T T d 1^n A q, d in "
                          "{3,4}, q=n//d tok 39..47), train n 4-12 "
                          "L=256 2000 steps seed 0; frontier eval n "
                          "13-16/17-20/21-24, 20 streams each; CE "
                          "256(train-n) vs 1024(eval-n)",
              "arms": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_div(name, m)
        fr, ce = eval_div(m, name)
        result["arms"][name] = {"params": n_params(m),
                                "loss_curve": hist,
                                "frontier_acc": fr, "ce": ce}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
