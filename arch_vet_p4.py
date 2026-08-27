# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 4 — generalization frontier + structure scaling
(cycle 51). Question: does the structural LM generalize along the
TRACK gap axis the way the C43-C49 structural controller did along
the (a,b) value axis — and does scaling the structure (k, d, K)
extend the frontier?

Single-task TRACK stream: T x <gap fillers> A x (same vocab as
arch_vet_lm.py; gap ranges parameterized). Arms (all track-train on
gap 4-16, L=256, 2000 steps (Phase-1 step budget), seed 0, 1 thread):
  VETbase  k=5, d=16, K=4   (8,372 p)
  VETbig   k=8, d=24, K=8   (~20.7k p, 2.5x scale of structure)
  MAMBA    depth-2 d_state=48 (9,360 p)
Frontier probes: exact-match track accuracy at eval gaps
  32-64 / 64-96 / 96-144 / 144-192 / 192-256  (L = gap_hi + 8)
plus CE @256(train-gap) vs @1024.  Tag ARCH-VET-LM-P4.
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
TRACK = _ns["TRACK"]; A = _ns["A"]; MODS = _ns["MODS"]
VETLM = _ns["VETLM"]; MambaMicro = _ns["MambaMicro"]
val_ce = _ns["val_ce"]; n_params = _ns["n_params"]
make_batches = _ns["make_batches"]


def gen_track(rng, L, gap_lo, gap_hi):
    """One pure-TRACK stream of length L (gap in [lo,hi])."""
    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        if room < 6:
            x += [rng.randrange(8) + MODS] * room
            break
        gap = min(rng.randrange(gap_lo, gap_hi + 1), max(1, room - 4))
        sym = rng.randrange(8) + TRACK
        x += [T_TASK, sym] + [rng.randrange(8) + MODS for _ in range(gap)] \
             + [A, sym]
    x = x[:L]
    x.append(EOS)
    return x


def track_acc(model, n, gap_lo, gap_hi):
    model.eval()
    L = gap_hi + 8
    ok = tot = 0
    with torch.no_grad():
        for _ in range(n):
            x = torch.tensor(gen_track(random.Random(777 + _), L,
                                       gap_lo, gap_hi)).unsqueeze(0)
            lg = model(x)[:, :L, :]
            pred = lg.argmax(-1).squeeze(0)
            # find each T x ... A x block
            xl = x.squeeze(0).tolist()
            i = 0
            while i < L - 1:
                if xl[i] == T_TASK and TRACK <= xl[i + 1] < TRACK + 8:
                    j = i
                    while j < L - 1 and xl[j] != A:
                        j += 1
                    if j < L - 1:
                        tot += 1
                        ok += int(int(pred[j]) == xl[j + 1])
                    i = j + 2
                else:
                    i += 1
    return (ok / tot) if tot else float("nan"), tot


def train_track(name, model, steps=3000, B=8, lr=3e-3, L=256, pool_n=256):
    prng = random.Random(12345)
    pool = [torch.tensor(gen_track(prng, L, 4, 16)) for _ in range(pool_n)]
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


FRONTIER = [(32, 64), (64, 96), (96, 144), (144, 192), (192, 256)]


def eval_frontier(m, name):
    m.eval()
    fr = {}
    for (lo, hi) in FRONTIER:
        a, tot = track_acc(m, 20, lo, hi)
        fr[f"gap{lo}_{hi}"] = round(a, 4)
        print(f"  [{name}] gap {lo}-{hi}: acc {a:.3f} ({tot} blocks)",
              flush=True)
    rng = random.Random(999)
    v256 = torch.stack([torch.tensor(gen_track(random.Random(999 + i),
                                              256, 4, 16))
                        for i in range(8)])
    r10 = random.Random(31415)
    v1024 = torch.stack([torch.tensor(gen_track(r10, 1024, 32, 64))
                         for _ in range(2)])
    ce = {"256_traingap": val_ce(m, v256, 256),
          "1024_evalgap": val_ce(m, v1024, 1024)}
    print(f"  [{name}] ce={ce}", flush=True)
    return fr, ce


if __name__ == "__main__":
    arms = [("VETbase", VETLM(V, 16, k=5, K=4)),
            ("VETbig", VETLM(V, 24, k=8, K=8)),
            ("MAMBA", MambaMicro(V, 16))]
    result = {"tag": "ARCH-VET-LM-P4",
              "protocol": "single-task TRACK stream (T x <gap fills> A x), "
                          "train gap 4-16 L=256 2000 steps seed 0; frontier "
                          "eval gaps 32-64/64-96/96-144/144-192/192-256, "
                          "20 streams each; CE 256(train-gap) vs 1024(eval-gap)",
              "arms": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_track(name, m, steps=2000)
        fr, ce = eval_frontier(m, name)
        result["arms"][name] = {"params": n_params(m),
                                "loss_curve": hist,
                                "frontier_acc": fr, "ce": ce}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
