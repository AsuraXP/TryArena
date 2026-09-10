# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 7b (cycle 55) — DIV LENGTH ISOLATION CONTROL.
P7 found IDENTICAL 0.6/0.45/0.0 frontiers (VETbase/VETbig/MAMBA) but
those accs were measured at L = n_hi+8 (single block, NO length
stress), while CE@1024evaln diverged (VETbase 15.19). P7b measures
the SAME trained VETbase on each n-band at L = n_hi+8 (reproduces
P7) vs L=256 vs L=1024 (multi-block, length stress): separates the
count-RANGE axis from the LENGTH axis on the accuracy scale.
Question: is the VET's length invariance (flat on TRACK/P1) intact
for DIV, or does DIV multi-block streams break it (compounding
error across ~36 blocks at L=1024)?
Protocol: VETbase 8,372p, train n 4-12 L=256 pool 256 seed 12345,
2000 steps, seed 0 — identical to P7's VETbase arm (deterministic).
Tag ARCH-VET-LM-P7B.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.manual_seed(0)
T0 = time.time()

_src = open("arch_vet_p7.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p7.py", "exec"), _ns)
V = _ns["V"]; T_TASK = _ns["T_TASK"]; A = _ns["A"]
D3, D4 = _ns["D3"], _ns["D4"]
VETLM = _ns["VETLM"]
gen_div = _ns["gen_div"]; train_div = _ns["train_div"]
n_params = _ns["n_params"]


@torch.no_grad()
def div_acc_L(model, n, n_lo, n_hi, L):
    """P7 div_acc with parameterized L (multi-block streams at long L)."""
    model.eval()
    ok = tot = 0
    for i in range(n):
        x = torch.tensor(gen_div(random.Random(888 + i * 31 + L), L,
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


if __name__ == "__main__":
    m = VETLM(V, 16, k=5, K=4)
    print(f"[P7b VETbase] params={n_params(m)}", flush=True)
    hist = train_div("P7b", m, 2000, 8)
    result = {"tag": "ARCH-VET-LM-P7B",
              "protocol": "VETbase trained exactly as P7 (n 4-12 L=256 "
                          "pool 256 seed 12345, 2000 steps, seed 0); "
                          "acc per n-band at L=n_hi+8 (P7 protocol, "
                          "single block) vs L=256 vs L=1024 "
                          "(multi-block length stress); 20 streams/point",
              "params": n_params(m), "loss_curve": hist,
              "frontier_by_length": {}}
    for (lo, hi) in [(13, 16), (17, 20), (21, 24)]:
        row = {}
        for L in [hi + 8, 256, 1024]:
            a, tot = div_acc_L(m, 20, lo, hi, L)
            row[f"L{L}"] = round(a, 4)
            print(f"  [P7b] n {lo}-{hi} L={L}: acc {a:.3f} ({tot} blocks)",
                  flush=True)
        result["frontier_by_length"][f"n{lo}_{hi}"] = row
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
