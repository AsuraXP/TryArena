# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 18 (cycle 57) — DEPTH-DIVERSE DYCK WIN:
MULTI-SEED ROBUSTNESS. P13D produced the only big dyck win across
P11-P17: deep-mixed exact-shape random-type training (train depths
2-6) gave STACKDCC2-big D12 close-type .943 @ d12, flat over 32x
length (L=16396) — vs the P11/P14/P17 depth-2 stochastic protocol
whose basin is 1/3 at ANY budget (P17: L-DYCK-BUDGET-NOT-CAPTURED,
mean close_d3 .674 ≈ TFMicro .678). P13D was SINGLE-SEED (seed 0);
every "big win" in this program that stayed single-seed later
shrank under multi-seed (P11 .925 → P14 1/3; P9 .962 → P15 1/3@
2000 → P16 3/3@4000). P18 asks: is the depth-diverse dyck win
init-robust, or another lucky seed? Protocol = P13D verbatim
(train depths {2..6} weighted deep, L=256 pool 256 seed 12345,
2000 steps, ctor under per-seed manual_seed), arm = STACKDCC2-big
D12 (the P13D best: .943 d12). Eval = rt_close_acc d1-12 (per-depth
L; TFMicro n/a). Bars: in-train close d6 >= .90 (3/3 mastery) and
OOD close d12 >= .85 in >= 2/3 seeds (the .92-.94 win robust) or
>= .9 in >= 2/3 (strong).

Prior art: depth-diversity induction (Zhou et al. 2023 arXiv
2310.13349); init-fragility of OOD generalization (Zhou et al.
2024 arXiv 2402.09371); this program's own P13D (win), P14/P17
(2000/4000-step basin 1/3 on the shallow protocol), P16 (pair
budget-captured). No new mechanism — multi-seed measurement.

FALSIFIABLE PREDICTIONS:
  (a) robust: close d12 >= .85 in >= 2/3 seeds AND close d6 >= .90
      in 3/3 → the depth-diverse protocol captures the dyck
      close-tracking basin across inits (L-DEPTH-DIVERSITY-
      CAPTURES-DYCK-BASIN): the depth-diverse dyck claim becomes
      the certified multi-seed dyck win (with the honest scope:
      close-type on exact-shape random-type grammar, not whole-
      segment exact, not the shallow stochastic protocol);
  (b) lucky: close d12 basin <.5 (<= 1/3 seeds) → dyck close-type
      is seed-lottery under every protocol tried; the axis needs a
      mechanism change (supervised stack-use / auxiliary stack-
      trace loss), not corpus or budget variation.
Tag ARCH-VET-LM-P18.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

_src = open("arch_vet_p13d.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p13d.py", "exec"), _ns)
V = _ns["V"]; STACKDCC2_D12 = _ns["STACKDCC2_D12"]
train_dyck = _ns["train_dyck"]; n_params = _ns["n_params"]
gen_mix_pool = _ns["gen_mix_pool"]; rt_close_acc = _ns["rt_close_acc"]
seg_len = _ns["seg_len"]

L_by_depth = {d: (256 if seg_len(d) <= 252 else seg_len(d) + 16)
              for d in range(1, 13)}


def n_streams(d):
    L = L_by_depth[d]
    if L <= 256:
        return 8
    if L <= 4096:
        return 4
    return 2


def run_seed(seed, pool):
    torch.manual_seed(seed)
    m = STACKDCC2_D12(V, 24, k=8, K=8)
    print(f"[STACKDCC2-big-D12 seed {seed}] params={n_params(m)}",
          flush=True)
    hist = train_dyck(f"D12-s{seed}", m, pool, 2000, 8)
    m.eval()
    cl = {}
    for d in range(1, 13):
        L = L_by_depth[d]
        a, tot = rt_close_acc(m, n_streams(d), d, L)
        cl[f"close_d{d}"] = a
        print(f"  [D12 s{seed}] d{d} L{L}: close-type {a} ({tot} pos)",
              flush=True)
    return {"loss_curve": hist, "close_type_by_depth": cl}


if __name__ == "__main__":
    pool = gen_mix_pool(256, 256, 12345)
    rows = {}
    for seed in (111, 222, 333):
        rows[str(seed)] = run_seed(seed, pool)
    c6 = [rows[s]["close_type_by_depth"]["close_d6"] for s in rows]
    c8 = [rows[s]["close_type_by_depth"]["close_d8"] for s in rows]
    c12 = [rows[s]["close_type_by_depth"]["close_d12"] for s in rows]
    summary = {
        "close_d6_in_train_dist": c6,
        "close_d8_dist": c8,
        "close_d12_dist": c12,
        "basin_d6_ge_90": sum(1 for c in c6 if c >= 0.9) / 3,
        "basin_d8_ge_85": sum(1 for c in c8 if c >= 0.85) / 3,
        "basin_d12_ge_85": sum(1 for c in c12 if c >= 0.85) / 3,
        "basin_d12_ge_90": sum(1 for c in c12 if c >= 0.9) / 3,
        "close_d12_mean": round(sum(c12) / 3, 4),
        "close_d12_sd": round(
            (sum((c - sum(c12) / 3) ** 2 for c in c12) / 3) ** 0.5, 4)}
    print(f"[STACKDCC2-big-D12] SUMMARY "
          f"{json.dumps({k: v for k, v in summary.items()})}", flush=True)
    result = {"tag": "ARCH-VET-LM-P18",
              "protocol": "P13D depth-mixed exact-shape random-type "
                          "dyck protocol (train depths {2..6} "
                          "weighted deep, L=256 pool 256 seed "
                          "12345, 2000 steps) x seeds 111/222/333, "
                          "STACKDCC2-big D12, ctor under manual_"
                          "seed; eval rt_close_acc d1-12; P13D "
                          "seed-0 reference close d6 .980 d8 .966 "
                          "d12 .943; bars d6>=.9 (in-train 3/3), "
                          "d12>=.85 in >=2/3 seeds",
              "arms": {"STACKDCC2-big-D12": {"per_seed": rows,
                                              "summary": summary}}}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
