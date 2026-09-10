# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 19c (cycle 58/59) — VETDCC-big x DEPTH-DIVERSE
MIX CORPUS: THE MISSING CELL OF THE FUSION 2x2. C58 ran the arm x
corpus matrix on STACKDCC2-big only:
    P16  VETDCC-big x vanilla @4000: pair .7925/.7925/.717 = 3/3
         (mean .767), ratio <=.6 3/3, modk 1.0 3/3  (the pair-
         carrying certification)
    P19  STACKDCC2-big x mixdd @4000: dyck d12 3/3 .957 (survives
         fusion), pair 1/3, ratio 0/3, modk 1.0 3/3
    P19B STACKDCC2-big x vanilla @4000: dyck d12 0/3 (.682),
         pair 1/3, ratio 3/3, modk 1.0 3/3
P19B proved STACKDCC2-big is INTRINSICALLY pair-lottery (1/3 even
on the vanilla corpus), so P19's pair/ratio failure could NOT be
blamed on the mix corpus for that arm. P19c = VETDCC-big (the
pair-carrying arm, P16 3/3) x the SAME mixdd corpus (P19's
gen_mixdd_pool, depth-diverse exact-shape random-type dyck 2..6
weighted d-1 inside the 4-task stream) at the SAME protocol (seeds
111/222/333, ctor under per-seed manual_seed, pool 512 L=256 seed
12345, 4000 steps batch 8). Three discriminating questions:
  (a) DYCK: does the no-type-stack arm (continuous VET state +
      depth/mod channels; P13D single-task d12 .89-.96) hold the
      deep-dyck basin under the mix? If d12 >=.85 in >=2/3 ->
      L-MIXDD-DYCK-CORPUS-CARRIED (arm-orthogonal: deep-dyck under
      fusion needs neither the type stack nor single-task
      isolation). If <2/3 -> L-MIXDD-DYCK-ARM-DEPENDENT (the type
      stack IS required once the stream is mixed/diluted).
  (b) PAIR: does the pair-carrying arm hold pair >=.717 in >=2/3
      under the mixdd corpus (vs its vanilla 3/3, P16)? If yes ->
      L-MIXDD-PAIR-CORPUS-INERT: the mix does NOT starve pair; the
      P19 pair failure was purely STACKDCC2's intrinsic arm lottery
      (already banked at P19B), and the fusion route can use
      VETDCC-big for pair + the mix corpus for dyck. If <2/3 ->
      L-MIXDD-STARVES-PAIR: even the pair-carrying arm loses pair
      OOD under equal-rate mixing (deep d5/d6 segments dominate
      stream tokens -> pair exemplar density collapses), so fusion
      REQUIRES per-family budgets regardless of arm.
  (c) RATIO: CE 1024/256hard <=.6 in >=2/3 (P16 vanilla 3/3 vs P19
      STACKDCC2-mixdd 0/3)? If VETDCC-mixdd 3/3 -> the ratio break
      under mixing is ARM-coupled (VETDCC's continuous register
      absorbs the deep-dyck CE load); if 0/3 -> corpus-coupled on
      both arms (the deep segments' long-range structure inflates
      hard-interval-256 CE relative to long streams for both).
Prior art (searched 2026-09-07/08): this program's P16/P19/P19B
(matrix above); Zhou et al. 2023 arXiv 2310.13349 (depth-diverse
training induces recursive structure); Zhou et al. 2024 arXiv
2402.09371 (OOD generalization fragile to init/order — why every
cell needs seeds 111/222/333, seed hygiene law). No new mechanism
(control cell re-using certified corpus + arm machinery) -> reuse
P19's generator/eval verbatim for cross-run comparability.
Tag ARCH-VET-LM-P19C.
"""
import importlib
import json, os, time
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

import arch_vet_p19 as p19
from arch_vet_p13d import VETDCC

if __name__ == "__main__":
    pool = p19.gen_mixdd_pool(512, 256, 12345)  # SAME mixdd corpus as P19
    result = {"tag": "ARCH-VET-LM-P19C",
              "protocol": "missing 2x2 cell: VETDCC-big (21,257p) "
                          "x the P19 depth-diverse mixdd corpus "
                          "(dyck 2..6 weighted d-1 in the 4-task "
                          "stream) x seeds 111/222/333, ctor under "
                          "manual_seed, pool 512 L=256 seed 12345, "
                          "4000 steps batch 8; eval = P19's eval_full "
                          "(task_acc train/eval + CE sweep) + "
                          "eval_dyck (rt_close_acc d1-12). Matrix: "
                          "P16 VETDCC-big x vanilla = pair 3/3 ratio "
                          "3/3; P19 STACKDCC2-big x mixdd = dyck 3/3 "
                          "pair 1/3 ratio 0/3; P19B STACKDCC2-big x "
                          "vanilla = dyck 0/3 pair 1/3 ratio 3/3.",
              "arms": {}}
    rows = []
    for seed in (111, 222, 333):
        torch.manual_seed(seed)
        m = VETDCC(p19.V, 24, k=8, K=8)
        print(f"[VETDCC-big-mixdd seed {seed}] params={p19.n_params(m)}",
              flush=True)
        hist = p19.train_arm(f"VETDDC-mix-s{seed}", m, pool, 4000, 8)
        row = p19.eval_full(m, "VETDDC-mix", seed)
        row["loss_curve"] = hist
        row["dyck_close"] = p19.eval_dyck(m, "VETDDC-mix", seed)
        rows.append(row)
    d12 = [r["dyck_close"]["close_d12"] for r in rows]
    mods = [r["acc_eval_interval"]["modk"] for r in rows]
    pairs = [r["acc_eval_interval"]["pair"] for r in rows]
    ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
    summary = {
        "dyck_close_d12_dist": d12,
        "basin_dyck_d12_ge_85": sum(1 for c in d12 if c >= 0.85) / 3,
        "modk_eval_dist": mods,
        "basin_modk_eq_1": sum(1 for m2 in mods if m2 == 1.0) / 3,
        "pair_eval_dist": pairs,
        "basin_pair_ge_717": sum(1 for p in pairs if p >= 0.717) / 3,
        "len_ratio_dist": ratios,
        "basin_ratio_le_6": sum(1 for r2 in ratios if r2 <= 0.6) / 3,
        "dyck_close_d12_mean": round(sum(d12) / 3, 4)}
    print(f"[VETDCC-big-mixdd] SUMMARY "
          f"{json.dumps(summary)}", flush=True)
    result["arms"]["VETDCC-big-mixdd"] = {
        "per_seed": {str(s): r for s, r in zip((111, 222, 333), rows)},
        "summary": summary}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
