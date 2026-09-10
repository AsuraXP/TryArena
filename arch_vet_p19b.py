# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 19b (cycle 58) — VANILLA-CORPUS CONTROL FOR THE
MIXED-STREAM FUSION TEST. P19 (mixed-stream fusion, same run) mixed
depth-diverse exact-shape random-type dyck (depths 2..6, weighted
d-1) into the P9 4-task stream and trained STACKDCC2-big D12 at
4000 steps. Seed results (P19): dyck close d12 .9563/.9535 (2/3 at
.85) — the depth-diverse dyck win SURVIVES fusion — but pair-eval
.377/.868 and CE ratio 1024/256hard .798/.873 against the P16
certified bars (pair >=.717, ratio <=.6). CONFOUND: P16's certified
4-task properties were measured on VETDCC-big with the VANILLA P9
corpus; STACKDCC2-big at 4000 steps on the vanilla corpus was never
run (P15 only ran it @2000: pair 1/3 lottery). So the pair/ratio
degrade in P19 is attributable to (i) the deep-dyck corpus
diluting/starving pair+track exemplar density (a d6 segment = a
whole 256 stream; d5 = half), (ii) the arm difference, or both.
P19b closes the arm dimension: SAME arm (STACKDCC2-big D12), SAME
protocol (seeds 111/222/333, pool 512 L=256 seed 12345, 4000 steps
batch 8, ctor under per-seed manual_seed), on the VANILLA P9
4-task pool (gen_stream, dyck depth 2 in train / 3 hard) — P16's
corpus. Compare P19b pair/ratio/CE against P16 (VETDCC-big vanilla,
pair .7925/.7925/.717 3/3, ratio 3/3) and P19 (mixdd):
  * P19b pair >=.717 2/3 AND ratio <=.6 2/3 while P19 fails both →
    the deep-dyck corpus CAUSES the 4-task degrade (data-level
    fusion not clean at naive mix; L-MIXED-DEPTH-DYCK-STARVES-PAIR);
  * P19b also fails pair/ratio (<=1/3) → STACKDCC2-big itself is
    not 4-task-certified at 4000 even on the vanilla corpus →
    P19's degrade is (at least partly) an ARM effect; a VETDCC-big
    x mixdd arm would be the next control (P19c) before blaming
    the corpus.
Falsifiable: P19b pair >=.717 in >= 2/3 AND ratio <= .6 in >= 2/3
    AND modk 1.0 in >= 2/3 (prediction: TRUE — vanilla corpus +
    4000 steps reproduces P16's certified properties on this arm).
Tag ARCH-VET-LM-P19B.
"""
import importlib
import json, os, random, time
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

import arch_vet_p19 as p19

if __name__ == "__main__":
    pool = p19.make_pool(512, 256, 12345)      # VANILLA P9 corpus
    result = {"tag": "ARCH-VET-LM-P19B",
              "protocol": "vanilla-corpus control: STACKDCC2-big "
                          "D12 x seeds 111/222/333 on the VANILLA P9 "
                          "4-task pool (gen_stream, dyck depth 2) at "
                          "4000 steps batch 8, pool 512 L=256 seed "
                          "12345, ctor under manual_seed; same eval "
                          "as P19 (dyck rt_close_acc d1-12 + task_acc "
                          "train/eval + CE sweep). References: P16 "
                          "VETDCC-big vanilla pair 3/3 >=.717 ratio "
                          "3/3 <=.6; P19 mixdd pair .377/.868 ratio "
                          ".798/.873 dyck d12 .956/.954 (2/3).",
              "arms": {}}
    rows = []
    for seed in (111, 222, 333):
        torch.manual_seed(seed)
        m = p19.STACKDCC2_D12(p19.V, 24, k=8, K=8)
        print(f"[STACKDCC2-big-D12-vanilla seed {seed}] "
              f"params={p19.n_params(m)}", flush=True)
        hist = p19.train_arm(f"D12-van-s{seed}", m, pool, 4000, 8)
        row = p19.eval_full(m, "D12-van", seed)
        row["loss_curve"] = hist
        row["dyck_close"] = p19.eval_dyck(m, "D12-van", seed)
        rows.append(row)
    d12 = [r["dyck_close"]["close_d12"] for r in rows]
    mods = [r["acc_eval_interval"]["modk"] for r in rows]
    pairs = [r["acc_eval_interval"]["pair"] for r in rows]
    ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
    summary = {
        "dyck_close_d12_dist": d12,
        "modk_eval_dist": mods,
        "basin_modk_eq_1": sum(1 for m2 in mods if m2 == 1.0) / 3,
        "pair_eval_dist": pairs,
        "basin_pair_ge_717": sum(1 for p in pairs if p >= 0.717) / 3,
        "len_ratio_dist": ratios,
        "basin_ratio_le_6": sum(1 for r2 in ratios if r2 <= 0.6) / 3}
    print(f"[STACKDCC2-big-D12-vanilla] SUMMARY "
          f"{json.dumps(summary)}", flush=True)
    result["arms"]["STACKDCC2-big-D12-vanilla"] = {
        "per_seed": {str(s): r for s, r in zip((111, 222, 333), rows)},
        "summary": summary}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
