# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 14 (cycle 56) — BASIN RATE AT 2.5x STRUCTURE,
MULTI-SEED. L-BASIN-SCALE-CAPTURE (handover C55) says the favorable
basin (pair-eval .717 P5 VETbig; .962 P9 VETDCC-big) is reached at
2.5x structure under plain seed-0, while at BASE budget the basin
rate is 1/3 (P6, 6-init sample). The dyck win (P11 close-d3 .925
STACKDCC2-big > TFMicro .678) is n=2 as well (P11 citation .925 +
P12 same-session re-run .791 — the two differ because p11/p12
construct arm lists at different RNG positions). P14 asks: at the
2.5x-structure big config, on the SINGLE-TASK dyck protocol where
the win lives, what is the basin rate over 3 fresh init seeds —
is the big-config basin ~1.0 (scale captures the basin) or still a
lottery at ~1/3 (the win was luck)?

PROTOCOL: seeds (111, 222, 333), torch.manual_seed(seed) BEFORE
construction (P6 pattern — construction under the seed, training
then deterministic), arms VETDCC-big (21,257p) and STACKDCC2-big
(21,817p), on the P11 single-task STOCHASTIC dyck protocol (train
depth 2, L=256 pool 256 seed 12345, 2000 steps, batch 8 — the
winning budget/protocol of P11/P12). EVAL per seed: whole-segment
exact d3-8 + per-position open/close acc d3/d4/d6/d8 (16 streams,
p11 bracket_pos_acc/dyck_acc). BASIN BARS reported: fraction of
seeds with close_d3 >= .678 (the TFMicro control bar from P12 —
any structural win must clear it) and >= .85 (the P11-citation
regime); mean/sd of close_d3 over the 3 seeds.
Prior samples for the count (seed-0, different ctor RNG positions,
noted in log): P11 STACKDCC2-big close_d3 .925; P12 re-run .791.

Prior art (no NEW mechanism — measurement protocol as P6; baseline
claims from P5/P9/P11/P12 logged in log.md C51-C56). Multi-init
robustness reporting is standard practice (e.g., "the lottery
ticket" fragility at small scale; Henderson et al. 2018 RL
reproducibility argue seed distributions over point estimates).

FALSIFIABLE PREDICTIONS (logged per program protocol):
  (a) STACKDCC2-big basin rate >= 3/3 at bar .678 (structural win
      robust across inits at 2.5x structure) — if <= 1/3,
      L-BASIN-SCALE-CAPTURE does NOT extend to dyck and the P11 win
      is seed-luck (P13's D12/D6 close-type gap already hints init
      sensitivity);
  (b) STACKDCC2-big mean close_d3 >= VETDCC-big mean close_d3
      (content stack > depth counter across inits);
  (c) VETDCC-big basin rate at bar .678 <= STACKDCC2-big's (no
      content stack -> the depth-counter basin is narrower on dyck).
Tag ARCH-VET-LM-P14.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

_src11 = open("arch_vet_p11.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src11.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p11.py", "exec"), _ns)
V = _ns["V"]; VETDCC = _ns["VETDCC"]
STACKDCC2 = _ns["STACKDCC2"]
train_dyck = _ns["train_dyck"]; n_params = _ns["n_params"]
gen_dyck_pool = _ns["gen_dyck_pool"]
dyck_acc = _ns["dyck_acc"]; bracket_pos_acc = _ns["bracket_pos_acc"]

if __name__ == "__main__":
    pool = gen_dyck_pool(256, 256, 12345)
    result = {"tag": "ARCH-VET-LM-P14",
              "protocol": "basin rate at 2.5x structure on the P11 "
                          "single-task STOCHASTIC dyck protocol; "
                          "seeds (111,222,333) torch.manual_seed "
                          "BEFORE construction, train 2000 steps "
                          "batch 8, L=256 pool 256 seed 12345; "
                          "eval exact d3-8 + close/open acc "
                          "d3/4/6/8 (16 streams); bars .678 "
                          "(TFMicro P12 control) and .85 (P11 "
                          "citation regime); prior seed-0 samples "
                          "P11 .925 / P12 .791 noted",
              "arms": {}}
    for arm_name, ctor in [("VETDCC-big",
                            lambda: VETDCC(V, 24, k=8, K=8)),
                           ("STACKDCC2-big",
                            lambda: STACKDCC2(V, 24, k=8, K=8))]:
        seeds_row = {}
        for seed in (111, 222, 333):
            torch.manual_seed(seed)
            m = ctor()
            print(f"[{arm_name} seed {seed}] params={n_params(m)}",
                  flush=True)
            hist = train_dyck(f"{arm_name}-s{seed}", m, pool, 2000, 8)
            m.eval()
            dy = {}
            for d in (3, 4, 5, 6, 7, 8):
                a, tot = dyck_acc(m, 16, d)
                dy[f"exact_d{d}"] = round(a, 4)
            bp = {}
            for d in (3, 4, 6, 8):
                o, cl, to, tc = bracket_pos_acc(m, 16, d)
                bp[f"open_d{d}"] = o
                bp[f"close_d{d}"] = cl
            seeds_row[str(seed)] = {"loss_curve": hist,
                                    "exact_match": dy,
                                    "close_open_acc": bp}
            print(f"[{arm_name} s{seed}] exact: "
                  f"{ {k: v for k, v in dy.items()} }", flush=True)
            print(f"[{arm_name} s{seed}] close: "
                  f"{ {k: v for k, v in bp.items() if k.startswith('close')} }",
                  flush=True)
        closes = [seeds_row[str(s)]["close_open_acc"]["close_d3"]
                  for s in (111, 222, 333)]
        row = {"per_seed": seeds_row,
               "close_d3_dist": closes,
               "basin_rate_ge_678": sum(1 for c in closes if c >= 0.678) / 3,
               "basin_rate_ge_85": sum(1 for c in closes if c >= 0.85) / 3,
               "close_d3_mean": round(sum(closes) / 3, 4),
               "close_d3_sd": round(
                   (sum((c - sum(closes) / 3) ** 2 for c in closes)
                    / 3) ** 0.5, 4)}
        result["arms"][arm_name] = row
        print(f"[{arm_name}] RESULT ROW: "
              f"{json.dumps({k: row[k] for k in row if k != 'per_seed'})}",
              flush=True)
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
