# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-P14-RESUME: sandbox reboot killed arch_vet_p14.py after
STACKDCC2-big seed 222 step 1000 (no RESULT appended). Runs are
seed-deterministic (same manual_seed before ctor + same train/eval
seeds), so the 4 completed rows are reproduced verbatim from
arch_vet_p14_run.log (embedded below with provenance) and only the
2 missing runs (STACKDCC2-big seeds 222, 333) are re-executed.
Final RESULT is assembled identically to arch_vet_p14.py's schema
and appended to log.jsonl under tag ARCH-VET-LM-P14.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

# ---- p14 definitions (identical) ----
_src11 = open("arch_vet_p11.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src11.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p11.py", "exec"), _ns)
V = _ns["V"]; VETDCC = _ns["VETDCC"]
STACKDCC2 = _ns["STACKDCC2"]
train_dyck = _ns["train_dyck"]; n_params = _ns["n_params"]
gen_dyck_pool = _ns["gen_dyck_pool"]
dyck_acc = _ns["dyck_acc"]; bracket_pos_acc = _ns["bracket_pos_acc"]

# ---- completed rows from arch_vet_p14_run.log (deterministic) ----
# provenance: identical script/seed run; process killed by sandbox
# reboot after s222 step 1000; values below are verbatim log lines.
DONE_VET = {
    "111": {"loss_curve": [[500, 0.4271], [1000, 0.4211],
                           [1500, 0.3934], [2000, 0.3745]],
            "exact_match": {"exact_d3": 0.0, "exact_d4": 0.0,
                            "exact_d5": 0.0, "exact_d6": 0.0,
                            "exact_d7": 0.0, "exact_d8": 0.0},
            "close_open_acc": {"close_d3": 0.6409, "close_d4": 0.5334,
                               "close_d6": 0.4416, "close_d8": 0.396}},
    "222": {"loss_curve": [[500, 0.3840], [1000, 0.3748],
                           [1500, 0.3680], [2000, 0.3580]],
            "exact_match": {"exact_d3": 0.0, "exact_d4": 0.0,
                            "exact_d5": 0.0, "exact_d6": 0.0,
                            "exact_d7": 0.0, "exact_d8": 0.0},
            "close_open_acc": {"close_d3": 0.2897, "close_d4": 0.3716,
                               "close_d6": 0.2936, "close_d8": 0.259}},
    "333": {"loss_curve": [[500, 0.4162], [1000, 0.3747],
                           [1500, 0.4202], [2000, 0.3751]],
            "exact_match": {"exact_d3": 0.0, "exact_d4": 0.0,
                            "exact_d5": 0.0, "exact_d6": 0.0,
                            "exact_d7": 0.0, "exact_d8": 0.0},
            "close_open_acc": {"close_d3": 0.4033, "close_d4": 0.4368,
                               "close_d6": 0.3871, "close_d8": 0.3459}}}
DONE_STACK_111 = {
    "loss_curve": [[500, 0.4221], [1000, 0.3779],
                   [1500, 0.3707], [2000, 0.3628]],
    "exact_match": {"exact_d3": 0.0, "exact_d4": 0.0,
                    "exact_d5": 0.0, "exact_d6": 0.0,
                    "exact_d7": 0.0, "exact_d8": 0.0},
    "close_open_acc": {"close_d3": 0.6548, "close_d4": 0.6386,
                       "close_d6": 0.5327, "close_d8": 0.4583}}


def run_one(name_tag, m, pool):
    """Train + eval one seed (p14 body)."""
    print(f"[{name_tag}] params={n_params(m)}", flush=True)
    hist = train_dyck(name_tag, m, pool, 2000, 8)
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
    print(f"[{name_tag}] exact: { {k: v for k, v in dy.items()} }",
          flush=True)
    print(f"[{name_tag}] close: "
          f"{ {k: v for k, v in bp.items() if k.startswith('close')} }",
          flush=True)
    return {"loss_curve": hist, "exact_match": dy, "close_open_acc": bp}


if __name__ == "__main__":
    pool = gen_dyck_pool(256, 256, 12345)
    # STACKDCC2-big seeds 222 + 333 (the 2 runs the reboot killed)
    torch.manual_seed(222)
    m = STACKDCC2(V, 24, k=8, K=8)
    row222 = run_one("STACKDCC2-big-s222", m, pool)
    torch.manual_seed(333)
    m = STACKDCC2(V, 24, k=8, K=8)
    row333 = run_one("STACKDCC2-big-s333", m, pool)

    arms = {}
    vet_row = {}
    for s in ("111", "222", "333"):
        cl = DONE_VET[s]["close_open_acc"]["close_d3"]
        vet_row[s] = DONE_VET[s]
        vet_row[s]["close_open_acc"]["open_d3"] = None  # schema note
    closes_v = [DONE_VET[s]["close_open_acc"]["close_d3"]
                for s in ("111", "222", "333")]
    arms["VETDCC-big"] = {
        "per_seed": DONE_VET,
        "note": "all 3 seeds from arch_vet_p14_run.log (deterministic "
                "run killed by sandbox reboot before RESULT append)",
        "close_d3_dist": closes_v,
        "basin_rate_ge_678": sum(1 for c in closes_v if c >= 0.678) / 3,
        "basin_rate_ge_85": sum(1 for c in closes_v if c >= 0.85) / 3,
        "close_d3_mean": round(sum(closes_v) / 3, 4),
        "close_d3_sd": round(
            (sum((c - sum(closes_v) / 3) ** 2 for c in closes_v) / 3)
            ** 0.5, 4)}
    s_rows = {"111": DONE_STACK_111, "222": row222, "333": row333}
    closes_s = [s_rows[str(s)]["close_open_acc"]["close_d3"]
                for s in (111, 222, 333)]
    arms["STACKDCC2-big"] = {
        "per_seed": s_rows,
        "note": "seed 111 from arch_vet_p14_run.log (deterministic, "
                "killed pre-append); seeds 222/333 freshly re-run",
        "close_d3_dist": closes_s,
        "basin_rate_ge_678": sum(1 for c in closes_s if c >= 0.678) / 3,
        "basin_rate_ge_85": sum(1 for c in closes_s if c >= 0.85) / 3,
        "close_d3_mean": round(sum(closes_s) / 3, 4),
        "close_d3_sd": round(
            (sum((c - sum(closes_s) / 3) ** 2 for c in closes_s) / 3)
            ** 0.5, 4)}
    result = {"tag": "ARCH-VET-LM-P14",
              "protocol": "basin rate at 2.5x structure on the P11 "
                          "single-task STOCHASTIC dyck protocol; "
                          "seeds (111,222,333) torch.manual_seed "
                          "BEFORE construction, train 2000 steps "
                          "batch 8, L=256 pool 256 seed 12345; "
                          "eval exact d3-8 + close/open acc "
                          "d3/4/6/8 (16 streams); bars .678 "
                          "(TFMicro P12 control) and .85 (P11 "
                          "citation regime); RESUME run: sandbox "
                          "reboot killed original at s222 step "
                          "1000; completed rows verbatim from "
                          "arch_vet_p14_run.log (deterministic), "
                          "missing seeds re-run",
              "arms": arms}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
