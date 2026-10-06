# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 17 (cycle 57) — DYCK BASIN AT 2x BUDGET.
P16 proved the PAIR-OOD lottery was 2000-step underconvergence
(L-PAIR-LOTTERY-BUDGET): at 4000 steps the big-config pair basin is
3/3 at .717 (mean .767, sd .036) vs 1/3 at 2000. P14's dyck basin
was ALSO measured at 2000 steps: STACKDCC2-big close_d3
.655/.763/.631 → 1/3 at the TFMicro .678 bar (the ".925 = lucky
basin" downgrade of C56 rests on that 2000-step measurement). P17
runs the P14 protocol at 4000 steps (2x budget): if the dyck basin
is ALSO a budget artifact, close_d3 basin@.678 should rise toward
3/3 and the C56 dyck downgrade must be re-scoped again (this time
with the mechanism understood: budget-captured, like pair).

No new mechanism — P14 protocol, doubled budget, same seed hygiene
(ctor under per-seed manual_seed). Prior art: as P14/P16 (Zhou et
al. 2024 arXiv 2402.09371 init-fragility of OOD generalization;
this program's P16 shows budget captures the pair basin).

PROTOCOL: STACKDCC2-big (21,817p) × seeds (111,222,333), P11
single-task stochastic dyck (train depth 2, L=256 pool 256 seed
12345), 4000 STEPS (2× P14's 2000), ctor under manual_seed(seed).
EVAL (P14-identical): whole-segment exact d3-8 + per-position
close/open acc d3/d4/d6/d8 (16 streams). Bars: close_d3 >= .678
(TFMicro P12 control) and >= .85 (P11 citation regime); also
close_d4/d6/d8 dist. P14 2000-step reference (same protocol):
close_d3 .655/.763/.631, basin@.678 = 1/3.

FALSIFIABLE PREDICTIONS:
  (a) budget effect: close_d3 basin@.678 >= 2/3 (>= 2 seeds clear
      .678) at 4000 steps and mean close_d3 >= .72 → the C56
      "lucky basin" reading of P14 is itself a 2000-step artifact;
      the dyck win is budget-captured (L-DYCK-BUDGET-CAPTURED) and
      the P11 .925 citation is the converged-regime sample.
  (b) structural: basin@.678 stays <= 1/3 with mean < .70 →
      dyck close-type OOD is genuinely harder than pair (deep
      type-structure routing vs gap routing) — the stack/close
      routing does not converge from most inits even at 2x budget;
      keep the C56 downgrade and attack the routing (supervised
      stack-use loss? curriculum over depth, P13d-style, at 4000
      steps?).
Tag ARCH-VET-LM-P17.
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
V = _ns["V"]; STACKDCC2 = _ns["STACKDCC2"]
train_dyck = _ns["train_dyck"]; n_params = _ns["n_params"]
gen_dyck_pool = _ns["gen_dyck_pool"]
dyck_acc = _ns["dyck_acc"]; bracket_pos_acc = _ns["bracket_pos_acc"]


def run_seed(seed, pool):
    torch.manual_seed(seed)
    m = STACKDCC2(V, 24, k=8, K=8)
    print(f"[STACKDCC2-big seed {seed}] params={n_params(m)}", flush=True)
    hist = train_dyck(f"STACKDCC2-big-s{seed}", m, pool, 4000, 8)
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
    print(f"[STACKDCC2-big s{seed}] exact: "
          f"{ {k: v for k, v in dy.items()} }", flush=True)
    print(f"[STACKDCC2-big s{seed}] close: "
          f"{ {k: v for k, v in bp.items() if k.startswith('close')} }",
          flush=True)
    return {"loss_curve": hist, "exact_match": dy,
            "close_open_acc": bp}


if __name__ == "__main__":
    pool = gen_dyck_pool(256, 256, 12345)
    rows = {}
    for seed in (111, 222, 333):
        rows[str(seed)] = run_seed(seed, pool)
    closes = {s: rows[s]["close_open_acc"]["close_d3"] for s in rows}
    c3 = [closes[s] for s in ("111", "222", "333")]
    closes4 = [rows[s]["close_open_acc"]["close_d4"] for s in rows]
    summary = {
        "close_d3_dist": c3,
        "basin_close_d3_ge_678": sum(1 for c in c3 if c >= 0.678) / 3,
        "basin_close_d3_ge_85": sum(1 for c in c3 if c >= 0.85) / 3,
        "close_d4_dist": closes4,
        "close_d3_mean": round(sum(c3) / 3, 4),
        "close_d3_sd": round(
            (sum((c - sum(c3) / 3) ** 2 for c in c3) / 3) ** 0.5, 4)}
    print(f"[STACKDCC2-big] SUMMARY "
          f"{json.dumps({k: v for k, v in summary.items()})}", flush=True)
    result = {"tag": "ARCH-VET-LM-P17",
              "protocol": "dyck basin at 2x budget: STACKDCC2-big x "
                          "seeds 111/222/333, ctor under manual_seed, "
                          "P11 single-task stochastic dyck (train "
                          "depth 2, L=256 pool 256 seed 12345), "
                          "4000 STEPS (2x P14's 2000); eval exact "
                          "d3-8 + close/open d3/4/6/8 (16 streams); "
                          "bars close_d3>=.678 (TFMicro P12 control) "
                          "/ >=.85 (P11 citation); P14 2000-step "
                          "reference .655/.763/.631 (1/3)",
              "arms": {"STACKDCC2-big": {"per_seed": rows,
                                          "summary": summary}}}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
