# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 12 (cycle 55) — THE TRANSFORMER CONTROL for
the dyck axis. P11 showed STACKDCC2-big close-type .925/.852 at
d3/d4 (single-task dyck, train depth 2) but had NO transformer
arm — the "beat TF on dyck generalization" claim was incomplete.
P12 runs TFMicro (2L d16 2h pre-LN sinusoidal-PE, 8,144p — the
EXACT P1 control class) on the identical single-task dyck
protocol + re-runs STACKDCC2-big for same-session comparability.
Prior art: Hahn 2020 (Transformers fail Dyck-2 ASYMPTOTICALLY —
depth-generalization failure beyond train depths; the failure may
not be visible at d3-4 micro-scale — this run tests exactly that
edge); P1 (TF dyck-ev .019 in the mixed 4-task stream at 8,144p).
Protocol: single-task stochastic dyck, train depth 2, L=256,
pool 256 (seed 12345), 2000 steps, seed 0 — identical to P11.
Eval: per-position open/close bracket acc at d3-10 + exact-match
d3-10 (16 streams; ceiling = 0.0 for all arms, P11).
SHARP PREDICTIONS:
  (a) STACKDCC2-big close d3-4 (.925/.852) > TFMicro close d3-4
      -> first clean structural win over attention on dyck
      generalization at matched-ish params (21.8k vs 8.1k);
  (b) TFMicro >= STACKDCC2 at d3-4 -> L-DYCK-TF-PARITY-AT-MICRO:
      micro-attention handles d3-4 fine; the Hahn-style failure
      needs deeper eval depths (P13: frontier d12-16).
Tag ARCH-VET-LM-P12.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.manual_seed(0)
T0 = time.time()

_src11 = open("arch_vet_p11.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src11.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p11.py", "exec"), _ns)
V = _ns["V"]; VETLM = _ns["VETLM"]
STACKDCC2 = _ns["STACKDCC2"]
gen_dyck_pool = _ns["gen_dyck_pool"]; train_dyck = _ns["train_dyck"]
bracket_pos_acc = _ns["bracket_pos_acc"]; dyck_acc = _ns["dyck_acc"]
n_params = _ns["n_params"]
# TFMicro lives in arch_vet_lm:
_src_lm = open("arch_vet_lm.py", encoding="utf-8").read()
_ns_lm = {}
exec(compile(_src_lm.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns_lm)
TFMicro = _ns_lm["TFMicro"]


if __name__ == "__main__":
    pool = gen_dyck_pool(256, 256, 12345)
    arms = [("TFMicro", TFMicro(V, 16)),
            ("STACKDCC2-big", STACKDCC2(V, 24, k=8, K=8))]
    result = {"tag": "ARCH-VET-LM-P12",
              "protocol": "single-task stochastic dyck, train depth 2, "
                          "L=256 pool 256 seed 12345, 2000 steps seed 0 "
                          "(identical to P11); per-position open/close "
                          "acc + exact-match d3-10 (16 streams). "
                          "Closes P11's missing TF control; "
                          "STACKDCC2-big re-run for same-session "
                          "comparability (P11 values cited: close "
                          "d3-8 .925/.852/.724/.595; VETbase "
                          ".657/.588/.520/.481)",
              "arms": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_dyck(name, m, pool, 2000, 8)
        m.eval()
        dy = {}
        for d in (3, 4, 5, 6, 7, 8, 9, 10):
            a, tot = dyck_acc(m, 16, d)
            dy[f"exact_d{d}"] = round(a, 4)
        bp = {}
        for d in (3, 4, 6, 8, 10):
            o, cl, to, tc = bracket_pos_acc(m, 16, d)
            bp[f"open_d{d}"] = o
            bp[f"close_d{d}"] = cl
        print(f"[{name}] exact: { {k: v for k, v in dy.items()} }",
              flush=True)
        print(f"[{name}] pos: { {k: v for k, v in bp.items()} }",
              flush=True)
        result["arms"][name] = {"params": n_params(m),
                                "loss_curve": hist,
                                "exact_match_frontier": dy,
                                "per_position_bracket_acc": bp}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
