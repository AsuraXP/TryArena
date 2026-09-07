# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 15 (cycle 57) — BASIN RATE AT 2.5x STRUCTURE ON
THE FULL CERTIFIED PROTOCOL. P14 showed the dyck-axis basin rate at
big config is 1/3 (STACKDCC2-big close_d3 .655/.763/.631; the C55
.925 citation = lucky basin) → L-BASIN-SCALE-CAPTURE does NOT
extend to dyck. The OTHER certified strongest-config claims
(handover C56): VETDCC-big seed-0 pair_eval .962, modk 1.0/1.0,
ce_1024 1.263 (ratio .5), track .62-.70 — remain n=1. P15 asks the
same basin question as P14 but on the FULL 4-task protocol: seeds
111/222/333 × {VETDCC-big, STACKDCC2-big} (ctor under per-seed
manual_seed, P6/P14 pattern — SEED HYGIENE LAW), P9 protocol
(4-task stream, L=256 pool 512 seed 12345, 2000 steps seed-free
train (train_arm uses the pool + Adam seed from global rng after
ctor), P9 evals). Bars: pair_eval >= .9 (P9 .962 regime), modk
eval = 1.0 (perfect), len_ratio_1024_over_256hard <= .6
(invariance), track-eval >= .5. Reports basin rate per bar per arm
+ mean/sd pair_eval.

PRIOR SAMPLES (seed-0, n=1): VETDCC-big pair .9623 modk 1.0/1.0
ce_1024 1.263 ratio .5 (P9, log.jsonl); VETbig pair .717 (P5);
STACKDCC2-big = VETDCC-big + stack (pair not separately measured —
P15's STACKDCC2 arm is the first full-protocol sample of the
strongest dyck config outside the dyck single-task protocol).

No new mechanism — pure multi-init measurement (protocol as P6/P9,
rationale as P14). Prior art: multi-init robustness reporting is
standard at micro scale (Henderson et al. 2018 RL reproducibility:
seed distributions over point estimates).

FALSIFIABLE PREDICTIONS:
  (a) VETDCC-big basin@.9 >= 2/3 AND modk 1.0 in >= 2/3 seeds →
      L-BASIN-SCALE-CAPTURE holds on the pair/counting axes
      (scale really captures the basin for these tasks);
      < 2/3 → downgrade to "seed-luck at n=1" for .962 too.
  (b) STACKDCC2-big basin@.9 <= VETDCC-big's (the stack organ
      perturbs the pair basin — or ties; the P13D finding that the
      stack is a modest enhancer suggests a tie is plausible).
  (c) len_ratio <= .6 in ALL 6 runs (length invariance is the
      architecture's certified core property — should be init-
      independent); any run > .6 flags an invariance basin.
Tag ARCH-VET-LM-P15.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

# VETDCC + full-protocol helpers from p9
_src9 = open("arch_vet_p9.py", encoding="utf-8").read()
_ns9 = {}
exec(compile(_src9.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p9.py", "exec"), _ns9)
V = _ns9["V"]; VETDCC = _ns9["VETDCC"]
make_pool = _ns9["make_pool"]; make_batches = _ns9["make_batches"]
train_arm = _ns9["train_arm"]; val_ce = _ns9["val_ce"]
task_acc = _ns9["task_acc"]; n_params = _ns9["n_params"]
dyck_acc = _ns9["dyck_acc"]

# STACKDCC2 from p11 (same VETDCC base + exact type stack)
_src11 = open("arch_vet_p11.py", encoding="utf-8").read()
_ns11 = {}
exec(compile(_src11.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p11.py", "exec"), _ns11)
STACKDCC2 = _ns11["STACKDCC2"]
assert _ns11["V"] == V


@torch.no_grad()
def eval_full(m, name, seed):
    """P9 eval_all: CE sweep + task_acc train/eval + dyck frontier."""
    m.eval()
    rng = random.Random(999);  vtr = make_batches(8, 256, rng)
    rh = random.Random(777);  vha = make_batches(8, 256, rh, hard=True)
    r5 = random.Random(31337); v512 = make_batches(2, 512, r5)
    r10 = random.Random(31415); v1024 = make_batches(2, 1024, r10)
    ce = {"256_train": val_ce(m, vtr, 256),
          "256_hard": val_ce(m, vha, 256),
          "512": val_ce(m, v512, 512),
          "1024": val_ce(m, v1024, 1024)}
    at = task_acc(m, 24, 256, random.Random(555), hard=False)
    ae = task_acc(m, 24, 256, random.Random(666), hard=True)
    dy = {}
    for d in (3, 4, 5, 6):
        a, tot = dyck_acc(m, 16, d)
        dy[f"depth{d}"] = round(a, 4)
    print(f"[{name} s{seed}] ce={ {k: round(v, 3) for k, v in ce.items()} }",
          flush=True)
    print(f"[{name} s{seed}] acc_eval="
          f"{ {k: round(v, 3) for k, v in ae.items()} }", flush=True)
    return {"ce": {k: round(v, 4) for k, v in ce.items()},
            "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
            "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()},
            "dyck_frontier_d3_6": dy,
            "pair_eval": round(ae["pair"], 4),
            "modk_eval": round(ae["modk"], 4),
            "track_eval": round(ae["track"], 4),
            "len_ratio_1024_over_256hard":
                round(ce["1024"] / max(1e-9, ce["256_hard"]), 3)}


def summarize(rows, name):
    pairs = [r["pair_eval"] for r in rows]
    mods = [r["modk_eval"] for r in rows]
    ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
    tracks = [r["track_eval"] for r in rows]
    summary = {
        "pair_eval_dist": pairs,
        "basin_pair_ge_9": sum(1 for p in pairs if p >= 0.9) / 3,
        "basin_pair_ge_717": sum(1 for p in pairs if p >= 0.717) / 3,
        "modk_eval_dist": mods,
        "basin_modk_eq_1": sum(1 for m2 in mods if m2 == 1.0) / 3,
        "len_ratio_dist": ratios,
        "basin_ratio_le_6": sum(1 for r2 in ratios if r2 <= 0.6) / 3,
        "track_eval_dist": tracks,
        "pair_mean": round(sum(pairs) / 3, 4),
        "pair_sd": round((sum((p - sum(pairs) / 3) ** 2 for p in pairs)
                          / 3) ** 0.5, 4)}
    print(f"[{name}] SUMMARY "
          f"{json.dumps({k: v for k, v in summary.items()})}", flush=True)
    return summary


if __name__ == "__main__":
    pool = make_pool(512, 256, 12345)
    result = {"tag": "ARCH-VET-LM-P15",
              "protocol": "basin rate at 2.5x structure on the FULL "
                          "P9 4-task protocol; seeds (111,222,333) "
                          "manual_seed BEFORE ctor; arms VETDCC-big "
                          "(21,257p) / STACKDCC2-big (21,817p); "
                          "L=256 pool 512 seed 12345, 2000 steps; "
                          "evals: CE 256_train/256_hard/512/1024 + "
                          "task_acc train/eval + dyck d3-6; bars "
                          "pair>=.9 (P9 .962 regime), modk=1.0, "
                          "ratio<=.6, track>=.5; prior n=1 seed-0 "
                          "samples P9 pair .9623 ratio .5 / P5 pair "
                          ".717 (VETbig) noted",
              "arms": {}}
    for arm_name, ctor in [
            ("VETDCC-big", lambda: VETDCC(V, 24, k=8, K=8)),
            ("STACKDCC2-big", lambda: STACKDCC2(V, 24, k=8, K=8))]:
        rows = []
        for seed in (111, 222, 333):
            torch.manual_seed(seed)
            m = ctor()
            print(f"[{arm_name} seed {seed}] params={n_params(m)}",
                  flush=True)
            hist = train_arm(f"{arm_name}-s{seed}", m, pool, 2000, 8)
            row = eval_full(m, arm_name, seed)
            row["loss_curve"] = hist
            rows.append(row)
            result.setdefault("per_seed", {})
        result["arms"][arm_name] = {
            "per_seed": {str(s): r for s, r in zip((111, 222, 333), rows)},
            "summary": summarize([r for r in rows], arm_name)}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
