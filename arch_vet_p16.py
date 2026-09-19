# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 16 (cycle 57) — PAIR-OOD BASIN: BUDGET VS
STRUCTURE. P15: pair_eval (key-value binding recall across gap
24-48, train 4-12) basin rate = 1/3 at 2000 steps / 2.5x structure
(VETDCC-big .302/.302/.887; STACKDCC2-big .057/.208/.925) while
train pair acc = 1.0 in ALL runs — the lottery is the OOD retention
gap, not learning. P5's single .717 (VETbig, 4000 steps) vs P9's
.962 (VETDCC-big, 2000 steps) vs P15's .30-ish (2000 steps, 3
seeds) leaves budget confounded. Prior art (searched 2026-09-07):
Zhou et al. arXiv 2402.09371 — OOD/length generalization is FRAGILE
to weight init for standard transformers too (universal at this
scale); Li et al. 2025 arXiv 2504.02827 — dictionary-lookup OOD has
a vanishing-variance mechanism (attention); Arora et al. ICLR 2024
"Measuring Recall in Efficient Architectures" — associative recall
needs width ~ length for gated convs (attention near-constant).
Frame: if P16 finds basin@.717 >= 2/3 at 4000 steps, the pair
lottery is a BUDGET effect (fixable by more training — the VET
register can learn the longer-gap routing); if still ~1/3, it is
STRUCTURAL (the soft register's write/read routing over a longer
filler span is not discoverable from seed — needs an exact
associative channel: the C12 SRAM organ line, or the interval
curriculum).

PROTOCOL: VETDCC-big (21,257p) x seeds (111,222,333), ctor under
manual_seed (SEED HYGIENE LAW), P9 4-task protocol, 4000 steps (2x
P15's 2000 — the P5 budget that produced .717), L=256 pool 512 seed
12345; eval exactly P15 (CE 256_train/256_hard/512/1024 + task_acc
train/eval + dyck d3-6). Bars (same as P15 for direct comparison):
pair_eval >= .9 / >= .717; modk = 1.0; ratio <= .6; track >= .5.
P15 2000-step reference: pair .302/.302/.887, modk 1.0 x3, ratio
.533/.585/.413.

FALSIFIABLE PREDICTIONS:
  (a) budget effect: basin@.717 >= 2/3 AND mean pair_eval >= .65 at
      4000 steps (the longer budget lets every init discover the
      long-gap routing) → P17 = push the gap further (48-96) to
      re-find the frontier; L-PAIR-LOTTERY-BUDGET.
  (b) structural: basin@.717 stays ~1/3 (mean ~.5) → the pair-OOD
      routing is init-structural; next attack = exact associative
      channel (SRAM organ, C12 line) or gap curriculum — L-PAIR-
      LOTTERY-STRUCTURAL.
  Either outcome is a falsifiable sharp test of why the flagship
  "reasoning/generalization" axis is a lottery.
Tag ARCH-VET-LM-P16.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

_src9 = open("arch_vet_p9.py", encoding="utf-8").read()
_ns9 = {}
exec(compile(_src9.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p9.py", "exec"), _ns9)
V = _ns9["V"]; VETDCC = _ns9["VETDCC"]
make_pool = _ns9["make_pool"]; make_batches = _ns9["make_batches"]
train_arm = _ns9["train_arm"]; val_ce = _ns9["val_ce"]
task_acc = _ns9["task_acc"]; n_params = _ns9["n_params"]
dyck_acc = _ns9["dyck_acc"]


@torch.no_grad()
def eval_full(m, name, seed):
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
    print(f"[{name} s{seed}] ce={ {k: round(v, 3) for k, v in ce.items()} }",
          flush=True)
    print(f"[{name} s{seed}] acc_eval="
          f"{ {k: round(v, 3) for k, v in ae.items()} }", flush=True)
    return {"ce": {k: round(v, 4) for k, v in ce.items()},
            "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
            "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()},
            "pair_eval": round(ae["pair"], 4),
            "modk_eval": round(ae["modk"], 4),
            "track_eval": round(ae["track"], 4),
            "len_ratio_1024_over_256hard":
                round(ce["1024"] / max(1e-9, ce["256_hard"]), 3)}


if __name__ == "__main__":
    pool = make_pool(512, 256, 12345)
    result = {"tag": "ARCH-VET-LM-P16",
              "protocol": "pair-OOD basin budget test: VETDCC-big x "
                          "seeds 111/222/333, ctor under manual_seed, "
                          "P9 4-task L=256 pool 512 seed 12345, "
                          "4000 STEPS (2x P15's 2000 = P5 budget); "
                          "eval CE sweep + task_acc + dyck d3-6 "
                          "(P15-identical); bars pair>=.9/.717, "
                          "modk=1.0, ratio<=.6; P15 2000-step "
                          "reference .302/.302/.887",
              "arms": {}}
    rows = []
    for seed in (111, 222, 333):
        torch.manual_seed(seed)
        m = VETDCC(V, 24, k=8, K=8)
        print(f"[VETDCC-big seed {seed}] params={n_params(m)}", flush=True)
        hist = train_arm(f"VETDCC-big-s{seed}", m, pool, 4000, 8)
        row = eval_full(m, "VETDCC-big", seed)
        row["loss_curve"] = hist
        rows.append(row)
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
    print(f"[VETDCC-big] SUMMARY "
          f"{json.dumps({k: v for k, v in summary.items()})}", flush=True)
    result["arms"]["VETDCC-big"] = {
        "per_seed": {str(s): r for s, r in zip((111, 222, 333), rows)},
        "summary": summary}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
