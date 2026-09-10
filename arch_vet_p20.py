# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 20 (cycle 60) — SCHEDULED-FUSION PILOT:
PER-FAMILY TOKEN BUDGETS IN ONE STREAM. C58/C59 closed queue #1
with the FUSION LAW (the 2x2, P16/P19/P19B/P19c): under NAIVE
equal-rate mixing NO arm x corpus config holds all three certified
bars —
    dyck d12>=.85 needs the depth-diverse dyck corpus: 3/3 either
      arm (STACKDCC2-big mean .957; VETDCC-big mean .881);
    pair >=.717 needs vanilla exemplar density + VETDCC-big: 3/3
      only VETDCC-big x vanilla (P16 .767); 0/3 VETDCC-big x mixdd
      (L-MIXDD-STARVES-PAIR);
    CE ratio <=.6 needs the vanilla stream: 3/3 both arms on
      vanilla, 0/3 both arms on mixdd (L-MIXDD-BREAKS-RATIO-ARM-
      INDEPENDENT).
P20 = the fusion-law payoff: SCHEDULE the per-family token budget
instead of equal-rate mixing. KEY DESIGN CONSTRAINT found during
construction: at L=256 a d6 segment (255 tokens) cannot coexist
with ANY other family at a healthy token share, and a d5 (127) needs
~50% of the stream — the mixdd corpus's streams that were 50-100%
dyck are exactly what starved pair (L-MIXDD-STARVES-PAIR). So P20
caps the DYCK TOKEN BUDGET at 50% of L: every stream is
GUARANTEED >=50% non-dyck tokens (vanilla P9 intervals for track/
modk/pair, so their exemplar density can never collapse to zero as
in the all-dyck mixdd streams), while depth-diverse dyck (pick_depth
2..6, same grammar) is still emitted up to that budget — a d5
(127t) fits exactly at the 50% cap; d6 (255t) is excluded at L=256
(flagged honestly: the depth-diverse ceiling of this L=256 schedule
is d5; a d6-inclusive schedule requires L>=1024 and is the C61
follow-up if the d5 ceiling under-delivers). This isolates the
FUSION MECHANISM (dyck-present-under-budget) from the STARVATION
(uncontrolled dyck dominance) at the certified L=256 budget.
Prior art (searched 2026-09-08): ADAPT arXiv 2512.04555 (learned
task-mixture proportions under explicit token budgets beats uniform
and size-proportional mixing in multi-task instruction tuning —
micro-scale proxy: our fixed per-family budget is the static SFT-P
baseline, and its success here is the precondition for any learned
allocation); Guo et al. ECCV 2018 (dynamic task prioritization by
difficulty); Lin et al. ICLR 2022 RLW (random loss weighting =
comparable to SOTA without tuning — a cheap random-budget control
if the fixed budget fails). Program lineage: P19/P19B/P19c (the
2x2, this header), P18/P13D (depth-diverse dyck single-task),
P16 (VETDCC-big pair @4000), P15 (4-task @2000).
ARM x seeds 111/222/333 (ctor under per-seed manual_seed, pool 512
L=256 seed 12345, 4000 steps batch 8 — all as P19-family):
  VETDCC-big x S-budget  (the pair-carrying arm: does the budget
    restore pair >=.717 AND keep dyck 3/3 AND ratio <=.6?)
EVAL per seed = eval_full (task_acc train/eval + CE sweep + ratio)
+ eval_dyck (rt_close_acc d1-12) — identical to P19/P19b/P19c.
TARGET (falsifiable, the whole-model certification): on the
VETDCC-big x S-budget arm, ALL THREE bars in >=2/3 seeds —
dyck close d12 >=.85, pair eval >=.717, CE ratio <=.6 (modk 1.0
expected 3/3). If dyck 3/3 + pair 3/3 + ratio 3/3 on VETDCC-big:
L-SCHEDULED-FUSION-CAPTURES-ALL — the first whole-model multi-
family multi-seed certification, and the multi-seed certified
table gains one unified row (counting + dyck + pair + invariance
in a SINGLE config). If pair still <=1/3 with dyck+ratio 3/3:
L-BUDGET-CAPTURES-DYCK-RATIO-NOT-PAIR (pair needs more exemplars
or a different mechanism); if ratio still 0/3: the deep dyck's
long-range CE load cannot be budgeted away at L=256 → test larger
L in C61.
Tag ARCH-VET-LM-P20.
"""
import importlib
import json, os, random, time
import torch

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

import arch_vet_p19 as p19
from arch_vet_p13d import VETDCC
STACKDCC2_D12 = p19.STACKDCC2_D12


def gen_sched_stream(rng, L=256, dyck_share=0.5):
    """P9 4-task stream with a per-stream DYCK TOKEN BUDGET.

    Non-dyck families keep the vanilla P9 train intervals verbatim
    (track gap 4-16, modk count 2-12, pair gap 4-12). The dyck
    family draws pick_depth() 2..6 (depth-diverse grammar) but only
    emits if the segment fits the remaining room AND the stream's
    cumulative dyck-token budget (cap ~= dyck_share * L). If it
    does not fit the budget, retry another family (up to 8 tries)
    so the budget is a hard cap, not a soft pressure.
    """
    dyck_toks_used = 0
    dyck_budget = int(dyck_share * L)
    x = [p19.BOS]
    while len(x) < L:
        room = L - len(x)
        cand = None
        for _attempt in range(8):
            task = rng.randrange(4)
            if task == 0:                 # TRACK: T x <gap> A x
                sym = rng.randrange(8) + p19.TRACK
                gap = min(rng.randrange(4, 17), max(0, room - 4))
                cand = ([p19.T_TASK, sym]
                        + [p19.fill_tok(rng) for _ in range(gap)]
                        + [p19.A, sym])
            elif task == 1:               # MODK: T T 1*1 ... A m
                n = min(rng.randrange(2, 13), max(0, room - 4))
                if n < 2:
                    continue
                cand = ([p19.T_TASK, p19.T_TASK] + [p19.ONE] * n
                        + [p19.A, p19.MANS + (n % 3)])
            elif task == 2:               # DYCK depth-diverse but
                # budget-capped: only emit if the segment's tokens
                # fit the remaining dyck budget
                for _dd in range(6):
                    d = p19.pick_depth(rng)
                    seg = p19.rt_segment(rng, d)
                    if (3 + len(seg) <= room and
                            3 + len(seg) <= dyck_budget - dyck_toks_used):
                        cand = ([p19.T_TASK, p19.T_TASK, p19.T_TASK]
                                + seg)
                        dyck_toks_used += 3 + len(seg)
                        break
                if cand is None:
                    continue            # over budget -> try another family
            else:                         # PAIR: T T k v <gap> A k v
                i = rng.randrange(4)
                j = rng.randrange(4)
                gap = min(rng.randrange(4, 13), max(0, room - 8))
                cand = ([p19.T_TASK, p19.T_TASK, p19.KEYS + i,
                         p19.VALS + j]
                        + [p19.fill_tok(rng) for _ in range(gap)]
                        + [p19.A, p19.KEYS + i, p19.VALS + j])
            if cand is not None and len(cand) <= room:
                break
            cand = None
        if cand is None:                  # filler pad (rare)
            x += [p19.fill_tok(rng)] * room
            break
        x += cand
    x = x[:L]
    x.append(p19.EOS)
    assert len(x) == L + 1, len(x)
    return x


def gen_sched_pool(n, L, seed, dyck_share=0.5):
    prng = random.Random(seed)
    return [torch.tensor(gen_sched_stream(prng, L, dyck_share))
            for _ in range(n)]


if __name__ == "__main__":
    pool = gen_sched_pool(512, 256, 12345, dyck_share=0.5)
    # composition audit: bracket-token share (dyck-body proxy) and
    # max-depth coverage
    from collections import Counter
    br_shares = []
    mx_by_stream = []
    for p in pool:
        toks = p.tolist()
        n_br = sum(1 for t in toks if p19.BRK <= t < p19.BRK + 4)
        br_shares.append(n_br / len(toks))
        cur = 0; mx = 0
        for t in toks:
            if p19.BRK <= t < p19.BRK + 2:
                cur += 1
            elif p19.BRK + 2 <= t < p19.BRK + 4:
                cur -= 1
            mx = max(mx, cur)
        mx_by_stream.append(mx)
    dh = Counter(mx_by_stream)
    print(f"[P20] sched-pool max-depth histogram: {dict(dh)}", flush=True)
    print(f"[P20] sched-pool bracket-token share: "
          f"mean {sum(br_shares)/len(br_shares):.3f} "
          f"max {max(br_shares):.3f}", flush=True)
    assert any(d >= 4 for d in mx_by_stream), "no deep dyck in sched pool"

    def run_arm(name, ctor, pool_):
        rows = []
        for seed in (111, 222, 333):
            torch.manual_seed(seed)
            m = ctor()
            print(f"[{name} seed {seed}] params={p19.n_params(m)}",
                  flush=True)
            hist = p19.train_arm(f"{name}-s{seed}", m, pool_, 4000, 8)
            row = p19.eval_full(m, name, seed)
            row["loss_curve"] = hist
            row["dyck_close"] = p19.eval_dyck(m, name, seed)
            rows.append(row)
        d12 = [r["dyck_close"]["close_d12"] for r in rows]
        d6 = [r["dyck_close"]["close_d6"] for r in rows]
        mods = [r["acc_eval_interval"]["modk"] for r in rows]
        pairs = [r["acc_eval_interval"]["pair"] for r in rows]
        ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
        summary = {
            "dyck_close_d6_dist": d6,
            "dyck_close_d12_dist": d12,
            "basin_dyck_d12_ge_85":
                sum(1 for c in d12 if c >= 0.85) / 3,
            "modk_eval_dist": mods,
            "basin_modk_eq_1": sum(1 for m2 in mods if m2 == 1.0) / 3,
            "pair_eval_dist": pairs,
            "basin_pair_ge_717": sum(1 for p2 in pairs if p2 >= 0.717)
                / 3,
            "len_ratio_dist": ratios,
            "basin_ratio_le_6": sum(1 for r2 in ratios if r2 <= 0.6)
                / 3,
            "dyck_close_d12_mean": round(sum(d12) / 3, 4)}
        print(f"[{name}] SUMMARY {json.dumps(summary)}", flush=True)
        return {"per_seed": {str(s): r for s, r in
                             zip((111, 222, 333), rows)},
                "summary": summary}

    result = {"tag": "ARCH-VET-LM-P20",
              "protocol": "scheduled-fusion pilot: P9 4-task stream "
                          "with a per-stream DYCK TOKEN BUDGET cap "
                          "= 50% of L (depth-diverse pick_depth "
                          "2..6 segments emitted only if they fit "
                          "remaining room AND the remaining dyck "
                          "budget; every stream is >=50% non-dyck "
                          "tokens with vanilla P9 train intervals; "
                          "L=256 ceiling = d5 (127t), d6 (255t) "
                          "needs L>=1024 — C61 follow-up if the d5 "
                          "ceiling under-delivers); arm VETDCC-big "
                          "x seeds 111/222/333, ctor under "
                          "manual_seed, pool 512 L=256 seed 12345, "
                          "4000 steps batch 8; eval = eval_full + "
                          "eval_dyck. Target: dyck d12>=.85 + "
                          "pair>=.717 + ratio<=.6 all >=2/3 "
                          "(whole-model certification). References: "
                          "vanilla VETDCC-big pair 3/3 .767 ratio "
                          "3/3 (P16); mixdd VETDCC-big dyck 3/3 "
                          ".881 pair 0/3 ratio 0/3 (P19c); "
                          "mixdd STACKDCC2-big dyck 3/3 .957 (P19).",
              "arms": {}}
    result["arms"]["VETDCC-big-sched"] = run_arm(
        "VETDCC-big-sched", lambda: VETDCC(p19.V, 24, k=8, K=8), pool)
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
