# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 19 (cycle 58) — MIXED-STREAM FUSION: DEPTH-
DIVERSE DYCK IN THE 4-TASK STREAM. P18 certified the depth-diverse
dyck win SINGLE-TASK (close d12 .984/.950/.973, 3/3 basin, mean
.969). The open question (C57-close queue, #1): does that corpus
COEXIST with pair/modk/track in the P9 mixed 4-task stream, or was
the win an artifact of the single-task protocol (the mirror image
of the banked L-DYCK-BUDGET-STARVED: dyck in the vanilla mixed
stream is shallow and starved)? P19 = data-level fusion test:
train the SAME arm (STACKDCC2-big D12, 21,817p — the P18 dyck-
certified config) on a mixed stream whose dyck family emits
EXACT-SHAPE RANDOM-TYPE segments at depth pick_depth() (2..6,
weighted d-1, identical to P13D/P18) inside the otherwise-vanilla
P9 4-task stream (track/modk/pair intervals unchanged).

PROTOCOL: 3 seeds (111/222/333), ctor under per-seed manual_seed
(SEED HYGIENE LAW), pool 512 L=256 seed 12345 (P9) of custom mixed
streams, 4000 steps (the 4-task certified budget of P16), batch 8.
EVAL per seed:
  (1) dyck: rt_close_acc d1..d12 on exact-shape random-type fixed-
      depth streams (P18 metric) — per-depth L, adaptive streams;
  (2) 4-task: task_acc train/eval (hard intervals) + CE sweep →
      track/modk/dyck/pair eval acc + length-invariance ratio
      (P9/P15/P16 metrics).
REFERENCE POINTS: dyck close d12 P18 single-task .984/.950/.973
(3/3 >= .85); modk 1.0 9/9 (P15+P16), pair >=.717 3/3 @4000 (P16
VETDCC-big — arm differs, noted), ratio <= .6 (P16 3/3).

Prior art (searched 2026-09-07): P18 L-DEPTH-DIVERSITY-CAPTURES-
DYCK-BASIN (single-task); this program's banked L-DYCK-BUDGET-
STARVED (shallow dyck in mixed stream starves at 2000 steps);
Zhou et al. 2023 arXiv 2310.13349 (depth-diverse training induces
recursive structure in transformers); Zhou et al. 2024 arXiv
2402.09371 (init-fragility of OOD generalization). Data-level
task-fusion at micro scale is the pre-fusion question for the
VET-LM+corpus fusion pilot (queue #4): if the exact-shape dyck
corpus breaks the other families, the fusion route must budget/
schedule per family rather than naively mix.

FALSIFIABLE PREDICTIONS:
  (a) dyck survives fusion: close d12 >= .85 in >= 2/3 seeds →
      L-MIXED-DEPTH-DIVERSE-DYCK-OK (the depth-diverse dyck win is
      single-task-independent; data-level fusion compatible);
      < 2/3 → the mixed stream under-trains the deep segments
      (L-MIXED-DYCK-DILUTED) — fusion needs per-family budgets.
  (b) 4-task properties hold: modk eval = 1.0 in >= 2/3 seeds AND
      pair eval >= .717 in >= 2/3 AND ratio <= .6 in >= 2/3 →
      the depth-diverse dyck corpus does not break the certified
      counting/pair/invariance properties (fusion clean).
      Violation of pair/ratio → the deep segments starve/rebalance
      the shallow families (budget-capture undone by the corpus).
Tag ARCH-VET-LM-P19.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

# LM prelude: constants (V, tokens, fill_tok) + the P9/P16 4-task
# machinery (make_batches/train_arm/val_ce/task_acc/n_params) from
# the canonical arch_vet_lm.py (same exec pattern as P9-P18).
_src = open("arch_vet_lm.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns)
V = _ns["V"]; BOS = _ns["BOS"]; EOS = _ns["EOS"]
T_TASK = _ns["T_TASK"]; A = _ns["A"]; ONE = _ns["ONE"]
TRACK = _ns["TRACK"]; MANS = _ns["MANS"]
KEYS = _ns["KEYS"]; VALS = _ns["VALS"]; MODS = _ns["MODS"]
BRK = _ns["BRK"]
fill_tok = _ns["fill_tok"]
make_pool = _ns["make_pool"]; make_batches = _ns["make_batches"]
train_arm = _ns["train_arm"]; val_ce = _ns["val_ce"]
task_acc = _ns["task_acc"]; n_params = _ns["n_params"]

# P13D prelude: the exact-shape random-type dyck grammar, the
# certified dyck arm STACKDCC2_D12, and rt_close_acc (P18 metric).
_src13d = open("arch_vet_p13d.py", encoding="utf-8").read()
_ns13 = {}
exec(compile(_src13d.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p13d.py", "exec"), _ns13)
STACKDCC2_D12 = _ns13["STACKDCC2_D12"]
rt_segment = _ns13["rt_segment"]; seg_len = _ns13["seg_len"]
rt_close_acc = _ns13["rt_close_acc"]; pick_depth = _ns13["pick_depth"]
assert (T_TASK, BRK) == (_ns13["T_TASK"], _ns13["BRK"])

L_by_depth = {d: (256 if seg_len(d) <= 252 else seg_len(d) + 16)
              for d in range(1, 13)}


def n_streams(d):
    L = L_by_depth[d]
    if L <= 256:
        return 8
    if L <= 4096:
        return 4
    return 2


# ---- custom mixed stream: P9 4-task + depth-diverse exact-shape
# random-type dyck (gen_stream with the dyck branch replaced) ------
def gen_mixdd_stream(rng, L=256):
    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        cand = None
        for _attempt in range(8):   # 8 tries (dyck deep fits rarely)
            task = rng.randrange(4)
            if task == 0:                 # TRACK: T x <gap> A x
                sym = rng.randrange(8) + TRACK
                gap = min(rng.randrange(4, 17),
                          max(0, room - 4))
                cand = ([T_TASK, sym]
                        + [fill_tok(rng) for _ in range(gap)]
                        + [A, sym])
            elif task == 1:               # MODK: T T 1*1 ... A m
                n = min(rng.randrange(2, 13), max(0, room - 4))
                if n < 2:
                    continue
                cand = ([T_TASK, T_TASK] + [ONE] * n
                        + [A, MANS + (n % 3)])
            elif task == 2:               # DYCK depth-diverse exact-
                # shape random-type (P13D/P18 grammar, train depths
                # 2..6 weighted deep)
                for _dd in range(6):
                    d = pick_depth(rng)
                    seg = rt_segment(rng, d)
                    if 3 + len(seg) <= room:
                        cand = [T_TASK, T_TASK, T_TASK] + seg
                        break
            else:                         # PAIR: T T k v <gap> A k v
                i = rng.randrange(4)
                j = rng.randrange(4)
                gap = min(rng.randrange(4, 13),
                          max(0, room - 8))
                cand = ([T_TASK, T_TASK, KEYS + i, VALS + j]
                        + [fill_tok(rng) for _ in range(gap)]
                        + [A, KEYS + i, VALS + j])
            if cand is not None and len(cand) <= room:
                break
            cand = None
        if cand is None:
            x += [fill_tok(rng)] * room
            break
        x += cand
    x = x[:L]
    x.append(EOS)
    assert len(x) == L + 1, len(x)
    return x


def gen_mixdd_pool(n, L, seed):
    prng = random.Random(seed)
    return [torch.tensor(gen_mixdd_stream(prng, L)) for _ in range(n)]


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
            "len_ratio_1024_over_256hard":
                round(ce["1024"] / max(1e-9, ce["256_hard"]), 3)}


def eval_dyck(m, name, seed):
    cl = {}
    for d in range(1, 13):
        L = L_by_depth[d]
        a, tot = rt_close_acc(m, n_streams(d), d, L)
        cl[f"close_d{d}"] = a
    print(f"[{name} s{seed}] dyck-close: "
          f"{ {k: v for k, v in cl.items()} }", flush=True)
    return cl


if __name__ == "__main__":
    pool = gen_mixdd_pool(512, 256, 12345)
    # sanity: depth coverage of the mixed corpus
    from collections import Counter
    mx = []
    for p in pool:
        toks = p.tolist()
        dep = cur = 0
        for t in toks:
            if BRK <= t < BRK + 2:
                cur += 1
            elif BRK + 2 <= t < BRK + 4:
                cur -= 1
            dep = max(dep, cur)
        mx.append(dep)
    hist = Counter(mx)
    print(f"[P19] mixed pool depth histogram: {dict(hist)}", flush=True)
    assert any(mx[d] >= 4 for d in range(len(mx))), "no deep segs"
    result = {"tag": "ARCH-VET-LM-P19",
              "protocol": "mixed-stream fusion: P9 4-task stream "
                          "with the dyck family emitting exact-shape "
                          "random-type segments at depth 2..6 "
                          "weighted d-1 (P13D/P18 grammar) inside "
                          "the vanilla track/modk/pair stream; "
                          "STACKDCC2-big D12 x seeds 111/222/333, "
                          "ctor under manual_seed, pool 512 L=256 "
                          "seed 12345, 4000 steps batch 8; eval "
                          "dyck rt_close_acc d1-12 (P18 metric) + "
                          "task_acc train/eval + CE sweep; "
                          "references: P18 dyck d12 .984/.950/.973, "
                          "P16 modk 1.0 3/3 / pair>=.717 3/3 / "
                          "ratio<=.6 3/3",
              "arms": {}}
    rows = []
    for seed in (111, 222, 333):
        torch.manual_seed(seed)
        m = STACKDCC2_D12(V, 24, k=8, K=8)
        print(f"[STACKDCC2-big-D12 seed {seed}] params={n_params(m)}",
              flush=True)
        hist_loss = train_arm(f"D12-mix-s{seed}", m, pool, 4000, 8)
        row = eval_full(m, "D12-mix", seed)
        row["loss_curve"] = hist_loss
        row["dyck_close"] = eval_dyck(m, "D12-mix", seed)
        rows.append(row)
    d12 = [r["dyck_close"]["close_d12"] for r in rows]
    d6 = [r["dyck_close"]["close_d6"] for r in rows]
    mods = [r["acc_eval_interval"]["modk"] for r in rows]
    pairs = [r["acc_eval_interval"]["pair"] for r in rows]
    ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
    summary = {
        "dyck_close_d6_dist": d6,
        "dyck_close_d12_dist": d12,
        "basin_dyck_d12_ge_85": sum(1 for c in d12 if c >= 0.85) / 3,
        "modk_eval_dist": mods,
        "basin_modk_eq_1": sum(1 for m2 in mods if m2 == 1.0) / 3,
        "pair_eval_dist": pairs,
        "basin_pair_ge_717": sum(1 for p in pairs if p >= 0.717) / 3,
        "len_ratio_dist": ratios,
        "basin_ratio_le_6": sum(1 for r2 in ratios if r2 <= 0.6) / 3,
        "dyck_close_d12_mean": round(sum(d12) / 3, 4)}
    print(f"[STACKDCC2-big-D12] SUMMARY "
          f"{json.dumps({k: v for k, v in summary.items()})}", flush=True)
    result["arms"]["STACKDCC2-big-D12"] = {
        "per_seed": {str(s): r for s, r in zip((111, 222, 333), rows)},
        "summary": summary}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
