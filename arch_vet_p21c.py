# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-P21C (cycle 61) — DEPTH-AWARE DISPATCH: CAN THE UNIFIED
ROW REACH 4/4 BARS 3/3 WITHOUT RETRAINING? P21 (modular fusion,
43,074p) certified pair 3/3, modk 3/3, dyck d12 3/3 in ONE system
and showed pair~1.0 WITH dyck .85-.99 in the same heterogeneous pass
(L-MODULAR-FUSION-BREAKS-THE-WALL), but the routed CE ratio
1024/256hard was 0/3 (1.058/1.238/0.884). P21B located the cause
(L-DEPTH-AWARE-DISPATCH): on the 1024/easy bank (dyck depth 2) expert
A is far better at dyck targets (CE 0.49-0.52 vs B 1.22-4.28) because
the d-1 depth curriculum gives depth 2 weight 1/15 in B's training,
while on the 256-hard bank (depth 3) B is better (2.80-3.33 vs
4.47-7.32). P21B's *non-causal* depth>=3 variant restored the ratio
(0.475/0.559/0.464, 3/3) — but it used the TARGET token's depth,
which a deployed router cannot see.
P21C = the CAUSALLY IMPLEMENTABLE version: route position t to B iff
the family is dyck (from tokens <=t) AND the bracket depth ENTERING
position t is >= T. Entering depth is known causally; for a closing
bracket the entering depth IS its segment depth (so closes of
depth>=3 segments go to B, matching P21B), and for an opening bracket
the entering depth is one below its post-push depth (flagged: opens
at exactly depth T-1->T differ from the P21B mask by one level).
Run T=3 (primary, the P21B-matching rule) and T=2 (secondary, "inside
a dyck segment of depth >=2" = all non-trivial dyck). NO RETRAINING
(existing p21_ckpt/ experts); eval-only, so the result is a pure
dispatch property.
Metric targets: unified row 4/4 bars — pair >=.717, modk 1.0,
ratio <=.6, dyck d12 >=.85, each basin 3/3 — plus the joint
heterogeneous pass (pair ~1.0 with dyck >=.85 in ONE forward pass)
must be preserved, otherwise the invariance patch has traded away
the fusion win.
Prior art: same as P21 (arXiv 2409.14981 task-specific specialization;
MoE task-level routing minimizes interference; Zhou et al. 2024
arXiv 2402.09371 OOD fragility) + the program's own P21B
decomposition. Tag ARCH-VET-LM-P21C.
"""
import json, os, random, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(REPO, "p21_ckpt")
T0 = time.time()

import torch
torch.set_num_threads(1)

import arch_vet_p21 as p21
import arch_vet_p19 as p19
import arch_vet_p13d as p13d

BRK = p19.BRK
T_TASK = p19.T_TASK
ONE = p19.ONE
KEYS = p19.KEYS
TRACK = p19.TRACK
A_MARK = p19.A


def make_deep_router(T):
    """Causal depth-threshold dispatch: B iff family==dyck (from
    tokens <= t) and entering bracket depth >= T."""
    def router(toks):
        n = len(toks)
        route = [0] * n
        fam = None
        depth = 0
        tt = 0
        for t in range(n):
            route[t] = 1 if (fam == "dyck" and depth >= T) else 0
            tok = toks[t]
            if tok == T_TASK:
                tt += 1
                if tt >= 3:
                    fam, depth = "dyck", 0
                continue
            if tt == 1:
                fam = "track"
            elif tt == 2:
                if tok == ONE:
                    fam = "modk"
                elif BRK <= tok < BRK + 4:
                    fam, depth = "dyck", 0
                elif KEYS <= tok < KEYS + 4:
                    fam = "pair"
                else:
                    fam = "other"
            elif tt >= 3:
                fam = "dyck"
            elif tok == p19.EOS:
                fam = None
            tt = 0
            if fam == "dyck":
                if BRK <= tok < BRK + 2:
                    depth += 1
                elif BRK + 2 <= tok < BRK + 4:
                    depth -= 1
                    if depth <= 0:
                        fam, depth = None, 0
        return route
    return router


def eval_config(T, seeds=(111, 222, 333)):
    p21.causal_route = make_deep_router(T)   # patch module global
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        mA = p21.VETDCC(p21.V, 24, k=8, K=8)
        mB = p21.STACKDCC2_D12(p21.V, 24, k=8, K=8)
        mA.load_state_dict(torch.load(
            os.path.join(CKPT, f"A_s{seed}.pt"))["sd"])
        mB.load_state_dict(torch.load(
            os.path.join(CKPT, f"B_s{seed}.pt"))["sd"])
        mA.eval(); mB.eval()
        row = {}
        row["task_acc_hard"] = p21.routed_task_acc(
            mA, mB, 24, 256, random.Random(666), hard=True)
        row["task_acc_train"] = p21.routed_task_acc(
            mA, mB, 24, 256, random.Random(555), hard=False)
        vha = p19.make_batches(8, 256, random.Random(777), hard=True)
        v1024 = p19.make_batches(2, 1024, random.Random(31415))
        ce_h = p21.routed_ce(mA, mB, vha, 256)
        ce_1024 = p21.routed_ce(mA, mB, v1024, 1024)
        row["ce"] = {"256_hard": ce_h, "1024": ce_1024}
        row["len_ratio_1024_over_256hard"] = round(
            ce_1024 / max(1e-9, ce_h), 3)
        cl = {}
        for d in range(1, 13):
            Ld = 256 if p13d.seg_len(d) <= 252 else p13d.seg_len(d) + 16
            nst = 8 if Ld <= 256 else (4 if Ld <= 4096 else 2)
            cl[f"close_d{d}"] = p21.routed_close_acc(mA, mB, nst, d, Ld)[0]
        row["dyck_close_routed"] = cl
        row["joint_mixdd"] = p21.joint_mixed_eval(mA, mB, 24, 256,
                                                 corpus="mixdd")
        rows.append(row)
        print(f"[p21c T={T} s{seed}] pair="
              f"{row['task_acc_hard']['pair']} modk="
              f"{row['task_acc_hard']['modk']} ratio="
              f"{row['len_ratio_1024_over_256hard']} dyck_d12="
              f"{cl['close_d12']} joint_pair="
              f"{row['joint_mixdd']['pair']} joint_dyck="
              f"{row['joint_mixdd']['dyck_close']}", flush=True)
    n = len(rows)
    md12 = [r["dyck_close_routed"]["close_d12"] for r in rows]
    pairs = [r["task_acc_hard"]["pair"] for r in rows]
    ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
    mods = [r["task_acc_hard"]["modk"] for r in rows]
    summary = {
        "T": T, "n_seeds": n,
        "dyck_close_d12_dist": md12,
        "basin_dyck_d12_ge_85": sum(1 for x in md12 if x >= 0.85) / n,
        "pair_eval_dist": pairs,
        "basin_pair_ge_717": sum(1 for x in pairs if x >= 0.717) / n,
        "len_ratio_dist": ratios,
        "basin_ratio_le_6": sum(1 for x in ratios if x <= 0.6) / n,
        "modk_eval_dist": mods,
        "basin_modk_eq_1": sum(1 for x in mods if x == 1.0) / n,
        "joint_pair_dist": [r["joint_mixdd"]["pair"] for r in rows],
        "joint_dyck_dist": [r["joint_mixdd"]["dyck_close"] for r in rows],
        "bars_passed_per_seed": [
            int(p >= 0.717) + int(m == 1.0) + int(x <= 0.6)
            + int(d >= 0.85)
            for p, m, x, d in zip(pairs, mods, ratios, md12)]}
    print(f"[p21c T={T}] SUMMARY {json.dumps(summary)}", flush=True)
    return {"per_seed": {str(s): r for s, r in zip(seeds, rows)},
            "summary": summary}


def main():
    result = {"tag": "ARCH-VET-LM-P21C",
              "protocol": "depth-aware dispatch (causally "
                          "implementable): route to the dyck "
                          "specialist iff family==dyck AND entering "
                          "bracket depth >= T; T=3 primary, T=2 "
                          "secondary; NO retraining (existing P21 "
                          "checkpoints), eval-only. Targets: "
                          "unified row 4/4 bars 3/3 + preserved "
                          "joint pair~1.0 with dyck>=.85.",
              "arms": {}}
    for T in (3, 2):
        result["arms"][f"deep-router-T{T}"] = eval_config(T)
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open(os.path.join(REPO, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
