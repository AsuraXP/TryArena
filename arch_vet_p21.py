# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 21 (cycle 61) — MODULAR-FUSION PILOT: THE
UNIFIED ROW. C58-C60 closed the DATA-level fusion route with the
FUSION WALL (P19/P19B/P19c/P20): in ONE shared L=256 stream the
three certified axes are pairwise contradictory — dyck d12 needs
d6-IN-TRAIN (~>=50% of a 256 stream; a d5 ceiling makes the
>train-depth carry a 1/3 seed lottery, P20), pair needs vanilla
exemplar density (0/3 under any deep-dyck share), and CE ratio <=.6
is broken by ANY deep-dyck training share (0/3 mixdd and 0/3
sched-50%, vs 3/3 vanilla on both arms). The C60 plan therefore
triggered the FORK: fusion must be ARCHITECTURAL.
P21 = the minimal, honest architectural fusion: TWO certified
experts + a CAUSAL STRUCTURAL ROUTER, evaluated as ONE deployed
system on ONE heterogeneous stream.
  EXPERT A (generalist, 21,257p, VETDCC-big) trained in its
    certified regime: the vanilla P9 4-task pool @4000 steps
    (P16: pair 3/3 .767, ratio 3/3, modk 1.0 3/3).
  EXPERT B (dyck specialist, 21,817p, STACKDCC2-big-D12) trained in
    its certified regime: the P13D deep-mixed depth-diverse corpus
    (train depths 2-6 weighted deep) @2000 steps (P13D/P18: close
    d12 ~.97, 3/3 when d6 is in train).
  COMBINED 43,074p — inside the program's 8k-70k envelope; the
    designed claim is a ONE-SYSTEM configuration, not two reports.
  ROUTER: causal structural dispatch. At every position t the
    family is decided from tokens < t ONLY, by the same prefix
    grammar the certified metrics parse: a single T_TASK -> track;
    T_TASK T_TASK + ONE -> modk; T_TASK T_TASK + bracket -> dyck;
    T_TASK T_TASK + key -> pair; three T_TASK -> dyck. Dyck-family
    positions go to expert B, everything else to expert A. The ONE
    genuinely ambiguous position is the token right after T_TASK
    T_TASK (the family is only decidable once that token appears);
    it defaults to A (one token per segment; counted honestly).
    This mirrors the program's existing honesty clause for the DCC
    channels: the router reads the task GRAMMAR, which is exactly
    what the certified protocol emits — the learned-router ablation
    is the queued follow-up, not claimed here.
Prior art (searched 2026-09-08): arXiv 2409.14981 "On The
Specialization of Neural Modules" — modules FAIL to specialize
unless trained with strong task-specific separation, i.e. naive
modularity is not enough and per-module training data is required:
this is precisely P21's design (each expert trained ONLY on its
own certified corpus), and it predicts why a shared-stream
monolith (P19-P20) could not specialize. MoE surveys
(arXiv 2507.11181; MoE field surveys 2026) — routing granularity
matters: TASK-LEVEL routing minimizes interference (token-level
gating specializes finer but couples); P21 uses task/family-level
routing for exactly that reason. arXiv 2408.00508 (Block-
Operations / Multiplexer) — routing-based inductive bias improves
compositional generalization. Program precedents: C22b fused
controller modules into one coherent 68,738p module (13/13 bars)
— architectural composition is the established path in this
program; P19/P19B/P19c/P20 = the fusion wall this pilot answers.
EVAL (per seed 111/222/333, ONE composed system):
  (1) UNIFIED ROW: routed task_acc on the vanilla hard protocol
      (pair/modk/track), routed CE sweep -> length-invariance ratio,
      routed close-type d1-12 (dyck path) -> dyck d12.
      Target: all four bars >= 2/3 (modk 1.0, pair >=.717,
      ratio <=.6, dyck d12 >=.85) in ONE system.
  (2) JOINT HETEROGENEOUS EVAL (the C58-C60 payoff): on the mixdd
      streams (depth-diverse dyck interleaved with track/modk/pair
      — the corpus that broke every monolith), measure per-family
      answer accuracy AND dyck close-type accuracy in the SAME
      pass. Target: pair at vanilla level (~.7-.9) WHILE dyck
      close-type is at certified level (~.85+) simultaneously —
      false for every single-training-stream config tested
      (P19 pair 1/3, P19c pair 0/3, P20 pair 0/3 with dyck 1/3).
  (3) Attribution: A-alone and B-alone numbers on the same eval
      sets, and the routed-vs-oracle gap at the ambiguous position.
FALSIFIABILITY: if the joint eval shows pair collapsing again
despite routing, then the interference is CAUSAL (state/trunk
level) rather than data-level within the tested corpora -> the
composition needs separate state, documented as a banked negative.
Tag ARCH-VET-LM-P21.
"""
import argparse, json, os, random, subprocess, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")
CKPT = os.path.join(REPO, "p21_ckpt")
T0 = time.time()

import torch
torch.set_num_threads(1)

import arch_vet_p19 as p19
import arch_vet_p13d as p13d
import arch_vet_lm as lm

V = p19.V; BOS = p19.BOS; T_TASK = p19.T_TASK
ONE = p19.ONE; A_MARK = p19.A
TRACK = p19.TRACK; MANS = p19.MANS
KEYS = p19.KEYS; VALS = p19.VALS
BRK = p19.BRK; MODS = p19.MODS
VETDCC = p13d.VETDCC
STACKDCC2_D12 = p19.STACKDCC2_D12

A_STEPS, B_STEPS = 4000, 2000


def expert_spec(name):
    if name == "A":          # generalist / pair-ratio-modk expert
        return (lambda: VETDCC(V, 24, k=8, K=8), A_STEPS,
                lambda: p19.make_pool(512, 256, 12345), "vanilla")
    if name == "B":          # dyck specialist
        return (lambda: STACKDCC2_D12(V, 24, k=8, K=8), B_STEPS,
                lambda: p13d.gen_mix_pool(256, 256, 12345), "deepmix")
    raise SystemExit(f"unknown expert {name}")


# ------------------------------------------------------- causal router
def causal_route(toks):
    """route[t] = 1 (dyck expert) or 0 (generalist), decided from
    tokens < t ONLY (causal structural dispatch)."""
    n = len(toks)
    route = [0] * n
    fam = None
    depth = 0
    tt = 0                      # run of consecutive T_TASKs consumed
    for t in range(n):
        route[t] = 1 if fam == "dyck" else 0
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


def routed_logits(mA, mB, x):
    """x: (1, L+1) long. Returns logits (1, L+1, V) chosen per
    position by the causal router."""
    lgA = mA(x)
    routes = causal_route(x[0].tolist())
    if any(routes):
        lgB = mB(x)
        sel = torch.tensor(routes, dtype=torch.bool).view(1, -1, 1)
        return torch.where(sel, lgB, lgA)
    return lgA


# --------------------------------------------- routed metric helpers
@torch.no_grad()
def routed_ce(mA, mB, xs, L):
    lg = routed_logits(mA, mB, xs[:, :L + 1])
    nll = -torch.log_softmax(lg, -1).gather(
        -1, xs[:, 1:L + 1].unsqueeze(-1)).squeeze(-1)
    return round(float(nll.mean()), 4)


@torch.no_grad()
def routed_task_acc(mA, mB, n, L, rng, hard):
    """Same gold-parse logic as arch_vet_lm.task_acc, routed logits."""
    fam_ok = {"track": [0, 0], "modk": [0, 0], "dyck": [0, 0],
              "pair": [0, 0]}
    for _ in range(n):
        x2 = lm.gen_stream(rng, L, hard)[0][:L + 1]
        lg = routed_logits(mA, mB, torch.tensor(x2).unsqueeze(0))[:, :L, :]
        pred = lg.argmax(-1).squeeze(0)
        i = 0
        while i < len(x2) - 4:
            if x2[i] != T_TASK:
                i += 1
                continue
            n1 = x2[i + 1]
            n2 = x2[i + 2] if i + 2 < len(x2) else -1
            if TRACK <= n1 < TRACK + 8:                 # track
                j = i
                while j < len(x2) - 1 and x2[j] != A_MARK:
                    j += 1
                gold = [j + 1]
                fam = "track"
            elif n1 == T_TASK and n2 == T_TASK:         # dyck
                i2, d, seg = i + 3, 0, []
                while i2 < len(x2):
                    t2 = x2[i2]
                    if BRK <= t2 < BRK + 4:
                        d += 1 if t2 < BRK + 2 else -1
                        seg.append(i2)
                        if d == 0:
                            i2 += 1
                            break
                    i2 += 1
                gold = [g for g in seg if BRK + 2 <= x2[g] < BRK + 4]
                fam = "dyck"
            elif n1 == T_TASK:                          # modk or pair
                if n2 == ONE:
                    j = i
                    while j < len(x2) - 1 and x2[j] != A_MARK:
                        j += 1
                    gold = [j + 1]
                    fam = "modk"
                else:
                    j = i
                    while j < len(x2) - 1 and x2[j] != A_MARK:
                        j += 1
                    gold = [j + 1, j + 2]
                    fam = "pair"
            else:
                i += 1
                continue
            for g in gold:
                fam_ok[fam][1] += 1
                fam_ok[fam][0] += int(int(pred[g - 1]) == x2[g])
            i = max(gold) + 1 if gold else i + 1
    return {k: round(v[0] / v[1], 4) if v[1] else 0.0
            for k, v in fam_ok.items()}


@torch.no_grad()
def routed_close_acc(mA, mB, n, depth, L, model_only=None):
    """rt_close_acc (p13d) with routed logits; model_only='A'/'B'
    runs a single expert instead of the route."""
    ok = tot = 0
    for i in range(n):
        x = torch.tensor(p13d.gen_rt_stream(
            random.Random(9000 + i * 17 + depth), L, depth)).unsqueeze(0)
        if model_only == "A":
            lg = mA(x)
        elif model_only == "B":
            lg = mB(x)
        else:
            lg = routed_logits(mA, mB, x)
        pred = lg[:, :L, :].argmax(-1).squeeze(0)
        xl = x.squeeze(0).tolist()
        j = 0
        while j < L - 3:
            if xl[j] == T_TASK and xl[j + 1] == T_TASK \
                    and xl[j + 2] == T_TASK:
                i2, d, seg = j + 3, 0, []
                while i2 < L:
                    t2 = xl[i2]
                    if BRK <= t2 < BRK + 4:
                        d += 1 if t2 < BRK + 2 else -1
                        seg.append(i2)
                        if d == 0:
                            i2 += 1
                            break
                    i2 += 1
                for g in seg:
                    if BRK + 2 <= xl[g] < BRK + 4:
                        tot += 1
                        ok += int(int(pred[g - 1]) == xl[g])
                j = max(seg) + 1 if seg else j + 3
            else:
                j += 1
    return (round(ok / tot, 4) if tot else float("nan")), tot


@torch.no_grad()
def joint_mixed_eval(mA, mB, n, L, seed=4242, corpus="mixdd"):
    """ONE pass over heterogeneous streams (all four families,
    depth-diverse dyck included): per-family answer accuracy AND
    dyck close-type accuracy SIMULTANEOUSLY, via routed logits."""
    if corpus == "mixdd":
        gen = p19.gen_mixdd_stream
    elif corpus == "sched":
        import arch_vet_p20 as p20
        gen = p20.gen_sched_stream
    else:
        gen = lm.gen_stream
    rng = random.Random(seed)
    fam_ok = {"track": [0, 0], "modk": [0, 0], "pair": [0, 0]}
    cok = ctot = 0
    for _ in range(n):
        x2 = gen(rng, L)[:L + 1]
        lg = routed_logits(mA, mB, torch.tensor(x2).unsqueeze(0))[:, :L, :]
        pred = lg.argmax(-1).squeeze(0)
        i = 0
        while i < len(x2) - 4:
            if x2[i] != T_TASK:
                i += 1
                continue
            n1 = x2[i + 1]
            n2 = x2[i + 2] if i + 2 < len(x2) else -1
            if TRACK <= n1 < TRACK + 8:
                j = i
                while j < len(x2) - 1 and x2[j] != A_MARK:
                    j += 1
                fam_ok["track"][1] += 1
                fam_ok["track"][0] += int(int(pred[j]) == x2[j + 1])
                i = j + 2
            elif n1 == T_TASK and n2 == T_TASK:
                i2, d, seg = i + 3, 0, []
                while i2 < len(x2):
                    t2 = x2[i2]
                    if BRK <= t2 < BRK + 4:
                        d += 1 if t2 < BRK + 2 else -1
                        seg.append(i2)
                        if d == 0:
                            i2 += 1
                            break
                    i2 += 1
                for g in seg:
                    if BRK + 2 <= x2[g] < BRK + 4:
                        ctot += 1
                        cok += int(int(pred[g - 1]) == x2[g])
                i = max(seg) + 1 if seg else i + 3
            elif n1 == T_TASK:
                if n2 == ONE:
                    j = i
                    while j < len(x2) - 1 and x2[j] != A_MARK:
                        j += 1
                    fam_ok["modk"][1] += 1
                    fam_ok["modk"][0] += int(int(pred[j]) == x2[j + 1])
                    i = j + 2
                else:
                    j = i
                    while j < len(x2) - 1 and x2[j] != A_MARK:
                        j += 1
                    fam_ok["pair"][1] += 2
                    fam_ok["pair"][0] += (int(int(pred[j]) == x2[j + 1])
                                          + int(int(pred[j + 1]) == x2[j + 2]))
                    i = j + 3
            else:
                i += 1
    out = {k: round(v[0] / v[1], 4) if v[1] else 0.0
           for k, v in fam_ok.items()}
    out["dyck_close"] = round(cok / ctot, 4) if ctot else float("nan")
    out["n_close"] = ctot
    return out


# ------------------------------------------------------------- modes
def cmd_train(a):
    os.makedirs(CKPT, exist_ok=True)
    ctor, steps, pool_fn, cname = expert_spec(a.expert)
    torch.manual_seed(a.seed)                 # SEED HYGIENE LAW
    m = ctor()
    pool = pool_fn()
    t0 = time.time()
    hist = p19.train_arm(f"P21-{a.expert}-{cname}-s{a.seed}", m, pool,
                         steps, 8)
    path = os.path.join(CKPT, f"{a.expert}_s{a.seed}.pt")
    torch.save({"sd": m.state_dict(), "expert": a.expert,
                "seed": a.seed, "steps": steps, "corpus": cname,
                "params": p19.n_params(m), "hist": hist}, path)
    print(f"[p21 train] {path} in {time.time()-t0:.0f}s "
          f"params={p19.n_params(m)}", flush=True)
    return 0


def cmd_compose(a):
    seeds = [int(s) for s in a.seeds.split(",")]
    result = {"tag": "ARCH-VET-LM-P21",
              "protocol": "modular fusion: expert A (VETDCC-big, "
                          "21,257p) on the vanilla P9 4-task pool "
                          "@4000 + expert B (STACKDCC2-big D12, "
                          "21,817p) on the P13D deep-mixed "
                          "depth-diverse corpus @2000, combined "
                          "43,074p, dispatched by a CAUSAL "
                          "STRUCTURAL router (family from tokens "
                          "<t only; ambiguous position after "
                          "T_TASK T_TASK defaults to A). Eval per "
                          "seed: (1) unified row = routed hard "
                          "task_acc + routed CE ratio + routed "
                          "close d1-12; (2) joint mixed-stream eval "
                          "on mixdd (per-family acc AND dyck "
                          "close in the SAME pass); (3) attribution "
                          "A-alone / B-alone close-type + P19/P19c "
                          "monolith references (pair 1/3 / 0/3).",
              "arms": {}}
    rows = []
    for seed in seeds:
        cka = os.path.join(CKPT, f"A_s{seed}.pt")
        ckb = os.path.join(CKPT, f"B_s{seed}.pt")
        if not (os.path.exists(cka) and os.path.exists(ckb)):
            print(f"[p21] seed {seed}: MISSING checkpoints, skipped",
                  flush=True)
            continue
        torch.manual_seed(seed)
        mA = VETDCC(V, 24, k=8, K=8)
        mB = STACKDCC2_D12(V, 24, k=8, K=8)
        mA.load_state_dict(torch.load(cka)["sd"])
        mB.load_state_dict(torch.load(ckb)["sd"])
        mA.eval(); mB.eval()
        row = {"params_total": 21257 + 21817}
        # (1) unified row
        rng_h = random.Random(666)
        row["task_acc_hard"] = routed_task_acc(mA, mB, 24, 256, rng_h,
                                               hard=True)
        rng_t = random.Random(555)
        row["task_acc_train"] = routed_task_acc(mA, mB, 24, 256, rng_t,
                                                hard=False)
        vha = p19.make_batches(8, 256, random.Random(777), hard=True)
        v512 = p19.make_batches(2, 512, random.Random(31337))
        v1024 = p19.make_batches(2, 1024, random.Random(31415))
        ce_h = routed_ce(mA, mB, vha, 256)
        ce_512 = routed_ce(mA, mB, v512, 512)
        ce_1024 = routed_ce(mA, mB, v1024, 1024)
        row["ce"] = {"256_hard": ce_h, "512": ce_512, "1024": ce_1024}
        row["len_ratio_1024_over_256hard"] = round(
            ce_1024 / max(1e-9, ce_h), 3)
        cl = {}
        for d in range(1, 13):
            Ld = p19.L_by_depth[d] if hasattr(p19, "L_by_depth") else (
                256 if p13d.seg_len(d) <= 252 else p13d.seg_len(d) + 16)
            nst = 8 if Ld <= 256 else (4 if Ld <= 4096 else 2)
            cl[f"close_d{d}"] = routed_close_acc(mA, mB, nst, d, Ld)[0]
        row["dyck_close_routed"] = cl
        # attribution on a cheap subset (d6, d12)
        row["attr"] = {
            "close_d6_A_alone": routed_close_acc(
                mA, mB, 8, 6, 256, model_only="A")[0],
            "close_d6_B_alone": routed_close_acc(
                mA, mB, 8, 6, 256, model_only="B")[0],
            "close_d12_B_alone": routed_close_acc(
                mA, mB, 2, 12, p13d.seg_len(12) + 16,
                model_only="B")[0]}
        # (2) joint heterogeneous eval (the fusion-wall payoff)
        row["joint_mixdd"] = joint_mixed_eval(mA, mB, 24, 256,
                                             corpus="mixdd")
        row["joint_sched"] = joint_mixed_eval(mA, mB, 24, 256,
                                             corpus="sched")
        print(f"[p21 s{seed}] unified: pair="
              f"{row['task_acc_hard']['pair']} modk="
              f"{row['task_acc_hard']['modk']} ratio="
              f"{row['len_ratio_1024_over_256hard']} dyck_d12="
              f"{cl['close_d12']}", flush=True)
        print(f"[p21 s{seed}] joint_mixdd={row['joint_mixdd']}",
              flush=True)
        rows.append(row)
    if not rows:
        raise SystemExit("no composed rows produced")
    n = len(rows)
    md12 = [r["dyck_close_routed"]["close_d12"] for r in rows]
    pairs = [r["task_acc_hard"]["pair"] for r in rows]
    ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
    mods = [r["task_acc_hard"]["modk"] for r in rows]
    jp = [r["joint_mixdd"]["pair"] for r in rows]
    jd = [r["joint_mixdd"]["dyck_close"] for r in rows]
    summary = {
        "n_seeds": n,
        "dyck_close_d12_dist": md12,
        "basin_dyck_d12_ge_85": sum(1 for x in md12 if x >= 0.85) / n,
        "pair_eval_dist": pairs,
        "basin_pair_ge_717": sum(1 for x in pairs if x >= 0.717) / n,
        "len_ratio_dist": ratios,
        "basin_ratio_le_6": sum(1 for x in ratios if x <= 0.6) / n,
        "modk_eval_dist": mods,
        "basin_modk_eq_1": sum(1 for x in mods if x == 1.0) / n,
        "joint_pair_dist": jp,
        "joint_dyck_dist": jd,
        "joint_both_ok": sum(1 for a, b in zip(jp, jd)
                             if a >= 0.717 and b >= 0.85) / n}
    print(f"[P21 COMPOSED 43,074p] SUMMARY {json.dumps(summary)}",
          flush=True)
    result["arms"]["fused-VETDCC+STACKDCC2-router"] = {
        "per_seed": {str(s): r for s, r in zip(seeds, rows)},
        "summary": summary}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open(os.path.join(REPO, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--expert", required=True, choices=["A", "B"])
    t.add_argument("--seed", type=int, required=True)
    t.set_defaults(func=cmd_train)
    c = sub.add_parser("compose")
    c.add_argument("--seeds", default="111,222,333")
    c.set_defaults(func=cmd_compose)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
