# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 13b (cycle 56) — DETERMINISTIC DYCK CAPACITY
FRONTIER, DEPTH-MIXED TRAINING. P13 ran the P11 protocol (train
depth 2) on the DETERMINISTIC grammar and found exact-match 0.0 at
ALL depths d3-12 for BOTH stack-capacity arms (D6 and D12, equal
params) and close-type ~chance decaying with depth, despite train
loss 0.011 (near-perfect fit). Diagnosis: with a deterministic
grammar at a FIXED train depth, every depth-2 segment is the SAME
string, so all 256 pool streams are byte-identical -> the corpus is
ONE periodic pattern; the model memorizes the period (loss 0.011)
and never needs its stack, so nothing generalizes (L-DETERMINISTIC-
SINGLE-STRING-MEMORIZATION; banked negative). The stochastic
grammar (P11/P12) avoided this by construction (coin-flip opens)
but made whole-segment exact-match unmeasurable (open flips are
not state-determined). P13b keeps the deterministic grammar (every
token state-determined -> exact-match ~1.0 is the certifiable bar)
and restores corpus DIVERSITY by training on a MIX of depths
{1,2,3}: S(1), S(2), S(3) interleaved in every stream. A stack
user must then track WHICH branch/level it is in to close
correctly (c_b->c_a vs c_b->c_b vs c_b->T_TASK all occur,
distinguished only by stack top), so stack-use is forced in
training, while eval at d4..d12 is strictly DEEPER than training
(recursion one+ level beyond the max seen) -> measures algorithmic
length/depth invariance of the learned stack policy. In-train
d1..d3 exact is the sanity gate (must be ~1.0 if the model masters
the corpus under the 2000-step budget).

GRAMMAR (identical to P13): emit(d) = [a] emit(d-1) [c_a] [b]
emit(d-1) [c_b], a=BRK, c_a=BRK+2, b=BRK+1, c_b=BRK+3,
|S(d)| = 2^(d+2) - 4. Max nesting of S(d) == d -> capacity-D stack
user exact for d <= D, must fail at d = D+1.

Prior art (searched 2026-09-06, carried from P13): Suzgun et al.
COLING 2020 (bounded-depth Dyck generalizes across length; fixed-
state nets are bounded-depth over Dyck); Hahn TACL 2020
(transformers fail Dyck depth-generalization asymptotically);
soft-stack line (Joulin-Mikolov 2015 arXiv 1503.01007; Stogin 2020
arXiv 2006.03651; Dusell-Chiang 2025 arXiv 2511.03547); Zhou et
al. 2023 (train-on-more-depth-hierarchy improves Dyck
generalization in transformers, arXiv 2310.13349 - training-depth
diversity, not just length diversity, is what induces the
recursive algorithm; direct rationale for depth-mixed training).
Gap: exact discrete capacity-parameterized content stack with a
learned controller at micro scale, depth-mixed train vs deeper OOD
eval, capacity dissociation D6/D12 at equal params.

ARMS (identical to P13 for comparability): train depth-mixed
{1,2,3}, L=256 pool 256 seed 12345, 2000 steps seed 0:
  STACKDCC2-big  D=6   (21,817p)
  STACKDCC2-big  D=12  (21,817p; D_STACK patched in p11 source)
  VETDCC-big           (21,257p; depth counter, NO content stack)
  TFMicro              (8,144p; attention control, d<=6 only)
EVAL: whole-segment exact-match d1..d12 + per-position close-type
d1/d2/d3/d4/d6/d8/d10/d12; per-depth L = 256 for |S(d)|<=252 else
|S(d)|+16; adaptive stream counts 8/4/2.
SHARP PREDICTIONS (falsifiable, logged per program protocol):
  (a) sanity: ALL arms exact ~1.0 at d1-3 (in-train; if < 0.9 the
      2000-step budget is the binding constraint -> re-run with
      2.5x budget before interpreting OOD);
  (b) STACKDCC2-big D=6: exact ~1.0 at d4-6 (in capacity),
      COLLAPSE at d7-12 (overflow at 7th nested open);
  (c) STACKDCC2-big D=12: exact ~1.0 out to d11-12;
  (d) TFMicro well below both stack arms at d4-6 (no type-order
      tracking without recurrence/stack; Hahn 2020 pre-failure
      window);
  (e) VETDCC-big (depth counter only) collapses at d>=3 where
      close type requires the CONTENT (type order), not just the
      count (L-DYCK-NEEDS-CONTENT-STACK carried to deterministic
      grammar).
Falsification of (b)+(c) together (both arms equal) = capacity is
NOT the frontier variable -> stack channel unused -> architecture
problem, bank and pivot. Falsification of (c) alone (D12 fails
early too) = capacity dissociates but policy doesn't scale.
Tag ARCH-VET-LM-P13B.
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
V = _ns["V"]; VETDCC = _ns["VETDCC"]
STACKDCC2 = _ns["STACKDCC2"]
train_dyck = _ns["train_dyck"]; n_params = _ns["n_params"]
T_TASK = _ns["T_TASK"]; BRK = _ns["BRK"]
_src_lm = open("arch_vet_lm.py", encoding="utf-8").read()
_ns_lm = {}
exec(compile(_src_lm.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns_lm)
TFMicro = _ns_lm["TFMicro"]
BOS = _ns_lm["BOS"]; EOS = _ns_lm["EOS"]; MODS = _ns_lm["MODS"]
assert (T_TASK, BRK) == (_ns_lm["T_TASK"], _ns_lm["BRK"])

_ns12 = {}
src12 = _src11.replace('D_STACK = _ns["D_STACK"]',
                       'D_STACK = 12')
exec(compile(src12.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p11_D12.py", "exec"), _ns12)
STACKDCC2_D12 = _ns12["STACKDCC2"]
assert STACKDCC2_D12 is not STACKDCC2

TRAIN_DEPTHS = (1, 2, 3)   # max nesting 3 in training; eval d4+ is OOD


# ---- deterministic grammar ---------------------------------------
def det_emit(d):
    if d <= 0:
        return []
    return ([BRK] + det_emit(d - 1) + [BRK + 2]
            + [BRK + 1] + det_emit(d - 1) + [BRK + 3])


def det_seg_len(d):
    return 4 if d == 1 else (2 * det_seg_len(d - 1) + 4)


def gen_det_mixed_stream(rng, L, depths=TRAIN_DEPTHS):
    """Stream whose SEGMENTS alternate among the given depths (rng-
    chosen) -> many DISTINCT streams, all depth <= max(depths)."""
    x = [BOS]
    while len(x) < L:
        d = depths[rng.randrange(len(depths))]
        seg = [T_TASK, T_TASK, T_TASK] + det_emit(d)
        room = L - len(x)
        if len(seg) <= room:
            x += seg
        else:
            x.append(rng.randrange(8) + MODS)
    x = x[:L]
    x.append(EOS)
    return x


def gen_det_mixed_pool(n, L, seed):
    prng = random.Random(seed)
    return [torch.tensor(gen_det_mixed_stream(prng, L)) for _ in range(n)]


def gen_det_stream(rng, L, depth):
    """Fixed-depth eval stream (as P13) - single-string per depth is
    fine at EVAL time (the segment is the unit scored)."""
    x = [BOS]
    seg = [T_TASK, T_TASK, T_TASK] + det_emit(depth)
    while len(x) < L:
        room = L - len(x)
        if len(seg) <= room:
            x += seg
        else:
            x.append(rng.randrange(8) + MODS)
    x = x[:L]
    x.append(EOS)
    return x


# ---- eval: exact-match + per-position close type ------------------
@torch.no_grad()
def det_exact(model, n, depth, L):
    """Whole-segment exact-match (teacher-forced, argmax vs gold)."""
    model.eval()
    ok = tot = 0
    for i in range(n):
        x = torch.tensor(gen_det_stream(random.Random(5000 + i * 31 + depth),
                                        L, depth)).unsqueeze(0)
        lg = model(x)[:, :L, :]
        pred = lg.argmax(-1).squeeze(0)
        xl = x.squeeze(0).tolist()
        j = 0
        while j < L - 3:
            if xl[j] == T_TASK and xl[j + 1] == T_TASK \
                    and xl[j + 2] == T_TASK:
                i2, depth_i, seg = j + 3, 0, []
                while i2 < L:
                    t2 = xl[i2]
                    if BRK <= t2 < BRK + 4:
                        depth_i += 1 if t2 < BRK + 2 else -1
                        seg.append(i2)
                        if depth_i == 0:
                            i2 += 1
                            break
                    i2 += 1
                if seg and depth_i == 0:
                    tot += 1
                    ok += int(all(int(pred[g - 1]) == xl[g]
                                  for g in seg))
                j = max(seg) + 1 if seg else j + 3
            else:
                j += 1
    return (round(ok / tot, 4) if tot else float("nan")), tot


@torch.no_grad()
def det_close_acc(model, n, depth, L):
    """Per-position CLOSE-type accuracy (close == stack top)."""
    model.eval()
    ok = tot = 0
    for i in range(n):
        x = torch.tensor(gen_det_stream(random.Random(9000 + i * 17 + depth),
                                        L, depth)).unsqueeze(0)
        lg = model(x)[:, :L, :]
        pred = lg.argmax(-1).squeeze(0)
        xl = x.squeeze(0).tolist()
        j = 0
        while j < L - 3:
            if xl[j] == T_TASK and xl[j + 1] == T_TASK \
                    and xl[j + 2] == T_TASK:
                i2, depth_i, seg = j + 3, 0, []
                while i2 < L:
                    t2 = xl[i2]
                    if BRK <= t2 < BRK + 4:
                        depth_i += 1 if t2 < BRK + 2 else -1
                        seg.append(i2)
                        if depth_i == 0:
                            i2 += 1
                            break
                    i2 += 1
                for g in seg:
                    if BRK + 2 <= xl[g] < BRK + 4:   # close position
                        tot += 1
                        ok += int(int(pred[g - 1]) == xl[g])
                j = max(seg) + 1 if seg else j + 3
            else:
                j += 1
    return (round(ok / tot, 4) if tot else float("nan")), tot


if __name__ == "__main__":
    pool = gen_det_mixed_pool(256, 256, 12345)
    s2 = det_emit(2)
    assert det_seg_len(2) == len(s2) == 12
    for d in range(1, 13):
        assert det_seg_len(d) == len(det_emit(d))
    distinct = len(set(tuple(p.tolist()) for p in pool))
    print(f"[P13B] mixed-depth pool: {len(pool)} streams, "
          f"{distinct} distinct (P13 had 1; fix verified)", flush=True)
    assert distinct > 100, "pool degeneracy NOT fixed - abort"
    L_by_depth = {d: (256 if det_seg_len(d) <= 252
                      else det_seg_len(d) + 16) for d in range(1, 13)}

    def n_streams(d):
        L = L_by_depth[d]
        if L <= 256:
            return 8
        if L <= 4096:
            return 4
        return 2

    # P13 confound fixed: arms are CONSTRUCTED under per-arm
    # manual_seed(0) (P13 built the list before the seed reset ->
    # D6/D12 had different random inits -> capacity confounded with
    # basin luck). Here seed 0 -> D6 and D12 get BIT-IDENTICAL
    # weights (param shapes do not depend on D_STACK); during
    # depth<=3 training sp<=3<6 so overflow never fires -> identical
    # gradients -> identical final weights. Mechanism checks:
    #  (i) assert weights allclose D6 vs D12 after training;
    #  (ii) d1-6 eval MUST be identical between D6 and D12 (no
    #       overflow below the 7th nested open); any d<=6
    #       difference would falsify "capacity is a pure buffer".
    # Dissociation can only appear at d7+ (overflow boundary).
    arm_specs = [("STACKDCC2-big-D6",
                  lambda: STACKDCC2(V, 24, k=8, K=8)),
                 ("STACKDCC2-big-D12",
                  lambda: STACKDCC2_D12(V, 24, k=8, K=8)),
                 ("VETDCC-big",
                  lambda: VETDCC(V, 24, k=8, K=8)),
                 ("TFMicro",
                  lambda: TFMicro(V, 16))]
    result = {"tag": "ARCH-VET-LM-P13B",
              "protocol": "DETERMINISTIC dyck, DEPTH-MIXED train "
                          "d in {1,2,3} per segment (P13 single-"
                          "string degeneracy fix: 256 DISTINCT "
                          "streams), L=256 pool 256 seed 12345, "
                          "2000 steps seed 0, arms constructed "
                          "UNDER manual_seed(0) (P13 init confound "
                          "fix: D6/D12 bit-identical weights, "
                          "capacity is a pure runtime buffer); "
                          "eval whole-segment exact d1-12 + "
                          "close-type d1/2/3/4/6/8/10/12; per-depth "
                          "L (256 if |S(d)|<=252 else |S(d)|+16); "
                          "TFMicro eval d<=6 only",
              "arms": {}}
    trained = {}
    for name, ctor in arm_specs:
        torch.manual_seed(0)
        m = ctor()
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_dyck(name, m, pool, 2000, 8)
        m.eval()
        ex, cl = {}, {}
        for d in range(1, 13):
            if name == "TFMicro" and d > 6:
                break
            L = L_by_depth[d]
            a, tot = det_exact(m, n_streams(d), d, L)
            ex[f"exact_d{d}"] = a
            print(f"  [{name}] d{d} L{L}: exact {a} ({tot} segs)",
                  flush=True)
        for d in (1, 2, 3, 4, 6, 8, 10, 12):
            if name == "TFMicro" and d > 6:
                break
            L = L_by_depth[d]
            a, tot = det_close_acc(m, n_streams(d), d, L)
            cl[f"close_d{d}"] = a
            print(f"  [{name}] d{d} L{L}: close-type {a} ({tot} pos)",
                  flush=True)
        result["arms"][name] = {"params": n_params(m),
                                "loss_curve": hist,
                                "exact_match_frontier": ex,
                                "close_type_by_depth": cl}
        trained[name] = m
    # ---- capacity-purity mechanism checks ----
    sd6 = trained["STACKDCC2-big-D6"].state_dict()
    sd12 = trained["STACKDCC2-big-D12"].state_dict()
    w_eq = all(torch.equal(sd6[k], sd12[k]) for k in sd6)
    print(f"[P13B] D6 vs D12 weights identical after training: "
          f"{w_eq}", flush=True)
    ex6 = result["arms"]["STACKDCC2-big-D6"]["exact_match_frontier"]
    ex12 = result["arms"]["STACKDCC2-big-D12"]["exact_match_frontier"]
    cl6 = result["arms"]["STACKDCC2-big-D6"]["close_type_by_depth"]
    cl12 = result["arms"]["STACKDCC2-big-D12"]["close_type_by_depth"]
    cap_check = {
        "weights_identical_d6_d12": w_eq,
        "max_abs_exact_diff_d1_6": max(abs(ex6[f"exact_d{d}"] - ex12[f"exact_d{d}"])
                                       for d in range(1, 7)),
        "max_abs_close_diff_d1_6": max(abs(cl6[f"close_d{d}"] - cl12[f"close_d{d}"])
                                       for d in (1, 2, 3, 4, 6))}
    result["capacity_purity_checks"] = cap_check
    print("[P13B] capacity-purity checks: "
          f"{json.dumps(cap_check)}", flush=True)
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
