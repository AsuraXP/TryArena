# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 13d (cycle 56) — DEEP-MIXED RANDOM-TYPE DYCK:
FORCING THE STACK. P13c showed the exact-type-stack channel is never
engaged when the train corpus is window-solvable: depth-2 exact-
shape random-type segments (S(2), max nesting 2) have every close
within a few tokens of its open, so the K=8 content window (present
in VETDCC AND STACKDCC2 alike) resolves all closes — VETDCC-big
(NO type stack) scores d2 close-type 1.0 identically to STACKDCC2,
and every arm collapses at d3+ (root closes ~15+ tokens from their
opens) because no arm was forced to learn a LIFO policy.
L-WINDOW-NOT-STACK-CANDIDATE: across P11-P13c no seed-clean test
ever observed the stk channel doing OOD work; the P11 ".925 stack
win" may be a K-window/structure effect.

P13d forces the stack IN TRAINING: the train corpus mixes EXACT-
SHAPE random-type segments at depths {2,3,4,5,6} (P13c rt_segment;
weights favor depth). At depth >= 4 the root-level closes sit ~15-60
tokens past their opens — far beyond the K=8 window and beyond any
local n-gram — so the ONLY way to fit the training loss on deep
segments is to push/pop types through the stk channel. If the
learned controller can drive the stack, STACKDCC2 must master the
deep train segments; VETDCC-big (window only) must FAIL them —
VETDCC's in-train depth-resolved close-type is the discriminating
control for mechanism attribution. Eval then measures whether the
learned stack policy EXTRAPOLATES one-or-more levels deeper (d7-12),
and whether capacity (D6 vs D12, bit-identical weights under seed-
clean ctor) finally dissociates at the overflow boundary.

Prior art (searched 2026-09-06; lineage in P13/P13c headers):
Suzgun et al. COLING 2020 (bounded-depth Dyck generalizes across
length; fixed-state nets bounded-depth); Hahn TACL 2020 (TF Dyck
depth-generalization fails asymptotically); Zhou et al. 2023 arXiv
2310.13349 (training-depth diversity induces recursive algorithm in
TF); soft-stack line (Dusell-Chiang 2025 arXiv 2511.03547).
P13c's law L-WINDOW-NOT-STACK-CANDIDATE is this phase's falsifiable
target: if STACKDCC2 still fails deep IN-TRAIN segments that
VETDCC also fails, the stack channel is not learnable-to-use at
this budget (controller problem, not presence problem) and the
architecture needs an explicit iteration/compiler mechanism.

ARMS (seed-clean, ctor under per-arm manual_seed(0)):
  STACKDCC2-big D6 (21,817p)   STACKDCC2-big D12 (21,817p)
  VETDCC-big (21,257p)         TFMicro (8,144p, eval d<=6 only)
TRAIN: mixed-depth {2..6} exact-shape random-type segments, L=256
pool 256 seed 12345, 2000 steps seed 0. EVAL: per-position
close-type d1..d12 on fixed-depth random-type streams (primary);
depth-resolved IN-TRAIN close d2..d6 = same eval at train depths
(mastery check + VETDCC discriminator); per-depth L (256 if
|S(d)|+3 <= 256 else |S(d)|+16); TFMicro d<=6.

SHARP PREDICTIONS:
  (a) STACKDCC2-big in-train: close-type d4 >= .8 (deep train
      segments mastered through the stack) while VETDCC-big d4
      in-train is well below (window bound) — the discriminator;
  (b) if (a): STACKDCC2-big D6 OOD close d5-6 high, COLLAPSE at
      d7+ (overflow); D12 extends to d11-12 (capacity frontier
      finally visible);
  (c) if (a) holds but (b) fails for BOTH D6 and D12 (identical) →
      controller learned in-train stack use but cannot iterate
      beyond max train depth → depth-window law, capacity still
      irrelevant.
  If (a) fails (STACKDCC2 == VETDCC on deep in-train) → the stk
  channel is not being driven even when needed → architecture
  controller gap (banked negative, then compiler/iteration organ).
Tag ARCH-VET-LM-P13D.
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

TRAIN_DEPTHS = (2, 3, 4, 5, 6)


def det_shape(d):
    if d <= 0:
        return []
    return ([0] + det_shape(d - 1) + [2]
            + [1] + det_shape(d - 1) + [3])


def rt_segment(rng, d):
    """Exact-depth-d random-type segment body (P13c)."""
    shape = det_shape(d)
    body = []
    st = []
    for k in shape:
        if k in (0, 1):
            t = rng.randrange(2)
            st.append(t)
            body.append(BRK + t)
        else:
            t = st.pop()
            body.append(BRK + 2 + t)
    return body


def seg_len(d):
    return 4 if d == 1 else (2 * seg_len(d - 1) + 4)


def pick_depth(rng):
    """Weighted toward deep: w(d) = d-1 over TRAIN_DEPTHS."""
    ws = [d - 1 for d in TRAIN_DEPTHS]
    return rng.choices(TRAIN_DEPTHS, weights=ws)[0]


def gen_mix_stream(rng, L):
    """Training stream: fresh exact-shape random-type segments at
    mixed depths (deep enough that window-only cannot fit)."""
    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        seg = ([T_TASK, T_TASK, T_TASK]
               + rt_segment(rng, pick_depth(rng)))
        if len(seg) <= room:
            x += seg
        else:
            x.append(rng.randrange(8) + MODS)
    x = x[:L]
    x.append(EOS)
    return x


def gen_mix_pool(n, L, seed):
    prng = random.Random(seed)
    return [torch.tensor(gen_mix_stream(prng, L)) for _ in range(n)]


def gen_rt_stream(rng, L, depth):
    """Fixed-depth random-type eval stream (as P13c)."""
    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        seg = [T_TASK, T_TASK, T_TASK] + rt_segment(rng, depth)
        if len(seg) <= room:
            x += seg
        else:
            x.append(rng.randrange(8) + MODS)
    x = x[:L]
    x.append(EOS)
    return x


@torch.no_grad()
def rt_close_acc(model, n, depth, L):
    """Per-position close-type over COMPLETED segments (P13c)."""
    model.eval()
    ok = tot = 0
    for i in range(n):
        x = torch.tensor(gen_rt_stream(
            random.Random(9000 + i * 17 + depth), L, depth)).unsqueeze(0)
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
                    if BRK + 2 <= xl[g] < BRK + 4:
                        tot += 1
                        ok += int(int(pred[g - 1]) == xl[g])
                j = max(seg) + 1 if seg else j + 3
            else:
                j += 1
    return (round(ok / tot, 4) if tot else float("nan")), tot


if __name__ == "__main__":
    for d in range(1, 13):
        assert seg_len(d) == len(det_shape(d))
    pool = gen_mix_pool(256, 256, 12345)
    distinct = len(set(tuple(p.tolist()) for p in pool))
    # depth coverage: fraction of streams containing an S(>=4) seg
    deep = sum(1 for p in pool
               if any(BRK <= t < BRK + 4 for t in p.tolist()[100:]))
    print(f"[P13D] mixed-depth pool: {len(pool)} streams, "
          f"{distinct} distinct", flush=True)
    assert distinct > 100
    L_by_depth = {d: (256 if seg_len(d) <= 252
                      else seg_len(d) + 16) for d in range(1, 13)}

    def n_streams(d):
        L = L_by_depth[d]
        if L <= 256:
            return 8
        if L <= 4096:
            return 4
        return 2

    arm_specs = [("STACKDCC2-big-D6",
                  lambda: STACKDCC2(V, 24, k=8, K=8)),
                 ("STACKDCC2-big-D12",
                  lambda: STACKDCC2_D12(V, 24, k=8, K=8)),
                 ("VETDCC-big",
                  lambda: VETDCC(V, 24, k=8, K=8)),
                 ("TFMicro",
                  lambda: TFMicro(V, 16))]
    result = {"tag": "ARCH-VET-LM-P13D",
              "protocol": "DEEP-MIXED exact-shape RANDOM-TYPE dyck "
                          "train depths {2..6} weighted deep (d-1), "
                          "L=256 pool 256 seed 12345, 2000 steps "
                          "seed 0, ctor under manual_seed(0); "
                          "window (K=8) cannot fit d>=4 segments "
                          "-> stack forced if learnable; eval "
                          "close-type d1-12 fixed-depth streams "
                          "(in-train d2-6 = same-depth mastery "
                          "check, VETDCC = discriminator); TFMicro "
                          "eval d<=6 only",
              "arms": {}}
    trained = {}
    for name, ctor in arm_specs:
        torch.manual_seed(0)
        m = ctor()
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_dyck(name, m, pool, 2000, 8)
        m.eval()
        cl = {}
        for d in range(1, 13):
            if name == "TFMicro" and d > 6:
                break
            L = L_by_depth[d]
            a, tot = rt_close_acc(m, n_streams(d), d, L)
            cl[f"close_d{d}"] = a
            print(f"  [{name}] d{d} L{L}: close-type {a} ({tot} pos)",
                  flush=True)
        result["arms"][name] = {"params": n_params(m),
                                "loss_curve": hist,
                                "close_type_by_depth": cl}
        trained[name] = m
    sd6 = trained["STACKDCC2-big-D6"].state_dict()
    sd12 = trained["STACKDCC2-big-D12"].state_dict()
    w_eq = all(torch.equal(sd6[k], sd12[k]) for k in sd6)
    print(f"[P13D] D6 vs D12 weights identical after training: "
          f"{w_eq}", flush=True)
    cl6 = result["arms"]["STACKDCC2-big-D6"]["close_type_by_depth"]
    cl12 = result["arms"]["STACKDCC2-big-D12"]["close_type_by_depth"]
    cap_check = {
        "weights_identical_d6_d12": w_eq,
        "max_abs_close_diff_d1_6": max(abs(cl6[f"close_d{d}"] - cl12[f"close_d{d}"])
                                       for d in range(1, 7))}
    result["capacity_purity_checks"] = cap_check
    print("[P13D] capacity-purity checks: "
          f"{json.dumps(cap_check)}", flush=True)
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
