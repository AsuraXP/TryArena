# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 13c (cycle 56) — EXACT-DEPTH RANDOM-TYPE DYCK
CAPACITY FRONTIER. P13/P13b showed the deterministic FIXED-type
grammar is NOT a stack-essential OOD test: S(d) embeds the trained
S(3) verbatim at aligned boundaries, so OOD close-type is reachable
by positional pattern matching — TFMicro (attention) OOD close
d4/d6 = .85/.70 ABOVE both stack arms (.50/.51) and VETDCC
(.38/.29). L-DETERMINISTIC-POSITION-SHORTCUT. P13c restores
stack-essentiality by randomizing each node's TYPE while keeping
the FIXED EXACT-DEPTH shape: every open's type is a fresh coin
(no position can predict it) and every close must match its open's
type (stack top) — so closes are ONLY predictable through the
stack, in train AND OOD, while the shape guarantees the segment
nests to exactly depth d (clean capacity boundary: a capacity-D
stack overflows at the (D+1)-th nested open of EVERY segment at
eval depth d = D+1). Exact-match stays unmeasurable (~0: random
open types are not state-determined — the quantified P11 ceiling),
so per-position CLOSE-TYPE is the certifiable axis.

GRAMMAR: shape = P13's S(d) = a S(d-1) c_a b S(d-1) c_b (a=BRK,
c_a=BRK+2, b=BRK+1, c_b=BRK+3; |S(d)| = 2^(d+2) - 4; max nesting
exactly d). Types randomized: walk the shape, on each OPEN draw
t = randrange(2) and emit BRK+t; the matching close (Dyck-matched,
i.e. the close that pops this open) emits BRK+2+t. Result: same
length/depth as S(d), every type independent per node per segment.

Prior art (searched 2026-09-06; mechanism/grammar lineage in
P13 header): Suzgun et al. COLING 2020 — bounded-depth Dyck
generalizes across LENGTH, fixed-state nets are bounded-depth;
Hahn TACL 2020 — transformers fail Dyck depth-generalization
asymptotically; P11 (log.md C55) proved STOCHASTIC-type dyck
induces close-type extrapolation in capacity (STACKDCC2-big .925
d3) with NO capacity boundary visible (branch draws make deep
segments rare/never-fit); P13b (log.md C56) proved deterministic
types are position-shortcuttable. P13c is the missing cell: random
types (stack-forcing, per P11) x EXACT depth (capacity boundary,
per P13b's shape) — no NEW architecture mechanism, a data-grammar
variant of the already-cited line.

ARMS (identical to P13/P13b, seed-clean: ctor under per-arm
manual_seed(0), capacity = pure runtime buffer):
  STACKDCC2-big D6   (21,817p)   STACKDCC2-big D12 (21,817p)
  VETDCC-big         (21,257p)   TFMicro          (8,144p, d<=6)
TRAIN: depth 2 random-type pool, L=256, 256 streams seed 12345,
2000 steps seed 0. EVAL: close-type d1..d12 (PRIMARY metric) +
whole-segment exact d1..d6 (record only; ~0 by construction);
per-depth L (256 if |S(d)|+3 <= 256 else |S(d)|+16); adaptive
streams 8/4/2. TFMicro eval limited to d<=6 (sin-PE length
confound, P1 — the d3-6 L=256 window is TF's depth-generalization
claim region).

SHARP PREDICTIONS (falsifiable, logged per protocol):
  (a) sanity: in-train close-type d2 >= .90 for STACKDCC2-big
      (type-tracking IS induced by random types — else training
      failed, rerun at 2.5x budget before interpreting);
  (b) STACKDCC2-big D6: close-type HIGH (>= .85) at d3-6 (in
      capacity), COLLAPSE (~.5 or below) at d7-12 (overflow on
      every d7+ segment — exact depth makes the boundary clean);
  (c) STACKDCC2-big D12: close-type HIGH out to d11-12;
  (d) TFMicro close-type at d3-6 clearly BELOW the D6 stack arm at
      the SAME depths (no stack; Hahn pre-failure window);
  (e) VETDCC-big ~.5 or below from d3 on (type order needed; depth
      counter alone is content-free — L-DYCK-NEEDS-CONTENT-STACK).
  Falsification of (b)+(c) together (D6 and D12 both flat/high to
  d12, or both collapse at the same depth) = the D_STACK buffer is
  NOT the binding frontier -> capacity question answered no;
  (b)+(c) confirmed = CERTIFIABLE capacity frontier + a structural
  close-type win over TF in the deterministic-random regime.
Tag ARCH-VET-LM-P13C.
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


# ---- exact-depth random-type grammar ------------------------------
def det_shape(d):
    """Fixed P13 shape at depth d (type pattern ignored)."""
    if d <= 0:
        return []
    return ([0] + det_shape(d - 1) + [2]
            + [1] + det_shape(d - 1) + [3])


def rt_segment(rng, d):
    """Random-type segment BODY of exact depth d: walk the fixed
    shape; each open draws a fresh type; Dyck-matched closes emit
    the same type. Returns list of vocab ids."""
    shape = det_shape(d)
    body = []
    st = []                      # stack of open positions
    for i, k in enumerate(shape):
        if k in (0, 1):          # open
            t = rng.randrange(2)
            st.append(t)
            body.append(BRK + t)
        else:                    # close (k in 2,3)
            t = st.pop()         # always balanced by construction
            body.append(BRK + 2 + t)
    assert not st and len(body) == len(shape)
    return body


def seg_len(d):
    return 4 if d == 1 else (2 * seg_len(d - 1) + 4)


def gen_rt_stream(rng, L, depth):
    """Stream of exact-depth random-type segments at length L. A
    FRESH random-type segment is drawn per segment (inside the
    loop) — a segment generated once outside the loop would repeat
    one type pattern per stream (P13c launch-1 bug: 63 distinct
    streams, period-memorization shortcut; fixed). Mirrors P11's
    per-segment regeneration."""
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


def gen_rt_pool(n, L, seed):
    """Training pool: depth-2 random-type streams, all distinct."""
    prng = random.Random(seed)
    return [torch.tensor(gen_rt_stream(prng, L, 2)) for _ in range(n)]


# ---- eval: per-position close-type + whole-segment exact ----------
@torch.no_grad()
def rt_close_acc(model, n, depth, L):
    """Close type == stack top (random per node) -> only a stack
    tracker can score above chance. Scores closes of COMPLETED
    segments (close to depth 0 before stream end)."""
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
                    if BRK + 2 <= xl[g] < BRK + 4:   # close position
                        tot += 1
                        ok += int(int(pred[g - 1]) == xl[g])
                j = max(seg) + 1 if seg else j + 3
            else:
                j += 1
    return (round(ok / tot, 4) if tot else float("nan")), tot


@torch.no_grad()
def rt_exact(model, n, depth, L):
    """Whole-segment exact-match (record only; ~0 by construction
    because open types are random coins)."""
    model.eval()
    ok = tot = 0
    for i in range(n):
        x = torch.tensor(gen_rt_stream(
            random.Random(5000 + i * 31 + depth), L, depth)).unsqueeze(0)
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


if __name__ == "__main__":
    # grammar sanity: exact depth, balance, random types
    for d in range(1, 13):
        assert seg_len(d) == len(det_shape(d))
        rng = random.Random(1)
        b = rt_segment(rng, d)
        assert len(b) == len(det_shape(d))
        dep = 0
        for tok in b:
            dep += 1 if tok < BRK + 2 else -1
            assert dep >= 0
        assert dep == 0
    pool = gen_rt_pool(256, 256, 12345)
    distinct = len(set(tuple(p.tolist()) for p in pool))
    print(f"[P13C] random-type pool: {len(pool)} streams, "
          f"{distinct} distinct", flush=True)
    assert distinct > 100
    # type diversity check: in 20 depth-2 segments every (open pos)
    # must take both types sometimes -> stack is really forced
    rng = random.Random(7)
    first_open_types = set()
    for _ in range(200):
        seg = rt_segment(rng, 2)
        first_open_types.add(seg[0] - BRK)
    print(f"[P13C] S(2) first-open type set over 200 segs: "
          f"{sorted(first_open_types)} (expect {{0,1}})", flush=True)
    assert first_open_types == {0, 1}
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
    result = {"tag": "ARCH-VET-LM-P13C",
              "protocol": "EXACT-DEPTH RANDOM-TYPE dyck: fixed P13 "
                          "shape S(d) (nesting exactly d), per-node "
                          "type coins (close == its open's type: "
                          "stack-essential, no position shortcut); "
                          "train depth 2 random-type L=256 pool 256 "
                          "seed 12345, 2000 steps seed 0, ctor "
                          "under manual_seed(0); eval per-position "
                          "close-type d1-12 (primary) + exact d1-6 "
                          "(record; ~0 by construction); per-depth "
                          "L; TFMicro eval d<=6 only",
              "arms": {}}
    trained = {}
    for name, ctor in arm_specs:
        torch.manual_seed(0)
        m = ctor()
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_dyck(name, m, pool, 2000, 8)
        m.eval()
        cl, ex = {}, {}
        for d in range(1, 13):
            if name == "TFMicro" and d > 6:
                break
            L = L_by_depth[d]
            a, tot = rt_close_acc(m, n_streams(d), d, L)
            cl[f"close_d{d}"] = a
            print(f"  [{name}] d{d} L{L}: close-type {a} ({tot} pos)",
                  flush=True)
        for d in range(1, 7):
            if name == "TFMicro":
                break
            L = L_by_depth[d]
            a, tot = rt_exact(m, n_streams(d), d, L)
            ex[f"exact_d{d}"] = a
            print(f"  [{name}] d{d} L{L}: exact {a} ({tot} segs)",
                  flush=True)
        result["arms"][name] = {"params": n_params(m),
                                "loss_curve": hist,
                                "close_type_by_depth": cl,
                                "exact_match_d1_6": ex}
        trained[name] = m
    sd6 = trained["STACKDCC2-big-D6"].state_dict()
    sd12 = trained["STACKDCC2-big-D12"].state_dict()
    w_eq = all(torch.equal(sd6[k], sd12[k]) for k in sd6)
    print(f"[P13C] D6 vs D12 weights identical after training: "
          f"{w_eq}", flush=True)
    cl6 = result["arms"]["STACKDCC2-big-D6"]["close_type_by_depth"]
    cl12 = result["arms"]["STACKDCC2-big-D12"]["close_type_by_depth"]
    cap_check = {
        "weights_identical_d6_d12": w_eq,
        "max_abs_close_diff_d1_6": max(abs(cl6[f"close_d{d}"] - cl12[f"close_d{d}"])
                                       for d in range(1, 7))}
    result["capacity_purity_checks"] = cap_check
    print("[P13C] capacity-purity checks: "
          f"{json.dumps(cap_check)}", flush=True)
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
