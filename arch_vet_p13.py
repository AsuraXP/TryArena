# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 13 (cycle 56) — DETERMINISTIC DYCK + CAPACITY
FRONTIER. P11/P12 (stochastic dyck): STACKDCC2-big close-type
.925/.852 at d3/d4 vs TFMicro .678/.581 — a real stack win — but
whole-segment exact-match was 0.0 for EVERY arm because the eval
grammar is STOCHASTIC (open-type coin flips + 30% branch draws are
not state-determined; ceiling quantified in P11). P13 replaces the
grammar with a DETERMINISTIC one where every token IS
state-determined, so exact-match is a MEASURABLE ~1.0 bar instead
of a construction-impossible 0.0.

GRAMMAR (deterministic, fixed-type double-branch; a = BRK,
c_a = BRK+2, b = BRK+1, c_b = BRK+3):
    emit(d) = [a] emit(d-1) [c_a] [b] emit(d-1) [c_b],  emit(0) = []
Segment length |S(d)| = 2^(d+2) - 4  (d1=4, d6=252, d7=508, d12=16380).
Max nesting depth of S(d) == d -> a capacity-D stack user is
exact in capacity (d <= D) and must fail at d = D+1 (overflow at
the (D+1)-th nested open). The close TYPE at every close position
is exactly the stack top (deterministic map), so bracket-level
close-type ~= 1.0 in-capacity is certifiable by construction for a
perfect stack user.

Prior art (searched 2026-09-06): Suzgun et al. COLING 2020 "On the
Practical Ability of RNNs to Recognize Dyck Languages" — bounded-
depth Dyck generalizes across LENGTH; precision/expressivity of
fixed-state nets over Dyck is bounded-depth (their stack-extraction
probes = our exact-channel design intent); Hahn TACL 2020 —
transformers fail Dyck-2 depth-generalization ASYMPTOTICALLY (d3-6
micro is the pre-failure window being tested); soft-stack line
(Joulin-Mikolov 2015 arXiv 1503.01007; Stogin 2020 arXiv
2006.03651; Dusell-Chiang 2025 arXiv 2511.03547) — all learned/soft,
no EXACT discrete capacity-parameterized content stack in a learned
controller at micro scale. Gap: capacity frontier (D=6 vs D=12 at
equal params) on a deterministic stack-essential grammar, vs the
attention control.

ARMS (train depth 2, deterministic pool, L=256 pool 256 seed 12345,
2000 steps seed 0 — P11 protocol, deterministic):
  STACKDCC2-big  D=6   (21,817p; p11 class verbatim)
  STACKDCC2-big  D=12  (SAME params — capacity is a buffer size, not
                        a parameter; class exec'd from p11 source with
                        D_STACK patched to 12)
  VETDCC-big           (21,257p; depth counter, NO content stack)
  TFMicro              (8,144p; the P1/P12 attention control)
EVAL: whole-segment exact-match at d3..d12 + per-position close-type
at d3/d4/d6/d8/d10/12. Streams use per-depth L (d<=6: L=256 in train
length; d7+: L = |S(d)| + 16 so a full segment fits — VET arms are
length-invariant (P1/P5), TF is NOT evaluated beyond d6 (sin-PE
extrapolation confound, P1 CE-collapse) — TF's window is d3-6 at
L=256, the depth-generalization claim window).
SHARP PREDICTIONS (handover cycle-56 plan):
  (a) STACKDCC2-big D=6: exact-match ~1.0 at d3-6 (in capacity),
      COLLAPSE at d7-12 (overflow at 7th nested open);
  (b) STACKDCC2-big D=12: frontier extends to ~d11-12 (in capacity),
      collapse at d13+;
  (c) TFMicro < STACKDCC2 at d3-6 (local attention cannot track
      multi-level type order even in-capacity);
  (d) VETDCC-big (depth counter only) well below both stack arms at
      d3-6 (L-DYCK-NEEDS-CONTENT-STACK carries to the deterministic
      grammar).
Falsifications logged honestly (per program protocol): any arm
beating its predicted frontier becomes the next attack.
Tag ARCH-VET-LM-P13.
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
# BOS/EOS/MODS live in arch_vet_lm's ns (p11 does not re-export them);
# both namespaces exec the SAME arch_vet_lm source, so the constants match.
_src_lm = open("arch_vet_lm.py", encoding="utf-8").read()
_ns_lm = {}
exec(compile(_src_lm.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns_lm)
TFMicro = _ns_lm["TFMicro"]
BOS = _ns_lm["BOS"]; EOS = _ns_lm["EOS"]; MODS = _ns_lm["MODS"]
assert (T_TASK, BRK) == (_ns_lm["T_TASK"], _ns_lm["BRK"])

# ---- D=12 variant: p11's STACKDCC2 forward reads the module-global
# D_STACK at call time; capacity is a buffer size, NOT a parameter,
# so param count is identical to D=6 (the pure capacity comparison).
_ns12 = {}
src12 = _src11.replace('D_STACK = _ns["D_STACK"]',
                       'D_STACK = 12')
exec(compile(src12.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p11_D12.py", "exec"), _ns12)
STACKDCC2_D12 = _ns12["STACKDCC2"]
assert STACKDCC2_D12 is not STACKDCC2


# ---- deterministic grammar ---------------------------------------
def det_emit(d):
    """Deterministic fixed-type double-branch dyck segment body."""
    if d <= 0:
        return []
    return ([BRK] + det_emit(d - 1) + [BRK + 2]
            + [BRK + 1] + det_emit(d - 1) + [BRK + 3])


def det_seg_len(d):
    return 4 if d == 1 else (2 * det_seg_len(d - 1) + 4)


def gen_det_stream(rng, L, depth):
    """One stream of deterministic S(depth) segments at length L."""
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


def gen_det_pool(n, L, seed):
    prng = random.Random(seed)
    return [torch.tensor(gen_det_stream(prng, L, 2)) for _ in range(n)]


# ---- eval: exact-match + per-position close type ------------------
@torch.no_grad()
def det_exact(model, n, depth, L):
    """Exact-match over full deterministic segments at fixed depth.
    Teacher-forced (model sees the gold stream; per-position argmax
    vs gold) — identical convention to dyck_acc (P9-P12)."""
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
    """Per-position CLOSE-type accuracy (close must match the stack
    top = deterministic map for a perfect stack user)."""
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
    pool = gen_det_pool(256, 256, 12345)
    # sanity: pool must contain ONLY depth-2 deterministic segments
    s2 = det_emit(2)
    assert det_seg_len(2) == len(s2) == 12
    for d in range(1, 13):
        assert det_seg_len(d) == len(det_emit(d))
    print(f"[P13] deterministic S(2) len={len(s2)} "
          f"(+3 T_TASK) in L=256 pool", flush=True)
    # d<=6: S(d)+3 <= 255 fits L=256 exactly (train length).
    # d>=7: need L = |S(d)| + 16 so one full segment + padding fits.
    L_by_depth = {d: (256 if det_seg_len(d) <= 252
                      else det_seg_len(d) + 16) for d in range(1, 13)}
    # adaptive stream count (long streams are the slow part for the
    # sequential VET forward): 8 streams at train length, 4 at
    # mid length, 2 at d11-12 (16384-token streams).
    def n_streams(d):
        L = L_by_depth[d]
        if L <= 256:
            return 8
        if L <= 4096:
            return 4
        return 2

    arms = [("STACKDCC2-big-D6", STACKDCC2(V, 24, k=8, K=8)),
            ("STACKDCC2-big-D12", STACKDCC2_D12(V, 24, k=8, K=8)),
            ("VETDCC-big", VETDCC(V, 24, k=8, K=8)),
            ("TFMicro", TFMicro(V, 16))]
    result = {"tag": "ARCH-VET-LM-P13",
              "protocol": "DETERMINISTIC dyck grammar emit(d) = "
                          "[a]emit(d-1)[c_a][b]emit(d-1)[c_b]; train "
                          "depth 2, L=256 pool 256 seed 12345, 2000 "
                          "steps seed 0 (P11 budget); eval "
                          "whole-segment exact-match d3-12 + "
                          "per-position close-type d3-12; per-depth "
                          "L=256 for d<=6 (in train length), "
                          "L=|S(d)|+16 for d>=7 (VET length-"
                          "invariant); TFMicro eval limited to d<=6 "
                          "(sin-PE extrapolation confound, P1); "
                          "capacity is a buffer size, D6==D12 params",
              "arms": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_dyck(name, m, pool, 2000, 8)
        m.eval()
        ex, cl = {}, {}
        for d in range(3, 13):
            if name == "TFMicro" and d > 6:
                break
            L = L_by_depth[d]
            a, tot = det_exact(m, n_streams(d), d, L)
            ex[f"exact_d{d}"] = a
            print(f"  [{name}] d{d} L{L}: exact {a} ({tot} segs)",
                  flush=True)
        for d in (3, 4, 6, 8, 10, 12):
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
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
