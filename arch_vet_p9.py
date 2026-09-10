# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 9 (cycle 53) — VETDCC: VET + DETERMINISTIC
COUNTER CHANNELS. Phase-1 prior art (searched 2026-08-29):
  - Counter-language LM line: Hahn 2020 (Transformers fail Dyck-2
    asymptotically), Bhattamishra et al. 2020, Strobl et al. 2024
    (Transformers learn Dyck-1/Shuffle-Dyck = implicit counter
    simulation), Suzgun et al. 2019 arXiv 1911.03329 (memory-
    augmented RNNs, generalized Dyck), emergent stack probes
    arXiv 2502.01432 (stack depth EMERGENT in trained Transformers),
    counting in small TFs Behrens/Biggio/Zdeborova ICML 2025.
  - Stack-augmented RNN line: Joulin & Mikolov 2015 arXiv 1503.01007,
    Grefenstette 2015, Stogin 2006.03651 — all CONTINUOUS/soft
    stacks.
  - Automata line: Weiss et al. 2018 ICML (extract DFAs from trained
    RNNs — post-hoc), N-FSM arXiv 2505.11694 (mod-n counter DFAs used
    in ANALYSIS only, not as trained components).
  GAP: no native trained LM with EXACT discrete counter channels as
  first-class architectural components feeding a learned controller +
  readout. DCC ports the certified VET+S counter organ (C25a /
  C47-C48 controller line: mod-k cycles, clamped depth counters)
  into the LM: zero approximation error, ~0 added params.
MECHANISM (VETDCC, base 8,372p + ~530p):
  mod channel:  c_t = (c_{t-1}+1) mod 3  iff x_t == ONE (tok 21);
  depth channel: d_t = clamp(d_{t-1}+1, 0, D) if open bracket (29,30)
                 d_t = max(d_{t-1}-1, 0)      if close bracket (31,32)
                 else unchanged.  D=6 (train depth 2, eval 3-4,
                 frontier probe 5-6 in-clamp / 7-10 out-of-clamp).
  BOTH channels reset to 0 at each T_TASK (tok 4) so that at the
  A-marker they hold the CURRENT task's count (a global counter
  would accumulate across tasks — caught by unit test).
  NOTE the actual vocab layout in arch_vet_lm.py (BRK=29..32,
  TRACK=21..28 overlaps ONE/MANS by design — tasks are
  shape-disjoint; fillers = MODS+0..7 = 13..20).
  Injection: controller input [x; mod_oh(3); depth_oh(7)] (Ws
  expanded, counter block ZERO-init); readout += W_mod @ mod_oh +
  W_depth @ depth_oh (ZERO-init). Hardwired token predicates = task
  grammar (logged honestly: novelty claim = the channel
  architecture, not the predicate source).
SHARP PREDICTION (testable): dyck exact-match accuracy stays high
up to depth <= D (in-clamp) and degrades beyond depth > D (out-
of-clamp) — the architecture's own bound predicts its own frontier.
Protocol: P1 4-task, 2000 steps, seed 0, arms VETDCC-base /
VETDCC-big (k8 d24 K8); MAMBA reference cited from P1 log
(256-hard 2.897 / 1024 1.378 / modk-ev .423 / dyck-ev 0.000).
Plus DYCK-DEPTH FRONTIER probe: train depth 2 -> eval depth
3-4 (P1 eval) / 5-6 (in-clamp) / 7-10 (out-of-clamp).
Tag ARCH-VET-LM-P9.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

_src = open("arch_vet_lm.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns)
V = _ns["V"]; VETLM = _ns["VETLM"]
make_pool = _ns["make_pool"]; make_batches = _ns["make_batches"]
train_arm = _ns["train_arm"]; val_ce = _ns["val_ce"]
task_acc = _ns["task_acc"]; n_params = _ns["n_params"]
ONE = _ns["ONE"]; BRK = _ns["BRK"]
BOS = _ns["BOS"]; EOS = _ns["EOS"]; T_TASK = _ns["T_TASK"]; A = _ns["A"]
MODS = _ns["MODS"]

M_MOD = 3          # mod-3 cycle
D_CLAMP = 6        # depth clamp
N_OH = M_MOD + (D_CLAMP + 1)   # 10 one-hot dims


class VETDCC(VETLM):
    """VETLM + exact mod-3 counter + clamped depth counter."""
    def __init__(self, V, d, k=5, K=4):
        super().__init__(V, d, k=k, K=K)
        Ws_old = self.Ws
        self.Ws = nn.Linear(d + N_OH, k)
        with torch.no_grad():
            self.Ws.weight[:, :d] = Ws_old.weight
            self.Ws.weight[:, d:] = 0.0
            self.Ws.bias = Ws_old.bias
        del Ws_old
        self.W_mod = nn.Parameter(torch.zeros(M_MOD, V))      # zero
        self.W_depth = nn.Parameter(torch.zeros(D_CLAMP + 1, V))  # zero

    def forward(self, x):
        B, L = x.shape
        e = self.E(x)
        R = torch.zeros(B, self.d, device=x.device)
        s = torch.full((B, self.k), 1.0 / self.k, device=x.device)
        buf = torch.zeros(B, self.K, self.d, device=x.device)
        valid = torch.zeros(B, self.K, dtype=torch.bool, device=x.device)
        c = torch.zeros(B, dtype=torch.long, device=x.device)    # mod
        dep = torch.zeros(B, dtype=torch.long, device=x.device)  # clamp
        lg = torch.empty(B, L, V, device=x.device)
        for t in range(L):
            xt = e[:, t]
            xid = x[:, t]
            # ---- DCC update (exact integer arithmetic) ----
            # per-task reset: counters restart at each task start,
            # so at the A-marker they hold the CURRENT task's count
            is_one = (xid == ONE)
            is_task = (xid == T_TASK)
            c = torch.where(is_task, 0,
                            torch.where(is_one, (c + 1) % M_MOD, c))
            is_open = ((xid == BRK) | (xid == BRK + 1))
            is_close = ((xid == BRK + 2) | (xid == BRK + 3))
            dep = torch.where(is_task, 0,
                              torch.where(is_open,
                                          (dep + 1).clamp(max=D_CLAMP),
                                          dep))
            dep = torch.where(is_close, (dep - 1).clamp(min=0), dep)
            mod_oh = F.one_hot(c, M_MOD).float()
            depth_oh = F.one_hot(dep, D_CLAMP + 1).float()
            # ---- controller (expanded input) ----
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1))
                          + self.Wss(s), -1)
            a = (s.unsqueeze(-1)
                 * torch.exp(-F.softplus(self.Alog))).sum(1)
            w = torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            R = a * R + w
            g = torch.sigmoid(self.Wg(torch.cat([s, xt], -1)))
            push = (g > 0.5) + (g - g.detach())
            buf = torch.roll(buf, 1, dims=1)
            buf[:, 0] = xt * push
            valid = torch.roll(valid, 1, dims=1)
            valid[:, 0] = (g > 0.5).squeeze(-1)
            y = self.Wo(R + xt)
            feat = torch.stack(
                [buf[:, j] for j in range(self.K)] + [xt], 1)
            logits = self.head(y)
            logits = logits + torch.einsum(
                "bk,bjd,kdv->bv", s, feat, self.M)
            sel = torch.zeros(B, self.K + 1, device=x.device)
            for j in range(self.K):
                newer = torch.zeros(B, device=x.device) + sum(
                    valid[:, i].float() for i in range(j))
                sel[:, j] = valid[:, j].float() * (newer == 0).float()
            sel[:, self.K] = 1.0
            logits = logits + torch.einsum("bs,ksv->bv", sel, self.T)
            # ---- DCC readout injection (zero-init) ----
            logits = logits + mod_oh @ self.W_mod \
                          + depth_oh @ self.W_depth
            lg[:, t] = logits
        return lg


# ---- dyck-depth frontier (single task, parameterized depth) ----
def gen_dyck(rng, L, depth):
    def emit(d):
        if d == 0:
            return []
        t = rng.randrange(2)
        if rng.random() < 0.7:
            return [BRK + t] + emit(d - 1) + [BRK + 2 + t]
        return ([BRK + t] + emit(d - 1) + [BRK + 2 + t]
                + [BRK + 1 - t] + emit(d - 1) + [BRK + 3 - t])
    x = [BOS]
    while len(x) < L:
        room = L - len(x)
        seg = None
        for _ in range(6):
            cand = [T_TASK, T_TASK, T_TASK] + emit(depth)
            if len(cand) <= room:
                seg = cand
                break
        if seg is None:
            x += [rng.randrange(8) + MODS] * room
            break
        x += seg
    x = x[:L]
    x.append(EOS)
    return x


@torch.no_grad()
def dyck_acc(model, n, depth):
    """Exact-match accuracy on full dyck segments at fixed depth."""
    model.eval()
    L = 256
    ok = tot = 0
    for i in range(n):
        x = torch.tensor(gen_dyck(random.Random(5000 + i), L,
                                  depth)).unsqueeze(0)
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
                if seg:
                    tot += 1
                    ok += int(all(int(pred[g - 1]) == xl[g]
                                  for g in seg))
                j = max(seg) + 1 if seg else j + 3
            else:
                j += 1
    return (ok / tot) if tot else float("nan"), tot


def eval_all(m, name):
    m.eval()
    rng = random.Random(999);  vtr = make_batches(8, 256, rng)
    rh  = random.Random(777);  vha = make_batches(8, 256, rh, hard=True)
    r5  = random.Random(31337); v512 = make_batches(2, 512, r5)
    r10 = random.Random(31415); v1024 = make_batches(2, 1024, r10)
    ce = {"256_train": val_ce(m, vtr, 256),
          "256_hard": val_ce(m, vha, 256),
          "512": val_ce(m, v512, 512),
          "1024": val_ce(m, v1024, 1024)}
    at = task_acc(m, 24, 256, random.Random(555), hard=False)
    ae = task_acc(m, 24, 256, random.Random(666), hard=True)
    print(f"[{name}] ce={ce}", flush=True)
    print(f"[{name}] acc(train)={{ {', '.join(f'{k}: {v:.3f}' for k, v in at.items())} }}".replace("{{", "{").replace("}}", "}"), flush=True)
    print(f"[{name}] acc(eval)={{ {', '.join(f'{k}: {v:.3f}' for k, v in ae.items())} }}".replace("{{", "{").replace("}}", "}"), flush=True)
    dy = {}
    for d in (3, 4, 5, 6, 7, 8, 9, 10):
        a, tot = dyck_acc(m, 16, d)
        dy[f"depth{d}"] = round(a, 4)
        print(f"  [{name}] dyck depth {d}: {a:.3f} ({tot} segs)", flush=True)
    return ce, at, ae, dy


if __name__ == "__main__":
    pool = make_pool(512, 256, 12345)
    arms = [("VETDCC-base", VETDCC(V, 16, k=5, K=4)),
            ("VETDCC-big", VETDCC(V, 24, k=8, K=8))]
    result = {"tag": "ARCH-VET-LM-P9",
              "protocol": "VETDCC = VETLM + exact mod-3 counter (ONE "
                          "tok 21) + clamped depth counter D=6 "
                          "(bracket toks 29-32), BOTH reset at "
                          "T_TASK (tok 4); counter one-hots (10 dims) into "
                          "controller input (Ws expanded, counter block "
                          "zero-init) and readout (zero-init W_mod/"
                          "W_depth); P1 4-task 2000 steps seed 0; "
                          "dyck-depth frontier 3-4 (P1 eval) / 5-6 "
                          "(in-clamp) / 7-10 (out-of-clamp); MAMBA "
                          "reference cited from P1 (256-hard 2.897, "
                          "1024 1.378, modk-ev .423, dyck-ev 0.000)",
              "arms": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_arm(name, m, pool, 2000, 8)
        ce, at, ae, dy = eval_all(m, name)
        result["arms"][name] = {
            "params": n_params(m), "loss_curve": hist,
            "ce": ce,
            "acc_train_interval": {k: round(v, 4) for k, v in at.items()},
            "acc_eval_interval": {k: round(v, 4) for k, v in ae.items()},
            "dyck_depth_frontier": dy,
            "len_ratio_1024_over_256hard":
                round(ce["1024"] / max(1e-9, ce["256_hard"]), 3)}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
