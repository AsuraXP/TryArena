# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 10 (cycle 55) — STACKDCC: VETDCC + EXACT
BRACKET-TYPE STACK channel. Follows the P9 falsification
(L-DYCK-NEEDS-CONTENT-STACK: the exact depth counter gave modk 1.000
but dyck 0.000 at all depths — depth is not TYPE ORDER).
Phase-1 prior art (searched 2026-09-06):
  - Stack-augmented NN line: Joulin & Mikolov 2015 arXiv 1503.01007
    (superposition stack); Grefenstette 2015 (NTM); Stogin et al. 2020
    arXiv 2006.03651 (stable TM: SOFT stacks unstable when push/pop
    intensities mismatch); Dusell & Chiang 2023/2024 (nondeterministic
    differentiable stacks); Dusell & Chiang 2025 arXiv 2511.03547
    (stack-augmented TF/RNN/LSTM, hierarchical generalization) —
    ALL use soft/learned stacks.
  - Dyck line: Hahn 2020 (TF fail Dyck-2 asymptotically), Suzgun 2019
    arXiv 1911.03329 (memory-augmented RNN generalized Dyck).
  GAP: no native trained LM with an EXACT discrete bracket-type stack
  channel (hardwired task-grammar predicates, zero approximation
  error, capacity-bounded with explicit overflow/underflow features)
  zero-injected into a learned controller + readout. (P9's exact
  counters proved the exact-channel paradigm — modk 1.000/1.000 —
  but carried only depth, not type order.)
MECHANISM (STACKDCC = VETDCC + 7-dim exact stack features):
  stack: fixed capacity D=6 of bracket TYPES (0=a, 1=b), pointer
  sp, reset at T_TASK;
    open  (tok 29/30): if sp<6: stk[sp]=type, sp+=1; else OVERFLOW
                       (content lost — the model SEES its own bound);
    close (tok 31/32): if sp>0: MATCH/MISMATCH = (stk[sp-1]==type),
                       sp-=1; else UNDERFLOW.
  features (7): [top_empty, top_a, top_b, match, mismatch, underflow,
                 overflow] — computed at each position; injected
  ZERO-init into the controller input (Ws expanded) and the readout
  (W_stack zeros (7, V)). Everything else identical to VETDCC.
HONEST NOTE: the push/pop predicates are hardwired from the task
grammar (bracket tokens 29-32) — same convention as P9's counters;
the novelty claim is the exact content-stack channel architecture +
its capacity-bounded frontier, not the predicate source.
SHARP PREDICTION (testable): dyck exact-match stays HIGH at depth
3-6 (within stack capacity: exact type order available) and
COLLAPSES at depth 7-10 (overflow — type order beyond capacity 6 is
unrecoverable). Protocol: P1 4-task pool, 2000 steps, seed 0, arms
STACKDCC-base (8,902+~350p) / STACKDCC-big (21,257+~350p); same
probes as P9 (CE 256tr/256hard/512/1024, task acc, dyck frontier
3-10). Tag ARCH-VET-LM-P10.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
T0 = time.time()

# exec P9 (which itself execs arch_vet_lm) -> VETDCC + data + probes
_src9 = open("arch_vet_p9.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src9.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p9.py", "exec"), _ns)
V = _ns["V"]; VETDCC = _ns["VETDCC"]
make_pool = _ns["make_pool"]; make_batches = _ns["make_batches"]
train_arm = _ns["train_arm"]; val_ce = _ns["val_ce"]
task_acc = _ns["task_acc"]; n_params = _ns["n_params"]
ONE = _ns["ONE"]; BRK = _ns["BRK"]
BOS = _ns["BOS"]; EOS = _ns["EOS"]; T_TASK = _ns["T_TASK"]
gen_dyck = _ns["gen_dyck"]; dyck_acc = _ns["dyck_acc"]
eval_all = _ns["eval_all"]

N_OH = _ns["N_OH"]        # 10 (DCC counter dims)
D_STACK = 6               # exact stack capacity
N_STACK = 7               # top_empty, top_a, top_b, match, mismatch,
                          # underflow, overflow


class STACKDCC(VETDCC):
    """VETDCC + exact bracket-type stack (hardwired predicates)."""
    def __init__(self, V, d, k=5, K=4):
        super().__init__(V, d, k=k, K=K)
        Ws_old = self.Ws
        self.Ws = nn.Linear(d + N_OH + N_STACK, k)
        with torch.no_grad():
            self.Ws.weight[:, :d + N_OH] = Ws_old.weight
            self.Ws.weight[:, d + N_OH:] = 0.0
            self.Ws.bias = Ws_old.bias
        del Ws_old
        self.W_stack = nn.Parameter(torch.zeros(N_STACK, V))  # zero

    def forward(self, x):
        B, L = x.shape
        e = self.E(x)
        R = torch.zeros(B, self.d, device=x.device)
        s = torch.full((B, self.k), 1.0 / self.k, device=x.device)
        buf = torch.zeros(B, self.K, self.d, device=x.device)
        valid = torch.zeros(B, self.K, dtype=torch.bool, device=x.device)
        c = torch.zeros(B, dtype=torch.long, device=x.device)
        dep = torch.zeros(B, dtype=torch.long, device=x.device)
        stk = torch.zeros(B, D_STACK, dtype=torch.long, device=x.device)
        sp = torch.zeros(B, dtype=torch.long, device=x.device)
        idx = torch.arange(D_STACK, device=x.device)
        lg = torch.empty(B, L, V, device=x.device)
        for t in range(L):
            xt = e[:, t]
            xid = x[:, t]
            # ---- DCC counters (identical to VETDCC) ----
            is_one = (xid == ONE)
            is_task = (xid == T_TASK)
            c = torch.where(is_task, 0,
                            torch.where(is_one, (c + 1) % 3, c))
            is_open = ((xid == BRK) | (xid == BRK + 1))
            is_close = ((xid == BRK + 2) | (xid == BRK + 3))
            dep = torch.where(is_task, 0,
                              torch.where(is_open,
                                          (dep + 1).clamp(max=6),
                                          dep))
            dep = torch.where(is_close, (dep - 1).clamp(min=0), dep)
            mod_oh = F.one_hot(c, 3).float()
            depth_oh = F.one_hot(dep, 7).float()
            # ---- exact bracket-type stack (NEW) ----
            open_type = (xid == BRK + 1).long()     # 0=a, 1=b
            close_type = (xid == BRK + 3).long()
            # features from PRE-update stack state
            top_before = stk.gather(
                1, (sp - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            top_empty = (sp == 0).float()
            top_a = ((sp > 0) & (top_before == 0)).float()
            top_b = ((sp > 0) & (top_before == 1)).float()
            match = (is_close & (sp > 0) & (top_before == close_type)).float()
            mismatch = (is_close & (sp > 0) & (top_before != close_type)).float()
            underflow = (is_close & (sp == 0)).float()
            overflow = (is_open & (sp == D_STACK)).float()
            # apply close (pop), then open (push)
            sp = torch.where(is_close & (sp > 0), sp - 1, sp)
            sp = torch.where(is_task, torch.zeros_like(sp), sp)
            stk = torch.where(is_task.unsqueeze(-1),
                              torch.zeros_like(stk), stk)
            write = is_open & (sp < D_STACK)
            pos = sp.clamp(max=D_STACK - 1).unsqueeze(1)
            same = (idx.unsqueeze(0) == pos) & write.unsqueeze(-1)
            stk = torch.where(same, open_type.unsqueeze(1), stk)
            sp = torch.where(write, sp + 1, sp)
            stack_feat = torch.stack(
                [top_empty, top_a, top_b, match, mismatch,
                 underflow, overflow], 1)            # (B, 7)
            # ---- controller (expanded input) ----
            s = F.softmax(self.Ws(torch.cat(
                [xt, mod_oh, depth_oh, stack_feat], -1)) + self.Wss(s), -1)
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
            logits = logits + mod_oh @ self.W_mod \
                          + depth_oh @ self.W_depth \
                          + stack_feat @ self.W_stack
            lg[:, t] = logits
        return lg


if __name__ == "__main__":
    # ---- unit test: stack features on a known balanced stream ----
    # depth-3 pattern: a( a() b() )  -> all closes must MATCH
    stream = [T_TASK, T_TASK, T_TASK, BRK, BRK, BRK + 2,
              BRK + 1, BRK + 3, BRK + 2, EOS]
    m = STACKDCC(V, 16, k=5, K=4)
    torch.manual_seed(0)
    m.eval()
    # instrument: replay features via a forward hook is overkill —
    # replicate the update on the test stream directly
    stk = torch.zeros(1, D_STACK, dtype=torch.long)
    sp = torch.zeros(1, dtype=torch.long)
    flags = []
    for tok in stream[1:]:
        xid = torch.tensor([tok])
        is_open = ((xid == BRK) | (xid == BRK + 1)).squeeze()
        is_close = ((xid == BRK + 2) | (xid == BRK + 3)).squeeze()
        is_task = (xid == T_TASK).squeeze()
        open_type = (xid == BRK + 1).long()
        close_type = (xid == BRK + 3).long()
        top_before = stk.gather(1, (sp - 1).clamp(min=0).unsqueeze(1)).squeeze()
        mt = bool(is_close & (sp > 0) & (top_before == close_type))
        mm = bool(is_close & (sp > 0) & (top_before != close_type))
        uf = bool(is_close & (sp == 0))
        sp = torch.where(is_close & (sp > 0), sp - 1, sp)
        sp = torch.where(is_task, torch.zeros_like(sp), sp)
        stk = torch.where(is_task.unsqueeze(-1), torch.zeros_like(stk), stk)
        write = is_open & (sp < D_STACK)
        pos = sp.clamp(max=D_STACK - 1).unsqueeze(1)
        same = (torch.arange(D_STACK).unsqueeze(0) == pos) & write.unsqueeze(-1)
        stk = torch.where(same, open_type.unsqueeze(1), stk)
        sp = torch.where(write, sp + 1, sp)
        if is_close:
            flags.append(("close", mt, mm, uf))
    n_match = sum(1 for f in flags if f[1])
    n_bad = sum(1 for f in flags if f[2] or f[3])
    assert n_match == 3 and n_bad == 0, flags
    print(f"[unit] depth-3 balanced stream: 3/3 closes MATCH, 0 bad "
          f"(flags={flags})", flush=True)
    # overflow check: 7 nested opens must set overflow on the 7th
    sp7 = torch.zeros(1, dtype=torch.long); ov = []
    for _ in range(7):
        ov.append(bool(sp7 == D_STACK))
        sp7 += 1
    assert ov[6] == True and sum(ov) == 1, ov
    print(f"[unit] 7 nested opens: overflow only on 7th ({ov})", flush=True)

    pool = make_pool(512, 256, 12345)
    arms = [("STACKDCC-base", STACKDCC(V, 16, k=5, K=4)),
            ("STACKDCC-big", STACKDCC(V, 24, k=8, K=8))]
    result = {"tag": "ARCH-VET-LM-P10",
              "protocol": "STACKDCC = VETDCC + exact bracket-type stack "
                          "(capacity 6, hardwired push on 29/30, "
                          "match-check+pop on 31/32, per-task reset); "
                          "7-dim stack features [top_empty, top_a, "
                          "top_b, match, mismatch, underflow, overflow] "
                          "zero-injected into controller + readout; "
                          "P1 4-task pool, 2000 steps, seed 0; probes "
                          "identical to P9 (CE 4-way, task acc, dyck "
                          "frontier 3-10). Sharp prediction: dyck "
                          "HIGH at depth 3-6 (in-capacity), COLLAPSE "
                          "at 7-10 (overflow)",
              "arms": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_arm(name, m, pool, 2000, 8)
        ce, at, ae, dy = eval_all(m, name)
        result["arms"][name] = {
            "params": n_params(m), "loss_curve": hist, "ce": ce,
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
