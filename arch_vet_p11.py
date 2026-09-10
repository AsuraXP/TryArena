# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-1 PHASE 11 (cycle 55) — DYCK SINGLE-TASK + STACKDCC-v2.
Follows the P10 double falsification (exact depth counter [P9] AND
exact type stack [P10] both gave dyck 0.000 at all depths, while
dyck TRAIN acc is only .12-.22 for EVERY arm in the program — the
4-task mixed stream at 2000 steps never gave dyck enough budget).
P11 changes the TASK PROTOCOL, not the architecture claim:
  (1) SINGLE-TASK stochastic-dyck stream (same emit() grammar as
      P9 gen_dyck), train depth 2, L=256, pool 256 (seed 12345),
      2000 steps, seed 0 — the protocol that discriminated P4
      (track frontier) and P7 (DIV fit).
  (2) STACKDCC-v2 = P10's exact stack + POST-UPDATE top features
      (top_after_empty/a/b): P10 injected only the PRE-top at
      position t, but the prediction at t targets t+1 which needs
      the stack state AFTER consuming t — a one-position latency.
      N_STACK = 10: [pre: top_empty, top_a, top_b, match, mismatch,
      underflow, overflow] + [post: ta_empty, ta_a, ta_b].
  (3) Per-POSITION bracket accuracy (fraction of bracket tokens
      predicted correctly, by position class: open / close):
      separates STACK USE (close must match top type — deterministic
      given the stack) from the STOCHASTIC GRAMMAR CEILING (the
      30% double-branch draw is not determined by any state —
      greedy argmax cannot exceed ~70% at those positions, so
      whole-segment exact-match is bounded well below 1.0 even for
      a perfect stack user).
ARMS (6): VETbase / VETDCC-base / STACKDCC2-base / VETDCC-big /
STACKDCC2-big / MAMBA — base 8.4-9.3k p, big 21-22k p.
SHARP PREDICTIONS:
  (a) single-task dyck TRAIN acc >> .22 for all arms (budget fix);
  (b) STACKDCC2 closes match their top type at near-perfect rate
      at depth 3-6 (per-position close acc), beating VETbase/VETDCC
      which have no type order;
  (c) whole-segment exact-match at 3-6 stays low for ALL arms
      (coin-flip ceiling) — if (b) holds with (c), the ceiling is
      PROVEN to be the grammar, not the architecture.
Tag ARCH-VET-LM-P11.
"""
import json, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.manual_seed(0)
T0 = time.time()

_src10 = open("arch_vet_p10.py", encoding="utf-8").read()
_ns = {}
exec(compile(_src10.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_p10.py", "exec"), _ns)
V = _ns["V"]; VETDCC = _ns["VETDCC"]
make_pool = _ns["make_pool"]; make_batches = _ns["make_batches"]
train_arm = _ns["train_arm"]; val_ce = _ns["val_ce"]
task_acc = _ns["task_acc"]; n_params = _ns["n_params"]
ONE = _ns["ONE"]; BRK = _ns["BRK"]
T_TASK = _ns["T_TASK"]
gen_dyck = _ns["gen_dyck"]; dyck_acc = _ns["dyck_acc"]
# VETLM/MambaMicro live in arch_vet_lm (p10 re-execs it internally but
# only re-exports V/VETDCC) — exec it directly for the base arms:
_src_lm = open("arch_vet_lm.py", encoding="utf-8").read()
_ns_lm = {}
exec(compile(_src_lm.rsplit('\nif __name__ == "__main__":', 1)[0],
             "arch_vet_lm.py", "exec"), _ns_lm)
VETLM = _ns_lm["VETLM"]; MambaMicro = _ns_lm["MambaMicro"]
assert _ns_lm["V"] == V

N_OH = _ns["N_OH"]
D_STACK = _ns["D_STACK"]
N_STACK = 10  # 7 pre + 3 post


class STACKDCC2(VETDCC):
    """P10 stack + post-update top features (fixes the latency)."""
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
            is_one = (xid == ONE)
            is_task = (xid == T_TASK)
            c = torch.where(is_task, 0,
                            torch.where(is_one, (c + 1) % 3, c))
            is_open = ((xid == BRK) | (xid == BRK + 1))
            is_close = ((xid == BRK + 2) | (xid == BRK + 3))
            dep = torch.where(is_task, 0,
                              torch.where(is_open,
                                          (dep + 1).clamp(max=6), dep))
            dep = torch.where(is_close, (dep - 1).clamp(min=0), dep)
            mod_oh = F.one_hot(c, 3).float()
            depth_oh = F.one_hot(dep, 7).float()
            # ---- exact stack: PRE features ----
            open_type = (xid == BRK + 1).long()
            close_type = (xid == BRK + 3).long()
            top_before = stk.gather(
                1, (sp - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            top_empty = (sp == 0).float()
            top_a = ((sp > 0) & (top_before == 0)).float()
            top_b = ((sp > 0) & (top_before == 1)).float()
            match = (is_close & (sp > 0) & (top_before == close_type)).float()
            mismatch = (is_close & (sp > 0)
                        & (top_before != close_type)).float()
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
            # ---- POST features (stack state AFTER consuming t) ----
            top_after = stk.gather(
                1, (sp - 1).clamp(min=0).unsqueeze(1)).squeeze(1)
            ta_empty = (sp == 0).float()
            ta_a = ((sp > 0) & (top_after == 0)).float()
            ta_b = ((sp > 0) & (top_after == 1)).float()
            stack_feat = torch.stack(
                [top_empty, top_a, top_b, match, mismatch,
                 underflow, overflow, ta_empty, ta_a, ta_b], 1)
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


def gen_dyck_pool(n, L, seed):
    prng = random.Random(seed)
    return [torch.tensor(gen_dyck(prng, L, 2)) for _ in range(n)]


def train_dyck(name, model, pool, steps=2000, B=8, lr=3e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    t0 = time.time()
    hist = []
    for step in range(1, steps + 1):
        sel = [(step * B + i) % len(pool) for i in range(B)]
        x = torch.stack([pool[i] for i in sel])
        y = x[:, 1:]
        lg = model(x[:, :256])
        loss = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            hist.append((step, round(float(loss), 4)))
            print(f"  [{name}] step {step}/{steps} loss {float(loss):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return hist


@torch.no_grad()
def bracket_pos_acc(model, n, depth):
    """Per-position bracket accuracy, split open/close."""
    model.eval()
    L = 256
    ok_open = ok_close = tot_open = tot_close = 0
    for i in range(n):
        x = torch.tensor(gen_dyck(random.Random(9000 + i * 17 + depth * 100),
                                  L, depth)).unsqueeze(0)
        lg = model(x)[:, :L, :]
        pred = lg.argmax(-1).squeeze(0)
        xl = x.squeeze(0).tolist()
        j = 3
        while j < L - 1:
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
                    if BRK <= xl[g] < BRK + 2:
                        tot_open += 1
                        ok_open += int(int(pred[g - 1]) == xl[g])
                    else:
                        tot_close += 1
                        ok_close += int(int(pred[g - 1]) == xl[g])
                j = (seg[-1] + 1) if seg else j + 3
            else:
                j += 1
    return (round(ok_open / tot_open, 4) if tot_open else float("nan"),
            round(ok_close / tot_close, 4) if tot_close else float("nan"),
            tot_open, tot_close)


if __name__ == "__main__":
    pool = gen_dyck_pool(256, 256, 12345)
    arms = [("VETbase", VETLM(V, 16, k=5, K=4)),
            ("VETDCC-base", VETDCC(V, 16, k=5, K=4)),
            ("STACKDCC2-base", STACKDCC2(V, 16, k=5, K=4)),
            ("VETDCC-big", VETDCC(V, 24, k=8, K=8)),
            ("STACKDCC2-big", STACKDCC2(V, 24, k=8, K=8)),
            ("MAMBA", MambaMicro(V, 16))]
    result = {"tag": "ARCH-VET-LM-P11",
              "protocol": "SINGLE-TASK stochastic dyck (same emit() as "
                          "P9), train depth 2 L=256 pool 256 seed 12345, "
                          "2000 steps seed 0; eval: whole-segment "
                          "exact-match depth 3-10 (16 streams) + "
                          "per-position open/close bracket acc at "
                          "depth 3/4/6/8 (16 streams). STACKDCC2 = "
                          "P10 stack + post-update top features "
                          "(10-dim). Coin-flip ceiling: 30% "
                          "double-branch draws are not state-"
                          "determined -> exact-match bounded < 1.0 "
                          "even for a perfect stack user",
              "arms": {}}
    for name, m in arms:
        torch.manual_seed(0)
        print(f"[{name}] params={n_params(m)}", flush=True)
        hist = train_dyck(name, m, pool, 2000, 8)
        m.eval()
        dy = {}
        for d in (3, 4, 5, 6, 7, 8, 9, 10):
            a, tot = dyck_acc(m, 16, d)
            dy[f"exact_d{d}"] = round(a, 4)
        bp = {}
        for d in (3, 4, 6, 8):
            o, cl, to, tc = bracket_pos_acc(m, 16, d)
            bp[f"open_d{d}"] = o
            bp[f"close_d{d}"] = cl
        print(f"[{name}] exact-match: "
              f"{ {k: v for k, v in dy.items()} }", flush=True)
        print(f"[{name}] pos-acc: { {k: v for k, v in bp.items()} }",
              flush=True)
        result["arms"][name] = {
            "params": n_params(m), "loss_curve": hist,
            "exact_match_frontier": dy,
            "per_position_bracket_acc": bp}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open("log.jsonl", "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
