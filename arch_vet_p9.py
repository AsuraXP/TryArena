#!/usr/bin/env python3
"""C53 P9 VETDCC — VET + EXACT deterministic counter channels.
mod-3 counter on tok 21 + clamped depth counter D=6 on BRK toks, reset at T_TASK;
zero-init injection into controller + readout.
Prediction: dyck exact-match stays high at depth 3-4 (in-clamp).
Prior: Neural GPUs / Grid LSTM counters; counting as state (Weiss et al. 2018 RNNs count);
P1 VET controller-STATE counting law L-VALUE-CHANNEL-CARRIES.
"""
import os, json
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
src = open("arch_vet_lm.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_vet_lm.py", "exec"), globals())
set_threads()

class VETDCC(VETLM):
    def __init__(self, **kw):
        super().__init__(**kw)
        D = 6
        self.D = D
        # zero-init injection: map counters into controller bias and readout
        self.W_mod = nn.Parameter(torch.zeros(3, self.k))
        self.W_dep = nn.Parameter(torch.zeros(D + 1, self.k))
        self.R_mod = nn.Parameter(torch.zeros(3, self.V))
        self.R_dep = nn.Parameter(torch.zeros(D + 1, self.V))

    def forward(self, x):
        B, T = x.shape
        k, d, K, Vv = self.k, self.d, self.K, self.V
        device = x.device
        s = torch.zeros(B, k, device=device); s[:, 0] = 1
        v = torch.zeros(B, d, device=device)
        buf = torch.zeros(B, K, d, device=device)
        depth = torch.zeros(B, device=device)
        modc = torch.zeros(B, dtype=torch.long, device=device)
        dpth = torch.zeros(B, dtype=torch.long, device=device)
        logits = []
        e_all = self.emb(x)
        for t in range(T):
            xt = x[:, t]; et = e_all[:, t]
            # reset at T_TASK
            rst = (xt == T_TASK)
            modc = torch.where(rst, torch.zeros_like(modc), modc)
            dpth = torch.where(rst, torch.zeros_like(dpth), dpth)
            modc = torch.where(xt == CNT_TOK, (modc + 1) % 3, modc)
            dpth = torch.where(xt == BRK_OPEN, (dpth + 1).clamp(max=self.D), dpth)
            dpth = torch.where(xt == BRK_CLOSE, (dpth - 1).clamp(min=0), dpth)
            inj = self.W_mod[modc] + self.W_dep[dpth]
            Wc = self.W_ctrl[xt]; bc = self.b_ctrl[xt] + inj
            s = F.softmax(torch.einsum("bk,bkj->bj", s, Wc) + bc, dim=-1)
            dec = torch.sigmoid(s @ self.decay)
            g = torch.sigmoid(self.g_write(torch.cat([et, s], dim=-1)))
            v = dec * v + g * torch.tanh(self.W_write(et))
            is_push = ((xt == BRK_OPEN) | (xt == PAIR_OPEN) | (xt == TOK_A)).float()
            is_pop = ((xt == BRK_CLOSE) | (xt == PAIR_CLOSE) | (xt == TOK_B)).float()
            hard_push, hard_pop = (is_push > 0.5).float(), (is_pop > 0.5).float()
            new_depth = (depth + hard_push - hard_pop).clamp(0, K)
            slot = depth.long().clamp(0, K - 1)
            buf = buf.clone()
            idx = torch.arange(B, device=device)
            buf[idx, slot] = buf[idx, slot] + self.stack_scale * et * hard_push.unsqueeze(-1)
            pop_slot = (depth - 1).long().clamp(0, K - 1)
            top = buf[idx, torch.where(hard_pop > 0, pop_slot, slot)]
            depth = new_depth
            q = v + 0.25 * top
            read = torch.einsum("bk,kdv,bd->bv", s, self.W_read, q) + self.W_stk(top) + self.b_read
            read = read + self.R_mod[modc] + self.R_dep[dpth]
            logits.append(read)
        return torch.stack(logits, 1)

m = VETDCC(k=5, d=16, K=4)
print("P9 VETDCC params", m.nparams(), flush=True)
hist = train_arm(m, ["TRACK", "MODK", "DYCK", "PAIR"], steps=2000, L=128, B=4,
                 seed=0, log_every=500, L_eval=(128, 256))
dyck = eval_acc(m, "DYCK", n=48, span=(16, 48), L=128, seed=0)
modk = eval_acc(m, "MODK", n=32, span=(16, 48), L=128, seed=0)
rec = {"tag": "ARCH-VET-LM-P9", "params": m.nparams(), "dyck_ev": dyck, "modk_ev": modk, "ce": hist[-1][2]}
open("log.jsonl", "a").write(json.dumps(rec) + "\n")
print("RESULT", rec, flush=True)
