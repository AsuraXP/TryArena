#!/usr/bin/env python3
"""
C59 IDMATCH — exact token-id addressing (soft |id_i-q|), not embedding attn.
Prior: STE discrete (Bengio 2013); NRAM / DNC content keys still embed-space;
SCAN 1711.00350; C58 frozen-emb OOD still ~chance.
Mutation: query is a SCALAR token-id; match positions by temperature equality
on integer ids; hop by reading matched VALUE ids. Symbol-invariant by construction.
TFCopy = embedding pointer control (C57).
"""
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
src = open("arch_c57_halthop.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_c57_halthop.py", "exec"), globals())

class IDMatch(nn.Module):
    def __init__(self, V=V, d=16, hops=3, beta=8.0):
        super().__init__()
        self.hops, self.beta = hops, beta
        self.emb = nn.Embedding(V, d)  # only for optional mix; match is on ids
        self.beta_p = nn.Parameter(torch.tensor(beta))
        # tiny GRU only to locate EQ (controller), not to bind entities
        self.rnn = nn.GRU(d, 8, batch_first=True)
        self.eq_head = nn.Linear(8, 1)

    def forward(self, x):
        B, T = x.shape
        xf = x.float()
        # locate EQ via (x==EQ) hard — allowed, EQ is a control token seen in train
        eq_pos = (x == EQ).float()
        ids = xf
        # start query id = token after QMARK (entity) — index of QMARK+1
        q = torch.zeros(B, device=x.device)
        for i in range(B):
            qm = int((x[i] == QMARK).nonzero()[0])
            q[i] = xf[i, qm + 1]
        ptr = None
        for h in range(self.hops):
            # match subject positions: ids ≈ q
            dist = -(self.beta_p.abs() + 1) * (ids - q.unsqueeze(1)).abs()
            # also require next token is the h-th rel if present in query
            w = F.softmax(dist, dim=-1)
            # gold ptr is VALUE index = subject+2; shift logits
            ptr = torch.roll(dist, shifts=2, dims=1)
            val_ids = torch.roll(ids, shifts=-2, dims=1)
            q = (w * val_ids).sum(-1)
        return ptr


def train_id(m, steps=1200, hops=3, seed=0, L=56):
    opt = torch.optim.Adam(m.parameters(), lr=2e-2)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    last = 0; t0 = time.time()
    for st in range(1, steps+1):
        x, ptr = make(8, L, rng, hops, False)
        loss = F.cross_entropy(m(x), ptr)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if st % 400 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f} p={nparams(m)}", flush=True)
    return last, time.time()-t0

if __name__ == "__main__":
    rec = {"tag": "ARCH-C59-IDMATCH"}
    for name, m, hops in [("IDMatch", IDMatch(hops=3), 3), ("TFCopy", TFCopy(), 3)]:
        print(f"=== {name} p={nparams(m)} ===", flush=True)
        tr = train_id if name == "IDMatch" else None
        if name == "IDMatch":
            loss, wall = train_id(m, hops=3)
        else:
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            rng = np.random.default_rng(0); torch.manual_seed(0)
            last=0; t0=time.time()
            for st in range(1,1201):
                x, ptr = make(8, 56, rng, 3, False)
                loss = F.cross_entropy(m(x), ptr)
                opt.zero_grad(); loss.backward(); opt.step()
                last=float(loss.detach())
                if st%400==0 or st==1:
                    print(f"  step {st} loss={last:.3f} p={nparams(m)}", flush=True)
            loss, wall = last, time.time()-t0
        rec[name] = {"params": nparams(m), "loss": loss, "wall": wall,
                     "h3_id": acc(m, 64, 3, False, 1),
                     "h3_ood": acc(m, 64, 3, True, 2),
                     "h2": acc(m, 64, 2, False, 3)}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl","a").write(json.dumps(rec)+"\n")
    print("RESULT", rec, flush=True)
