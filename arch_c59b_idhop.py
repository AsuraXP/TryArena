#!/usr/bin/env python3
"""C59b IDHOP: match (id==q AND next==rel_k) then read +2. Soft equality on ids."""
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
src = open("arch_c57_halthop.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_c57_halthop.py", "exec"), globals())

class IDHop(nn.Module):
    def __init__(self, hops=3, beta=12.0):
        super().__init__()
        self.hops = hops
        self.beta = nn.Parameter(torch.tensor(beta))

    def forward(self, x):
        B, T = x.shape
        xf = x.float()
        q = torch.zeros(B, device=x.device)
        rels = []
        for i in range(B):
            qm = int((x[i] == QMARK).nonzero()[0])
            q[i] = xf[i, qm + 1]
        # rels after entity until EQ
        rel_t = torch.zeros(B, self.hops, device=x.device)
        for i in range(B):
            qm = int((x[i] == QMARK).nonzero()[0])
            eq = int((x[i] == EQ).nonzero()[0])
            rs = x[i, qm + 2:eq].tolist()
            for k in range(self.hops):
                rel_t[i, k] = rs[k] if k < len(rs) else 0
        ptr = None
        b = self.beta.abs() + 1
        for k in range(self.hops):
            sub = -(b) * (xf - q.unsqueeze(1)).abs()
            nxt = torch.roll(xf, -1, 1)
            relm = -(b) * (nxt - rel_t[:, k:k+1]).abs()
            logits = sub + relm
            # mask last 2 pos
            logits = logits.clone(); logits[:, -2:] = -1e4
            w = F.softmax(logits, -1)
            ptr = torch.roll(logits, 2, 1)  # value pos
            val = torch.roll(xf, -2, 1)
            q = (w * val).sum(-1)
        return ptr

def run(m, steps, lr):
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    rng = np.random.default_rng(0); torch.manual_seed(0)
    last=0; t0=time.time()
    for st in range(1, steps+1):
        x, ptr = make(8, 56, rng, 3, False)
        loss = F.cross_entropy(m(x), ptr)
        opt.zero_grad(); loss.backward(); opt.step()
        last=float(loss.detach())
        if st%400==0 or st==1:
            print(f"  step {st} loss={last:.3f} p={nparams(m)}", flush=True)
    return last, time.time()-t0

if __name__ == "__main__":
    rec={"tag":"ARCH-C59B-IDHOP"}
    for name,m,lr,st in [("IDHop", IDHop(), 5e-2, 800), ("TFCopy", TFCopy(), 3e-3, 800)]:
        print(f"=== {name} ===", flush=True)
        loss,wall=run(m,st,lr)
        rec[name]={"params":nparams(m),"loss":loss,"wall":wall,
                   "h3_id":acc(m,64,3,False,1),"h3_ood":acc(m,64,3,True,2)}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl","a").write(json.dumps(rec)+"\n")
    print("RESULT", rec, flush=True)
