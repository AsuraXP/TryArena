#!/usr/bin/env python3
"""C57b: train hops=3 only, HopCopy fixed inner hops=3 vs TFCopy. Halt mix failed C57."""
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
# reuse make/models from c57 via exec split
src = open("arch_c57_halthop.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_c57_halthop.py", "exec"), globals())

def train3(m, steps=1800, seed=0, L=56, inner_fixed=False):
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    last = 0; t0 = time.time()
    for st in range(1, steps+1):
        x, ptr = make(8, L, rng, 3, False)
        loss = F.cross_entropy(m(x), ptr)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if st % 600 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f} p={nparams(m)}", flush=True)
    return last, time.time()-t0

if __name__ == "__main__":
    rec = {"tag": "ARCH-C57B-H3FIX"}
    for name, m in [("HopCopy3", HaltHop(maxh=3)), ("TFCopy", TFCopy())]:
        print(f"=== {name} p={nparams(m)} ===", flush=True)
        loss, wall = train3(m)
        rec[name] = {"params": nparams(m), "loss": loss, "wall": wall,
                     "h3_id": acc(m, 64, 3, False, 1),
                     "h3_ood": acc(m, 64, 3, True, 2),
                     "h2": acc(m, 64, 2, False, 3),
                     "h4_zs": acc(m, 64, 4, False, 4)}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl","a").write(json.dumps(rec)+"\n")
    print("RESULT", rec, flush=True)
