#!/usr/bin/env python3
"""
C60 SCATRAM — in-context integer-indexed dictionary (not embed attn).
Prior: Engram hashed lookup arXiv:2601.07372 (static ngram tables, not in-context hops);
DNC/NTM content keys in embedding space (OOD-fragile, C58).
Mutation: WRITE M[rel, src_id] += one_hot(dst_id) via scatter; READ is index_select
on token ids in the prompt. Hops = nested reads. Params ~0 on the memory (buffer only).
Tiny learned: relation embedding unused; beta unused. Control: TFCopy C57.
"""
import os, json, time
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
src = open("arch_c57_halthop.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_c57_halthop.py", "exec"), globals())

class ScatRAM(nn.Module):
    """No entity embeddings. Memory is a V×V table per rel, filled from the sequence."""
    def __init__(self, V=V, nrel=8, hops=4):
        super().__init__()
        self.V, self.nrel, self.hops = V, nrel, hops
        # dummy param so Adam isn't empty
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, T = x.shape
        Vv, R = self.V, self.nrel
        device = x.device
        # M[b, rel, src] -> dist over dst tokens  (use V as dst)
        # rel tokens are 50.. ; map rel_id = (tok-50).clamp(0,R-1)
        logits = torch.zeros(B, T, device=device)
        for bi in range(B):
            M = torch.zeros(R, Vv, Vv, device=device)
            seq = x[bi]
            # write all e,r,f triples (scan)
            sep = int((seq == SEP).nonzero()[0]) if (seq == SEP).any() else T
            for t in range(max(0, sep - 2)):
                e, r, f = int(seq[t]), int(seq[t+1]), int(seq[t+2])
                if t+2 >= sep:
                    break
                if 50 <= r < 50 + R and e < Vv and f < Vv:
                    M[r - 50, e, f] = M[r - 50, e, f] + 1
            # parse query
            qm = int((seq == QMARK).nonzero()[0])
            eq = int((seq == EQ).nonzero()[0])
            a = int(seq[qm + 1])
            rels = [int(t) - 50 for t in seq[qm + 2:eq].tolist()]
            cur = a
            for k, rr in enumerate(rels[: self.hops]):
                rr = max(0, min(R - 1, rr))
                dist = M[rr, cur]  # [V]
                if dist.sum() <= 0:
                    break
                cur = int(dist.argmax())
            # pointer logits: peak at positions equal to final cur that are values
            logits[bi] = (seq == cur).float() * self.scale
            # prefer the matching triple's value pos: last write of that dst with last rel
            # already handled if unique
        return logits


def run(m, steps, lr, hops=3):
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr) if params else None
    rng = np.random.default_rng(0); torch.manual_seed(0)
    last = 0; t0 = time.time()
    for st in range(1, steps + 1):
        x, ptr = make(8, 56, rng, hops, False)
        loss = F.cross_entropy(m(x), ptr)
        if opt:
            opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if st % 200 == 0 or st == 1:
            print(f"  step {st} loss={last:.3f} p={nparams(m)}", flush=True)
        if st >= 3 and "Scat" in type(m).__name__:
            break  # deterministic; no need to train
    return last, time.time() - t0

if __name__ == "__main__":
    rec = {"tag": "ARCH-C60-SCATRAM"}
    for name, m, steps, lr in [
        ("ScatRAM", ScatRAM(), 3, 1e-3),
        ("TFCopy", TFCopy(), 800, 3e-3),
    ]:
        print(f"=== {name} p={nparams(m)} ===", flush=True)
        loss, wall = run(m, steps, lr)
        rec[name] = {"params": nparams(m), "loss": loss, "wall": wall,
                     "h3_id": acc(m, 64, 3, False, 1),
                     "h3_ood": acc(m, 64, 3, True, 2),
                     "h4_zs": acc(m, 64, 4, False, 3),
                     "h2": acc(m, 64, 2, False, 4)}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl", "a").write(json.dumps(rec) + "\n")
    print("RESULT", rec, flush=True)
