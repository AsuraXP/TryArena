#!/usr/bin/env python3
"""
C58 FROZEN-RAND keys for entity-OOD hops.
Prior: SCAN Lake&Baroni 1711.00350 — seq2seq memorizes without systematicity.
VSA/Kanerva; Attention-as-binding arXiv:2512.14709 (TF still quadratic).
C57b: HopCopy3 h3_id 0.81 / h3_ood ~chance — learned emb rows for OOD IDs never trained.
Mutation: entity table = FROZEN N(0,I) for ALL vocab (incl OOD ids). Same geometry
train/test. Not VSA bind⊗; HopCopy on frozen keys. TFCopy frozen vs learned controls.
"""
import os, json, time, math
os.environ["OMP_NUM_THREADS"] = "1"
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
torch.set_num_threads(1)
src = open("arch_c57_halthop.py").read().rsplit('\nif __name__ == "__main__":', 1)[0]
exec(compile(src, "arch_c57_halthop.py", "exec"), globals())

def freeze_emb(m, seed=0):
    torch.manual_seed(seed)
    with torch.no_grad():
        m.emb.weight.copy_(torch.randn_like(m.emb.weight) * 0.3)
    m.emb.weight.requires_grad = False
    return m

def train3(m, steps=1800, seed=0, L=56):
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=3e-3)
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
    rec = {"tag": "ARCH-C58-FROZEN"}
    arms = [
        ("HopFrz", freeze_emb(HaltHop(maxh=3))),
        ("HopLrn", HaltHop(maxh=3)),
        ("TFFrz", freeze_emb(TFCopy())),
        ("TFLrn", TFCopy()),
    ]
    for name, m in arms:
        print(f"=== {name} p={nparams(m)} ===", flush=True)
        loss, wall = train3(m)
        rec[name] = {"params": nparams(m), "loss": loss, "wall": wall,
                     "h3_id": acc(m, 64, 3, False, 1),
                     "h3_ood": acc(m, 64, 3, True, 2)}
        print("RESULT", name, rec[name], flush=True)
    open("log.jsonl","a").write(json.dumps(rec)+"\n")
    print("RESULT", rec, flush=True)
