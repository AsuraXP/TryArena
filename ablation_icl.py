"""
C14b ABLATION: why is the ICL branch 5-10x slower in iso2 (1.24 target at
3333 pure-ICL steps) than the standalone SRAM organ (0.027 @2500)?
Pure ICL, batch 32, 2500 steps, dCE@4096 (cert = 0.0217 total | 0.0218-
0.0270 target; ln16 = 2.773):
  F1: branch1 components, 45-vocab CE (as built in iso2)
  F2: branch1 components, readout restricted to the 32 ICL sub-vocab
      (logits at positions 6-37, zeros elsewhere) — isolates the vocab tax
  F3: branch1 components, 45-vocab CE, NO router gradient on emb
      (emb rows frozen for the first-3-token router path by detaching)
"""
import json, math, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB, NK = 45, 16
g = {"__name__": "abl2"}
exec(open("unified_iso.py").read().split("\nRESULTS = {}")[0], g)
SSMBlock, gen_icl_t, gen_mixed = g["SSMBlock"], g["gen_icl_t"], g["gen_mixed"]
K0, V0 = 6, 22

class Branch1(nn.Module):
    def __init__(self, subvocab):
        super().__init__()
        d = 16
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.host = SSMBlock(d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        self.W = nn.Linear(d, 32 if subvocab else VOCAB)

    def sram_logits(self, x):
        B, L = x.shape
        d = 16
        slots = torch.zeros(B, NK, d, device=x.device)
        seen = torch.zeros(B, NK, dtype=torch.bool, device=x.device)
        last_key = torch.full((B,), -1, dtype=torch.long, device=x.device)
        emb = self.emb(x)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for t in range(L):
            tok = x[:, t]
            is_key = (tok >= K0) & (tok < K0 + NK)
            is_val = (tok >= V0) & (tok < V0 + NK)
            k = (tok - K0).clamp(0, NK - 1)
            idx = k.view(B, 1, 1).expand(B, 1, d)
            cand = self.W(slots.gather(1, idx).squeeze(1))
            mask = (is_key & seen.gather(1, k.unsqueeze(1)).squeeze(1)).float()
            if cand.shape[-1] == VOCAB:
                out[:, t] = cand * mask.unsqueeze(-1)
            else:
                out[:, t, K0:K0 + 32] = cand * mask.unsqueeze(-1)
            for b in range(B):
                if bool(is_val[b]) and int(last_key[b]) >= 0:
                    slots[b, int(last_key[b])] = emb[b, t].detach()
                    seen[b, int(last_key[b])] = True
                if bool(is_key[b]):
                    last_key[b] = int(k[b])
        return out

    def forward(self, x):
        h = self.norm(self.host(self.emb(x)))
        return self.head(h) + self.sram_logits(x)

@torch.no_grad()
def eval4096(model, reps=2):
    model.eval()
    ce = orc = n = tgt = tgt_n = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + 4096 + i)
        x, y, o = gen_icl_t(1, 4096, rng)
        nll = -F.log_softmax(model(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        tgt += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return (round((ce - orc) / n, 4), round(tgt / tgt_n, 4))

def run(tag, subvocab, router_ce=False, steps=2500, seed=0):
    torch.manual_seed(seed)
    m = Branch1(subvocab)
    router = nn.Sequential(nn.Linear(3 * 16, 16), nn.GELU(), nn.Linear(16, 3))
    params = list(m.parameters()) + list(router.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    t0 = time.time()
    for step in range(steps):
        x, y, o = gen_icl_t(32, 64, rng)
        loss = F.cross_entropy(m(x).reshape(-1, VOCAB), y.reshape(-1))
        if router_ce:
            xm, ym, tm = gen_mixed(32, 63, rng)
            rl = router(m.emb(xm[:, :3]).reshape(32, -1))
            loss = loss + 0.5 * F.cross_entropy(rl, tm)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad()
    d = eval4096(m)
    print(f"  {tag:<34} total|target @4096 = {d}   ({time.time()-t0:.0f}s)", flush=True)
    return d

print("C14b ABLATION — pure ICL, batch 32, L=63(64), 2500 steps "
      "(standalone cert = 0.0217|0.0218-0.0270; iso2_final = 0.4719|1.2382):", flush=True)
run("F1: 45-vocab (as built)", subvocab=False)
run("F2: 32-sub-vocab readout", subvocab=True)
run("F3: 45-vocab + router CE on emb", subvocab=False, router_ce=True)
print("DONE", flush=True)
