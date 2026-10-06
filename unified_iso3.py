"""
ARC-2 CYCLE 14c / THE MACHINE v3: isolated mixture + task cycling +
sub-vocab organ readouts + certified organ gate
=====================================================================
C14b diagnosis (ablation_icl, pure ICL batch 32, 2500 steps, cert =
0.0217|0.0218-0.0270; ln16 = 2.773):
  F1 45-vocab readout, no gate        = 0.6891|1.4362
  F2 32-sub-vocab readout, no gate    = 0.4053|0.8527
  F3 45-vocab + router CE on emb      = 0.5036|1.1787
  F4 45-vocab + organ_gate            = 0.2589|0.3614
  F5 32-sub-vocab + organ_gate        = 0.0072|0.0012  <- BELOW cert
Two independent findings:
 (1) ORGAN GATE: the certified sram_icl.py gates organ logits by
     exp(gate), gate init 0 (scale 1). All C13/C14 ports dropped it.
     Without it the linear readout co-training stalls 30-100x.
     L-ORGAN-GATE: a content-addressed organ with a LEARNED readout map
     needs a learned soft-start scale; a direct-lookup table (stack)
     does not (cert echo has none).
 (2) VOCAB TAX: a 16-d slot content mapped linearly into 45 classes must
     suppress 13 junk directions with no spare dims -> the readout can't
     separate 16 value classes while killing junk. Certified sram_icl
     never hit this (its organ spoke the whole 32-vocab). Fix: each organ
     reads out in ITS OWN sub-vocab, placed at its token range.
     L-ORGAN-ALPHABET: an organ's readout must live in its own alphabet.
Machine v3 = IsoModel (C14) + (1) organ_gate on the SRAM organ +
(2) SRAM readout 16->32 placed at tokens 6-37 + task cycling (C14b duty
fix) + zero-init host heads + shared token-disjoint emb + learned
per-example router. 13,024 params.
Protocol: 10000 cycling steps (~3333/task), ckpts 3000/9000. Baseline:
the iso2 micro_tf_10k arm (identical stream/steps/seed: echo 10.75,
icl 7.22|3.09, mod7 6.41 @4096) — cited, not re-run.
Win = ALL THREE tasks at/below standalone cert inside one model, stable
1x -> 1.33x budget, routing 1.0.
USAGE: OMP_NUM_THREADS=1 python3 -u unified_iso3.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB, NK = 45, 16
K0 = 6
CFG = dict(steps=10000, batch=32, train_len=63, ckpts=[3000, 9000],
           d_model=16, KSTACK=4096)
print(f"[setup] unified-iso3 cfg={CFG}", flush=True)
t_start = time.time()

g = {"__name__": "u3"}
exec(open("unified_iso.py").read().split("\nRESULTS = {}")[0], g)
SSMBlock, n_params, train_step, eval_task = (
    g["SSMBlock"], g["n_params"], g["train_step"], g["eval_task"])
gen_echo_t, gen_icl_t, gen_mod7_t, gen_mixed = (
    g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"], g["gen_mixed"])

class IsoModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.router = nn.Sequential(nn.Linear(3 * d, 16), nn.GELU(), nn.Linear(16, 3))
        self.hosts = nn.ModuleList([SSMBlock(d) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Linear(d, VOCAB) for _ in range(3)])
        for h in self.heads:
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.stack_table = nn.Parameter(0.1 * torch.randn(8, VOCAB))
        self.W_readout = nn.Linear(d, 32)               # L-ORGAN-ALPHABET
        self.organ_gate = nn.Parameter(torch.tensor(0.0))  # L-ORGAN-GATE

    def _stack_logits(self, x):
        B, L = x.shape
        KS = CFG["KSTACK"]
        feats = torch.empty(B, L, 3, dtype=torch.long)
        for b in range(B):
            stack = []
            for t in range(L):
                tok = int(x[b, t])
                if tok in (0, 1):
                    if len(stack) < KS:
                        stack.append(tok)
                elif tok == 2:
                    if stack:
                        stack.pop()
                feats[b, t, 0] = stack[-1] if stack else 0
                feats[b, t, 1] = 1 if not stack else 0
                feats[b, t, 2] = 1 if tok == 2 else 0
        combo = feats[:, :, 0] + feats[:, :, 1] * 2 + feats[:, :, 2] * 4
        return self.stack_table[combo]

    def _sram_logits(self, x):
        B, L = x.shape
        d = CFG["d_model"]
        K1, V0, V1 = K0 + NK, K0 + NK, K0 + 2 * NK
        slots = torch.zeros(B, NK, d, device=x.device)
        seen = torch.zeros(B, NK, dtype=torch.bool, device=x.device)
        last_key = torch.full((B,), -1, dtype=torch.long, device=x.device)
        emb = self.emb(x)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for t in range(L):
            tok = x[:, t]
            is_key = (tok >= K0) & (tok < K1)
            is_val = (tok >= V0) & (tok < V1)
            k = (tok - K0).clamp(0, NK - 1)
            idx = k.view(B, 1, 1).expand(B, 1, d)
            cand = self.W_readout(slots.gather(1, idx).squeeze(1))
            mask = (is_key & seen.gather(1, k.unsqueeze(1)).squeeze(1)).float()
            out[:, t, K0:V1] = cand * mask.unsqueeze(-1)
            for b in range(B):
                if bool(is_val[b]) and int(last_key[b]) >= 0:
                    slots[b, int(last_key[b])] = emb[b, t].detach()
                    seen[b, int(last_key[b])] = True
                if bool(is_key[b]):
                    last_key[b] = int(k[b])
        return out

    def forward(self, x):
        B, L = x.shape
        rl = self.router(self.emb(x[:, :3]).reshape(B, -1))
        task = rl.argmax(-1)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for r in range(3):
            idx = (task == r).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            xr = x[idx]
            hr = self.norms[r](self.hosts[r](self.emb(xr)))
            lg = self.heads[r](hr)
            if r == 0:
                lg = lg + self._stack_logits(xr)
            elif r == 1:
                lg = lg + torch.exp(self.organ_gate) * self._sram_logits(xr)
            out.scatter_(0, idx.view(-1, 1, 1).expand_as(lg), lg)
        return out, rl

# ---------------------------------------------------------------- experiment
def gen_pure(r, batch, length, rng):
    if r == 0:
        x, y, _ = gen_echo_t(batch, 64, rng)
        return x[:, :length], y[:, :length], torch.zeros(batch, dtype=torch.long)
    if r == 1:
        x, y, _ = gen_icl_t(batch, 64, rng)
        return x, y, torch.ones(batch, dtype=torch.long)
    x, y, _ = gen_mod7_t(batch, length, rng)
    return x, y, torch.full((batch,), 2, dtype=torch.long)

def eval_all(model, tag, results):
    r = {"echo": eval_task(model, gen_echo_t, 4096, 2, 0),
         "icl": eval_task(model, gen_icl_t, 4096, 2, 1, tgt=True),
         "mod7": eval_task(model, gen_mod7_t, 4096, 2, 2)}
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)

RESULTS = {}
torch.manual_seed(0)
iso = IsoModelV3()
print(f"[arm] machine v3 (isolated + cycling + sub-vocab organ + gate) "
      f"params={n_params(iso)}", flush=True)
iso.train()
opt = torch.optim.AdamW(iso.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
ckpts_done = set()
for step in range(1, CFG["steps"] + 1):
    r = (step - 1) % 3
    x, y, task = gen_pure(r, CFG["batch"], CFG["train_len"], rng)
    lm, rt = train_step(iso, opt, x, y, task)
    if step in CFG["ckpts"] and step not in ckpts_done:
        torch.save(iso.state_dict(), f"unified_iso3_{step}.pt")
        ckpts_done.add(step)
        print(f"    [iso3] checkpoint at step {step}", flush=True)
    if step % 2000 == 0:
        print(f"  [iso3 s0] step {step}/{CFG['steps']} lm {lm:.4f} rt {rt:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[iso3] trained in {time.time()-t0:.0f}s (gate={iso.organ_gate.item():.3f})", flush=True)
torch.save(iso.state_dict(), "unified_iso3_final.pt")

print("[eval] checkpoints @4096:", flush=True)
for c in CFG["ckpts"]:
    iso.load_state_dict(torch.load(f"unified_iso3_{c}.pt"))
    eval_all(iso, f"iso3_{c // 1000}x", RESULTS)
iso.load_state_dict(torch.load("unified_iso3_final.pt"))
eval_all(iso, "iso3_final", RESULTS)
RESULTS["iso3_final"]["params"] = n_params(iso)
RESULTS["iso3_final"]["gate"] = round(iso.organ_gate.item(), 3)
for c in CFG["ckpts"]:
    RESULTS[f"iso3_{c // 1000}x"]["params"] = n_params(iso)
del iso

print("\n" + "=" * 88)
print("MACHINE v3 @4096 (dCE; routing acc in parens)")
print("standalone certs: echo -0.2935 | icl 0.0217|0.0218-0.0270 | mod7 0.0025-0.0071")
print("baseline (iso2 micro_tf_10k, same stream/steps): echo 10.7542 | icl")
print("7.2156|3.0893 | mod7 6.4079")
print("=" * 88)
for k in ["iso3_3x", "iso3_9x", "iso3_final"]:
    v = RESULTS[k]
    print(f"{k:<10} params {v['params']:<7} echo {v['echo']}  icl {v['icl']}  mod7 {v['mod7']}",
          flush=True)
print("=" * 88)
final = {"tag": "ARC2-C14C-ISO3-MACHINE-V3", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
