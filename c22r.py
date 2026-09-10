"""
ARC-2 C22-R / chatbot state-organ repair (cycle 38).
DIAGNOSIS (instrumented): at answer positions top-1 is 163/166 but margins
are ~0 (avg -0.027): HOST branch logits compete with the state organ at
answer positions (head gate 1.83 x host vs bilinear ~3.6) -> high CE floor
(D1 0.227). Repair variants, each fine-tuned from dialog_chat_final.pt:
  A) more training, lr 1e-3, 4000 steps (undertraining test)
  B) organ scale x8 on state logits + 2000 steps
  C) QUERY-MASKED host: at query-active positions, state logits = organ
     only (structural L-QUERY-READOUT dominance) + 2000 steps
Eval after each: state@4096 dCE, overwrite@4096, mathplus/minus@4096,
chat@4096. Regression guards on D5/D6 included in eval.
NO TF arm (operator directive). USAGE: OMP_NUM_THREADS=1 python3 -u c22r.py
"""
import json, os, random, resource, time, types
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
T0 = time.time()
src = open("dialog_chat.py").read()
cut = src.index("# ----------------------------------------------------------------- run")
mod = types.ModuleType("dc")
exec(compile(src[:cut], "dialog_chat.py", "exec"), mod.__dict__)
DialogMachine = None
for v in vars(mod).values():
    if isinstance(v, type) and hasattr(v, "_state_logits"):
        DialogMachine = v
assert DialogMachine is not None


def make_model(scale=1.0, qmask=False):
    m = DialogMachine()
    m.load_state_dict(torch.load("dialog_chat_final.pt"))
    if scale != 1.0 or qmask:
        orig_state = m._state_logits

        def scaled_state(x, dbg=False):
            if dbg:
                return orig_state(x, dbg=True)
            Aadd, f, qo, idx = orig_state(x, dbg=True)
            bilin = torch.einsum("blm,bln,mnv->blv", f, qo, m.st_m)
            return Aadd + scale * bilin

        m._state_logits = scaled_state
    if qmask:
        import torch.nn as nn

        def masked_forward(self, x):
            B, L = x.shape
            rl = self.router(self.emb(x[:, :3]).reshape(B, -1))
            task = rl.argmax(-1)
            hg = torch.exp(self.head_gate)
            out = torch.zeros(B, L, mod.VOCAB, device=x.device)
            for r in range(3):
                idxb = (task == r).nonzero().squeeze(-1)
                if idxb.numel() == 0:
                    continue
                xr = x[idxb]
                hr = self.norms[r](self.hosts[r](self.emb(xr)))
                lg = hg[r] * self.heads[r](hr)
                if r == 0:
                    organ = self._state_logits(xr)
                    _, f, qo, _ = self._state_logits(xr, dbg=True)
                    qmask_ = (qo.sum(-1) > 0) & (f.sum(-1) > 0)
                    lg = torch.where(qmask_.unsqueeze(-1), organ, lg + organ)
                elif r == 1:
                    lg = lg + self._math_logits(xr)
                out.scatter_(0, idxb.view(-1, 1, 1).expand_as(lg), lg)
            return out, rl

        m.forward = masked_forward.__get__(m, DialogMachine)
    return m


@torch.no_grad()
def eval_bundle(m, tag, out):
    m.eval()
    out[f"{tag}/state4096"] = mod.stream_probe(m, 0, 4096, reps=2)
    out[f"{tag}/overwrite4096"] = mod.overwrite_probe(m, 4096, reps=2)
    out[f"{tag}/mathplus"] = mod.stream_probe(m, 1, 4096, reps=2, op=mod.PLUS)
    out[f"{tag}/mathminus"] = mod.stream_probe(m, 1, 4096, reps=2, op=mod.MINUS)
    out[f"{tag}/chat"] = mod.stream_probe(m, 2, 4096, reps=2)
    m.train()


def finetune(m, steps, lr, seed):
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    rng = random.Random(seed)
    for s in range(1, steps + 1):
        x, y, o, task = mod.gen_dialogue_t(32, 63, rng)
        lm, rt = mod.train_step(m, opt, x, y, task)
        if s % 1000 == 0:
            print(f"  ft s{s} lm {lm:.4f} rt {rt:.4f} "
                  f"st_m_abs {float(m.st_m.abs().sum()):.1f}", flush=True)


RESULTS = {}
m0 = make_model()
eval_bundle(m0, "baseline", RESULTS)
print(f"[baseline] { {k: v for k, v in RESULTS.items()} }", flush=True)

print("[arm A] more training lr=1e-3 x4000", flush=True)
mA = make_model()
finetune(mA, 4000, 1e-3, 101)
eval_bundle(mA, "A_more", RESULTS)

print("[arm B] organ scale x8 + 2000 steps", flush=True)
mB = make_model(scale=8.0)
finetune(mB, 2000, 1e-3, 102)
eval_bundle(mB, "B_scale8", RESULTS)

print("[arm C] query-masked host + 2000 steps", flush=True)
mC = make_model(qmask=True)
finetune(mC, 2000, 1e-3, 103)
eval_bundle(mC, "C_qmask", RESULTS)

for k, v in RESULTS.items():
    print(f"[final] {k} = {v}", flush=True)
torch.save(mC.state_dict(), "c22r_qmask.pt")
torch.save(mB.state_dict(), "c22r_scale8.pt")
torch.save(mA.state_dict(), "c22r_more.pt")
out = dict(tag="ARC2-C22R-REPAIR", results=RESULTS,
           wall_s=round(time.time() - T0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(out), flush=True)
with open("log.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
print("DONE", flush=True)
