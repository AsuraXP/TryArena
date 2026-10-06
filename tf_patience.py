"""
ARC-2 CYCLE 17 / PATIENCE TEST: does LONG training buy the transformer
the reasoning we have for free?
The paradigm claim under test: "generalization and reasoning come from
long-term training and massive resource usage." Evidence so far on this
hardware: micro-TF 103k p loses the 4-family stream at 12k-20k steps
(4.8-10.6 dCE @4096); STRONG-TF 796k p / 10k steps (C11, 8x compute)
also loses (3.4-8.3). Missing arm: 4x the patience at the same capacity
on the current 4-family stream.
Run: micro-TF d64/2L (104,843 p) x 40k steps on the machine-v4 mixed4
stream (8 rows/family, L=63), eval @4096 on all four families.
Machine v4 (21,305 p, 12k steps, ~10 min CPU) reference: echo -0.3004 |
icl 0.0072|0.0021 (9x) | mod7 0.0036 | add 0.0091, routing 1.0.
USAGE: OMP_NUM_THREADS=1 python3 -u tf_patience.py
"""
import json, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
VOCAB = 75
CFG = dict(steps=40000, batch=32, train_len=63)
print(f"[setup] tf-patience cfg={CFG}", flush=True)
t_start = time.time()

g = {"__name__": "tp"}
exec(open("unified_add.py").read().split("\nRESULTS = {}")[0], g)
n_params, gen_echo_t, gen_icl_t, gen_mod7_t, gen_add_t, gen_mixed4 = (
    g["n_params"], g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"], g["gen_add_t"], g["gen_mixed4"])


class MixedTF(nn.Module):
    def __init__(self, d_model=64, n_layers=2, n_heads=4):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, VOCAB)
        self.head.weight = self.emb.weight

    @staticmethod
    def sinusoidal(L, d):
        p = torch.zeros(L, d)
        pos = torch.arange(L).unsqueeze(1).float()
        i = torch.arange(0, d, 2).float()
        p[:, 0::2] = torch.sin(pos * torch.exp(-9 * i / d))
        p[:, 1::2] = torch.cos(pos * torch.exp(-9 * i / d))
        return p

    def forward(self, x):
        B, L = x.shape
        h = self.emb(x) + self.sinusoidal(L, self.d_model).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(L).to("cpu")
        return self.head(self.enc(h, mask))


torch.manual_seed(0)
tf = MixedTF()
print(f"[arm] micro TF x40k patience test params={n_params(tf)}", flush=True)
tf.train()
opt = torch.optim.AdamW(tf.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
for step in range(1, CFG["steps"] + 1):
    x, y, task = gen_mixed4(CFG["batch"], CFG["train_len"], rng)
    loss = F.cross_entropy(tf(x).reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    if step % 5000 == 0:
        print(f"  [tf-patience] step {step}/{CFG['steps']} CE {loss.item():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def eval_tf_task(model, gen, L, reps=2):
    model.eval()
    bs = max(1, min(4, 4096 // L))
    ce = orc = n = 0.0
    tgt_ce = tgt_n = 0.0
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gen(bs, L, rng)
        nll = -F.log_softmax(model(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
    return (round((ce - orc) / n, 4), round(tgt_ce / tgt_n, 4))


tf_time = round(time.time() - t0, 1)
r = {"echo": eval_tf_task(tf, gen_echo_t, 4096, 2),
     "icl": eval_tf_task(tf, gen_icl_t, 4096, 2),
     "mod7": eval_tf_task(tf, gen_mod7_t, 4096, 2),
     "add": eval_tf_task(tf, gen_add_t, 4096, 2),
     "params": n_params(tf)}
RESULTS = {"tf_patience_40k": r}
print(f"  tf_patience_40k (train wall {tf_time}s): {r}", flush=True)

print("\n" + "=" * 96)
print("PATIENCE TEST @4096 — TF 104,843p x 40k steps (~4x budget) vs machine v4")
print("machine v4: 21,305p x 12k steps ~10min | echo -0.3004 | icl 0.0072|0.0021 |")
print("mod7 0.0036 | add 0.0091 | routing 1.0 | +16384 no-decay (C16)")
print("prior TF arms: 12k steps: echo 4.78 | icl 5.39|10.57 | mod7 5.63 | add 4.82;")
print("STRONG-TF 796k/10k (C11): echo 3.42 | mod7 8.30 | icl n/a (3-task stream)")
print("=" * 96)
v = RESULTS["tf_patience_40k"]
print(f"tf_patience_40k params {v['params']}  echo {v['echo']}  icl {v['icl']}  "
      f"mod7 {v['mod7']}  add {v['add']}", flush=True)
print("=" * 96)
final = {"tag": "ARC2-C17-TF-PATIENCE-40K", "runs": RESULTS,
         "tf_train_wall_s": tf_time,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
