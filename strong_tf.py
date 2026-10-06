"""
ARC-2 CYCLE 11 / STRONG-TF: give the transformer the resources it needs
=======================================================================
User's point (validated, Hahn TACL-2020 + arxiv 2310.08661 + 2408.05506):
micro-TF negatives could be read as "under-resourced transformer". Hahn:
standard TFs cannot model periodic finite-state languages / hierarchy
unless layers/heads grow WITH INPUT LENGTH; 2310.08661: generalizable
iterative counting needs >= L layers (N layers => <= N sequential ops);
2408.05506: length-generalization failure = random-access failure in
context. Depth buys a FINITE bound, not the mechanism.

Test: TF_STRONG (d128, 4 layers, 10k steps, ~790k params — 8x the micro TF
compute) vs our 3,211-param ssm_d16_1 (2.5k steps) and 48-param table3 on
both decisive axes. If the strong TF still fails @4096 where the 3k/48p
models are exact, the limitation is ARCHITECTURAL, not a resource budget.

  axis 1 (hard tasks, VOCAB 27): mod7 + dyck10, eval 64/512/2048/4096
  axis 2 (Dyck-echo, VOCAB 6):   echo-dCE @4096 (true oracle = 0)
USAGE: OMP_NUM_THREADS=1 python3 -u strong_tf.py
"""
import json, math, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cpu"
torch.set_num_threads(1)
t_start = time.time()

# ---- reuse task defs
strip_src = open("ssm_strip.py").read()
gS = {"__name__": "stf"}
exec(strip_src.split("# ---------------------------------------------------------------- experiment")[0], gS)
echo_src = open("dyck_echo.py").read()
gE = {"__name__": "ste"}
exec(echo_src.split("# ---------------------------------------------------------------- experiment")[0], gE)
VOCAB27 = gS["VOCAB"]
VOCAB6 = gE["VOCAB"]
CFG27 = gS["CFG"]
CFGE = gE["CFG"]
STEPS_STRONG = 10000

def n_params(m):
    return sum(p.numel() for p in m.parameters())

class StrongTF(nn.Module):
    def __init__(self, vocab, d_model=128, n_layers=4, n_heads=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.d_model = d_model
        layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                           dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab)
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
        mask = nn.Transformer.generate_square_subsequent_mask(L).to(DEVICE)
        return self.head(self.enc(h, mask))

# class SSM16_1 for the hard tasks
class SSM16_1(nn.Module):
    def __init__(self):
        super().__init__()
        d = 16
        self.emb = nn.Embedding(VOCAB27, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.blocks = nn.ModuleList([gS["SSMBlock"](d) for _ in range(1)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB27)
        self.head.weight = self.emb.weight

    def forward(self, x):
        h = self.emb(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h))

# ---------------------------------------------------------------- training
def train_model(model, gen_pair, steps, seed=0, tag=""):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rng = random.Random(seed * 1000 + 17)
    vocab = model.emb.weight.shape[0]
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = gen_pair(rng)
        loss = F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        if step % 2000 == 0:
            print(f"  [{tag}] step {step}/{steps} CE {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    return model

def gen_pair_hard(rng):
    xd, yd, _ = gS["gen_dyck10"](CFG27["batch"] // 2, CFG27["train_len"], rng)
    xm, ym, _ = gS["gen_mod7"](CFG27["batch"] // 2, CFG27["train_len"], rng)
    return torch.cat([xm, xd]), torch.cat([ym, yd])

def gen_pair_echo(rng):
    x, y, o = gE["gen_echo"](CFGE["batch"], CFGE["train_len"], rng)
    return x, y

@torch.no_grad()
def eval_hard(model, task, L, reps=2):
    return gS["eval_dce"](model, task, L, reps)

@torch.no_grad()
def eval_echo_type(model, L=4096, reps=2):
    model.eval()
    bs = max(1, min(4, 4096 // L))
    per = {t: [0.0, 0] for t in range(VOCAB6)}
    for i in range(reps):
        rng = random.Random(700_000 + L + i)
        x, y, o = gE["gen_echo"](bs, L, rng)
        lp = F.log_softmax(model(x), -1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
        for t in range(VOCAB6):
            m = (y == t)
            per[t][0] += nll[m].sum().item(); per[t][1] += int(m.sum())
    echo_n = per[3][1] + per[4][1] + per[5][1]
    return (per[3][0] + per[4][0] + per[5][0]) / max(1, echo_n)

# ---------------------------------------------------------------- experiment
RESULTS = {}

print("=" * 76, flush=True)
print("AXIS 1: hard tasks (dyck10 + mod7), eval 64/512/2048/4096", flush=True)
torch.manual_seed(0)
ssm = SSM16_1()
print(f"[arm] ssm_d16_1 params={n_params(ssm)} (2500 steps, reference)", flush=True)
train_model(ssm, gen_pair_hard, CFG27["steps"], 0, "ssm ref")
r = {f"{t}@{L}": round(eval_hard(ssm, t, L), 4) for t in ("mod7", "dyck10")
     for L in CFG27["eval_lens"]}
RESULTS["ssm_d16_1 (3,211p, 2.5k)"] = r
print(f"  {r}", flush=True)
del ssm

torch.manual_seed(0)
tf = StrongTF(VOCAB27)
print(f"[arm] TF_STRONG params={n_params(tf)} (10k steps, 8x compute)", flush=True)
train_model(tf, gen_pair_hard, STEPS_STRONG, 0, "tf strong")
r = {f"{t}@{L}": round(eval_hard(tf, t, L), 4) for t in ("mod7", "dyck10")
     for L in CFG27["eval_lens"]}
RESULTS["TF_STRONG (790k, 10k)"] = r
print(f"  {r}", flush=True)
del tf

print("=" * 76, flush=True)
print("AXIS 2: Dyck-echo (non-regular), echo-dCE @4096 (true oracle = 0)", flush=True)
torch.manual_seed(0)
tf = StrongTF(VOCAB6)
print(f"[arm] TF_STRONG (echo vocab) params={n_params(tf)} (10k steps)", flush=True)
train_model(tf, gen_pair_echo, STEPS_STRONG, 0, "tf strong echo")
RESULTS["TF_STRONG echo-dCE@4096"] = round(eval_echo_type(tf), 4)
print(f"  {RESULTS['TF_STRONG echo-dCE@4096']}", flush=True)
del tf

# references (frozen)
torch.manual_seed(0)
hero = gE["EchoModel"]()
hero.load_state_dict(torch.load("dyke_s0.pt", weights_only=True))
RESULTS["hero_ref (2,974p) echo-dCE@4096"] = round(eval_echo_type(hero), 4)
print(f"  {RESULTS['hero_ref (2,974p) echo-dCE@4096']}", flush=True)

print("\n" + "=" * 76)
print("STRONG-TF RESULTS (dCE nats/token; 0 = oracle)")
print("=" * 76)
for k, v in RESULTS.items():
    print(f"{k}", flush=True)
    if isinstance(v, dict):
        for kk, vv in v.items():
            print(f"    {kk:<14} {vv}", flush=True)
    else:
        print(f"    {v}", flush=True)
print("=" * 76)
final = {"tag": "ARC2-C11-STRONG-TF", "runs": RESULTS,
         "note": "TF_STRONG = d128/4L/10k steps (~790k params, 8x micro TF)",
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
