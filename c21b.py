"""
ARC-2 CYCLE 39 / C21b LM HOST SCALE-UP (d64) — the chatbot axis: real-text language modeling
on the machine's linear host (P11 fluency axis, reopened by operator:
"it also needs to nail it as a chatbot")
============================================================================
What this cycle certifies (honest scope): the SSM host (L-LINEAR-HOST)
does language modeling on REAL mixed text (public-domain English prose +
the program's own code/prose, 1.0MB corpus, 542k BPE tokens, vocab 768)
with LENGTH-INVARIANT loss: CE held at 16384 (256x training length) where
the O(N^2) transformer cannot even allocate. This is the fluency engine
that the chatbot machine (C22+) will carry state + computation on top of.
Model: SSMBlock(d32) (the machine's host, verbatim) + tied embedding
(~35k params = 1/3 of machine v4, 1/23 of the C15 protocol TF).
Protocol: 12000 steps, batch 32, L=256, AdamW 3e-3, clip 1.0, seed 0,
ckpts 3k/6k/9k/12k; per ckpt: val CE @256/1024/4096/16384 (length
invariance is THE claim); final: 2 generation samples (prose + code
prompt) logged for the human eye. NO TF arm (C17 directive; the C8-era
P11 number and ln(V)=6.644 are the cited baselines).
SUCCESS = (i) val CE < 4.0 @256 at 12k (2.2x under the uniform prior);
          (ii) CE @16384 within 1.3x of CE @256 at 12k (no length decay);
          (iii) generations are coherent word sequences (logged).
USAGE: OMP_NUM_THREADS=1 python3 -u lm_host.py
"""
import json, math, os, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
t_start = time.time()

g = {"__name__": "lh"}
exec(open(os.path.join(HERE, "unified.py")).read().split("\nRESULTS = {}")[0], g)
SSMBlock, n_params = g["SSMBlock"], g["n_params"]

from bpe_tok import load_tk
VOCAB, encode, decode = load_tk()
CFG = dict(steps=6000, batch=32, train_len=256, ckpts=[3000, 6000],
           d_model=64)
print(f"[setup] lm-host cfg={CFG} VOCAB={VOCAB}", flush=True)

# ---------------------------------------------------------------- data
with open(os.path.join(HERE, "corpus", "corpus_full.txt"), "rb") as f:
    TOKS = encode(f.read())
N_TRAIN = int(len(TOKS) * 0.9)
print(f"[data] corpus tokens {len(TOKS)} (train {N_TRAIN} / val {len(TOKS)-N_TRAIN})", flush=True)

def sample(batch, length, arr, rng):
    n = len(arr) - length - 1
    xs = ys = None
    rows_x, rows_y = [], []
    for _ in range(batch):
        s = rng.randrange(n)
        w = arr[s:s + length + 1]
        rows_x.append(w[:-1]); rows_y.append(w[1:])
    return torch.tensor(rows_x), torch.tensor(rows_y)

# ---------------------------------------------------------------- model
class LMHost(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.host = SSMBlock(d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB)
        self.head.weight = self.emb.weight            # tied embedding

    def forward(self, x):
        return self.head(self.norm(self.host(self.emb(x))))

# ---------------------------------------------------------------- experiment
def train_step(model, opt, x, y):
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    return loss.item()

@torch.no_grad()
def val_ce(model, L, reps=1, arr=None):
    model.eval()
    arr = arr if arr is not None else TOKS[N_TRAIN:]
    bs = max(1, min(2, 256 * 1024 // L))
    ce = n = 0.0
    for i in range(reps):
        rng = random.Random(800_000 + L + i)
        x, y = sample(bs, L, arr, rng)
        nll = -F.log_softmax(model(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); n += y.numel()
    return round(ce / n, 4)

@torch.no_grad()
def generate(model, prompt, n=160, temp=0.8, seed=1):
    model.eval()
    torch.manual_seed(seed)
    cur = torch.tensor([encode(prompt.encode("utf-8"))], dtype=torch.long)
    for _ in range(n):
        lg = model(cur)[:, -1, :]
        nxt = torch.multinomial(torch.softmax(lg / temp, -1), 1)
        cur = torch.cat([cur, nxt], 1)
    return decode(cur[0].tolist()).decode("utf-8", "replace")

def eval_all(model, tag, results):
    r = {f"ce{L}": val_ce(model, L, 2 if L <= 4096 else 1)
         for L in [256, 1024, 4096, 16384]}
    r["ppl256"] = round(math.exp(r["ce256"]), 2)
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)

RESULTS = {}
torch.manual_seed(0)
m = LMHost()
print(f"[arm] lm host (SSM d32, tied) params={n_params(m)}", flush=True)
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
rng = random.Random(17)
# env-reset recovery: resume from latest on-disk ckpt (weights only),
# fast-forwarding the data rng exactly
START = 1
if os.environ.get("RESUME") == "1":
    existing = [c for c in CFG["ckpts"]
                if os.path.exists(os.path.join(HERE, f"c21b_{c}.pt"))]
    if existing:
        last = max(existing)
        m.load_state_dict(torch.load(os.path.join(HERE, f"c21b_{last}.pt")))
        for s in range(1, last + 1):
            sample(CFG["batch"], CFG["train_len"], TOKS[:N_TRAIN], rng)
        START = last + 1
        print(f"[lm] RESUME from step {last} (data rng fast-forwarded)", flush=True)
t0 = time.time()
ckpts_done = set()
arr = TOKS[:N_TRAIN]
for step in range(START, CFG["steps"] + 1):
    x, y = sample(CFG["batch"], CFG["train_len"], arr, rng)
    lm = train_step(m, opt, x, y)
    if step in CFG["ckpts"] and step not in ckpts_done:
        torch.save(m.state_dict(), os.path.join(HERE, f"c21b_{step}.pt"))
        ckpts_done.add(step)
        print(f"    [lm] checkpoint at step {step}", flush=True)
    if step % 2000 == 0:
        print(f"  [lm s0] step {step}/{CFG['steps']} CE {lm:.4f} ({time.time()-t0:.0f}s)", flush=True)
print(f"[lm] trained in {time.time()-t0:.0f}s", flush=True)
torch.save(m.state_dict(), os.path.join(HERE, "c21b_final.pt"))

print("[eval] val CE by length, per ckpt:", flush=True)
for c in CFG["ckpts"]:
    m.load_state_dict(torch.load(os.path.join(HERE, f"c21b_{c}.pt")))
    eval_all(m, f"lm_{c // 1000}k", RESULTS)
m.load_state_dict(torch.load(os.path.join(HERE, "c21b_final.pt")))
gen1 = generate(m, "It is a truth universally acknowledged, that a single man in possession of a good fortune", 160, 0.8, 1)
gen2 = generate(m, "def add(a, b):\n    return", 160, 0.8, 2)
RESULTS["gen_prose"] = gen1
RESULTS["gen_code"] = gen2
print("\n  [gen prose] " + gen1.replace("\n", " | "), flush=True)
print("  [gen code ] " + gen2.replace("\n", " | "), flush=True)

print("\n" + "=" * 92)
print("LM HOST — real-text fluency on the machine's linear host (P11 axis)")
print("baseline ln(768) = 6.644; C8-era P11 gap measured vs RoPE micro TF")
print("SUCCESS: (i) CE @256 <= 4.0 at 12k; (ii) CE @16384 within 1.3x @256;")
print("(iii) coherent generations.")
print("=" * 92)
for k in [x for x in ["lm_3k", "lm_6k", "lm_9k", "lm_12k"] if x in RESULTS]:
    v = RESULTS[k]
    print(f"{k:<7} ce256 {v['ce256']}  ce1024 {v['ce1024']}  ce4096 {v['ce4096']}  "
          f"ce16384 {v['ce16384']}  ppl256 {v['ppl256']}", flush=True)
print("=" * 92)
final = {"tag": "ARC2-C21B-FLUENCY-D64", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open(os.path.join(HERE, "log.jsonl"), "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
