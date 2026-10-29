"""kaggle_lin_30min.py — ONE-OFF 30-minute GPU run of OUR core only (LIN, arch_vet_lin.py), on ~100 MB of prose.

Paste into a Kaggle notebook cell (Accelerator: GPU T4 x2, Internet: ON):

    !curl -sL https://raw.githubusercontent.com/AsuraXP/TryArena/arena/01a07767-tryarena/kaggle_lin_30min.py -o run.py && python run.py

or, if raw.githubusercontent is blocked:  !git clone -b arena/01a07767-tryarena --depth 1 https://github.com/AsuraXP/TryArena && python TryArena/kaggle_lin_30min.py

What it does (nothing else; no Transformer arm — that comparison stays on Actions per plan):
  1. clones this repo branch + builds the 100 MB prose corpus (build_corpus.py, GitHub-hosted public-domain books)
  2. GPU0: LIN d256 (plain config, as in T2/S1: roles 4, no store flags);  GPU1: LIN d512.  Same batches (same seed).
  3. trains each for TRAIN_MIN minutes (time-boxed, cosine lr on elapsed time), logs held-out CE every 100 steps
     with bytes_seen (data-efficiency curve), then held-out CE at L / 4L / 16L, then fixed-prompt unselected samples
     (temperature 0.8, seed 0), then saves checkpoint + log to /kaggle/working/lin_d{d}.{pt,log}.
  4. prints both logs at the end.  Download /kaggle/working/*.log (and .pt if wanted) and paste the logs back.

Pre-registered (written before running): plain LIN d256 reached 1.313 held-out CE on 5 MB dialogue at 24 MB seen
(CPU).  On 100 MB prose (harder: vocabulary, no templates) we expect d256 ~1.45-1.55 nats/byte after one 30-min
pass, d512 lower by >= .05 if the scaling slope holds; CE at 16L within +.03 of CE at L (flat-length claim);
samples: real words and clause structure, not yet consistent paragraphs.  Falsifiers: d512 not better than d256;
16L worse by > .1; samples with < 80 % dictionary words.
"""
import os, sys, time, json, math, random, subprocess

TRAIN_MIN = float(os.environ.get("TRAIN_MIN", 24))     # training minutes per GPU (evals + samples add ~3-4 min)
B, L = int(os.environ.get("B", 32)), 512
DS = [int(x) for x in os.environ.get("DS", "256,512").split(",")]
BRANCH, REPO = "arena/01a07767-tryarena", "https://github.com/AsuraXP/TryArena"
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")
os.makedirs(WORK, exist_ok=True)

def setup():
    if not os.path.isdir(f"{WORK}/TryArena"):
        subprocess.check_call(["git", "clone", "-q", "-b", BRANCH, "--depth", "1", REPO, f"{WORK}/TryArena"])
    if not os.path.exists(f"{WORK}/prose100.txt"):
        subprocess.check_call([sys.executable, f"{WORK}/TryArena/build_corpus.py", "--out", f"{WORK}/prose100.txt", "--mb", "100", "--src", f"{WORK}/sfgram"])

def worker(d, gpu):
    import torch, torch.nn as nn, torch.nn.functional as F
    sys.path.insert(0, f"{WORK}/TryArena"); import arch_vet_lin as LINM
    dev = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    logf = open(f"{WORK}/lin_d{d}.log", "w")
    def log(*x):
        s = " ".join(str(v) for v in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    raw = open(f"{WORK}/prose100.txt", "rb").read().lower(); cut = int(len(raw) * 0.95); tr, te = raw[:cut], raw[cut:]
    torch.manual_seed(1); rng = random.Random(1)
    m = LINM.LIN(256, d, roles=4).to(dev); npar = sum(p.numel() for p in m.parameters() if p.requires_grad)
    lr0 = 2e-3 if d <= 256 else 1e-3; opt = torch.optim.Adam(m.parameters(), lr=lr0)
    log(f"[LIN d{d}] device {dev} trainable params {npar} B {B} L {L} lr {lr0} corpus {len(raw)/1e6:.1f} MB train_min {TRAIN_MIN}")
    def chunk(src, n, Lc, r):
        return torch.tensor([list(src[o:o + Lc]) for o in [r.randrange(0, len(src) - Lc - 1) for _ in range(n)]], device=dev)
    @torch.no_grad()
    def ev(Lc, n=8, reps=8):
        m.eval(); er = random.Random(999); tot = cnt = 0.0
        for _ in range(reps):
            x = chunk(te, n, Lc, er); lg = m(x)
            tot += F.cross_entropy(lg[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum").item(); cnt += x[:, 1:].numel()
        m.train(); return round(tot / cnt, 4)
    @torch.no_grad()
    def sample(prompt, n=500, temp=0.8, seed=0):
        g = torch.Generator(device=dev).manual_seed(seed); x = torch.tensor([list(prompt.encode())], device=dev); m.eval()
        for _ in range(n):
            lg = m(x)[:, -1] / temp; nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=g); x = torch.cat([x, nxt], 1)
        m.train(); return bytes(x[0].tolist()).decode("utf8", "replace")
    t0 = time.time(); budget = TRAIN_MIN * 60; s = 0; curve = []; b = B
    while True:
        el = time.time() - t0
        if el > budget: break
        frac = el / budget; lr = lr0 * min(1.0, (s + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for gq in opt.param_groups: gq["lr"] = lr
        try:
            x = chunk(tr, b, L, rng); lg = m(x)
            loss = F.cross_entropy(lg[:, :-1].reshape(-1, 256), x[:, 1:].reshape(-1)) + m.aux
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); b = max(1, b // 2); log(f"  OOM -> batch {b}"); continue
        if s % 100 == 0:
            c = ev(L, n=4, reps=4)
            curve.append((s, c)); log(f"  [LIN d{d}] step {s} bytes_seen {s * b * L} train {loss.item():.3f} heldout {c} lr {lr:.2e} ({el:.0f}s)")
        s += 1
    seen = s * b * L
    log(f"[LIN d{d}] trained {s} steps, {seen/1e6:.1f} MB seen in {TRAIN_MIN:.0f} min ({seen/(time.time()-t0)/1e3:.0f} kB/s)")
    r = {"d": d, "params": npar, "steps": s, "bytes_seen": seen, "ce_L": ev(L), "ce_4L": ev(4 * L, n=4), "ce_16L": ev(16 * L, n=2, reps=4)}
    log(f"[RESULT LIN d{d}] params {npar} bytes_seen {seen} heldout CE at L/4L/16L {r['ce_L']} {r['ce_4L']} {r['ce_16L']}")
    for pr in ["the captain looked at the stars and said, \"", "it was a cold morning when ", "she opened the door and "]:
        log(f"--- SAMPLE [LIN d{d}] prompt={pr!r}\n" + sample(pr) + "\n---")
    torch.save({"state": m.state_dict(), "d": d, "result": r}, f"{WORK}/lin_d{d}.pt"); json.dump(r, open(f"{WORK}/lin_d{d}.json", "w"))

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--worker": worker(int(sys.argv[2]), int(sys.argv[3])); sys.exit()
    setup(); import torch; ng = max(1, torch.cuda.device_count()); print("GPUs:", ng, flush=True)
    procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(d), str(i % ng)]) for i, d in enumerate(DS)]
    for p in procs: p.wait()
    for d in DS:
        print(f"\n===== lin_d{d}.log =====", flush=True); print(open(f"{WORK}/lin_d{d}.log").read(), flush=True)
