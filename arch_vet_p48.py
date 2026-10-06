#!/usr/bin/env python3
"""C96 / P48 — CORE LANE on TEXT: slope of the slot-binding core (SBC, arch_vet_p47)
vs fair micro-TF vs GRU on natural dialogue bytes (corpus/chat_dialogues_big.txt).
Train: contiguous byte windows L=128, 3000 steps, B=16, identical batches per
size/arm (seeded).  Eval: nats/byte on held-out windows at L=128 (in-range),
512 and 1024 (length OOD; nothing in SBC/GRU depends on position; TF uses
sinusoidal PE so it *can* run longer).  Slope = d(CE)/d(log2 params).
SBC here is the exact model of P47 with V=256 bytes and M=8 slots; the only
loss is CE.  Pre-registered in log.md C96.
"""
import sys, json, random, time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
sys.argv_p48 = list(sys.argv); sys.argv = ["x"]
import arch_vet_lm as A
import arch_vet_p47 as P
sys.argv = sys.argv_p48
torch.set_num_threads(2)

RAW = open("corpus/chat_dialogues_big.txt", "rb").read()
CUT = int(len(RAW) * 0.95); TR, VA = RAW[:CUT], RAW[CUT:]
V = 256


def batch(rng, src, B, L):
    xs = []
    for _ in range(B):
        s = rng.randrange(0, len(src) - L - 1); xs.append(list(src[s:s + L + 1]))
    return torch.tensor(xs)


class GRUCore(nn.Module):
    def __init__(self, d):
        super().__init__(); self.E = nn.Embedding(V, d); self.g = nn.GRU(d, d, batch_first=True); self.h = nn.Linear(d, V)

    def forward(self, x, **kw): return self.h(self.g(self.E(x))[0])


class SBCText(P.SBC):
    def __init__(self, d, M=8):
        nn.Module.__init__(self); self.d, self.M = d, M
        self.E = nn.Embedding(V, d); self.ctrl = nn.GRUCell(d, d)
        self.ka = nn.Parameter(torch.full((2, d), 0.5)); self.qa = nn.Parameter(torch.full((2, d), 0.5))
        self.v = nn.Linear(2 * d, d); self.K0 = nn.Parameter(torch.randn(16, d) * 0.5)
        self.g = nn.Linear(d, 1); self.theta = nn.Parameter(torch.tensor(1.0)); self.head = nn.Linear(2 * d + 1, V)
        nn.init.constant_(self.g.bias, P.GATE_INIT)


def make(arm, d):
    return A.TFMicro(V, d) if arm == "TF" else GRUCore(d) if arm == "GRU" else SBCText(d)


def ce(logits, x): return F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))


@torch.no_grad()
def val(m, L, n=8, arm="GRU"):
    m.eval(); rng = random.Random(10_000 + L); tot = 0.0
    for _ in range(n):
        x = batch(rng, VA, 4, L)
        lg = m(x, tau=0.05) if arm == "SBC" else m(x); tot += ce(lg, x).item()
    m.train(); return round(tot / n, 4)


def train(arm, d, seed, steps, B, L, log):
    torch.manual_seed(seed); rng = random.Random(seed); m = make(arm, d)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time()
    for s in range(steps):
        x = batch(rng, TR, B, L)
        if arm == "SBC":
            tau = (0.1) ** (s / max(steps - 1, 1)); gmin = P.GMIN0 * (1 - s / max(steps - 1, 1))
            lg = m(x, tau=tau, gmin=gmin)
        else:
            lg = m(x)
        loss = ce(lg, x); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 500 == 0 or s == steps - 1: log(f"  [{arm} d{d} s{seed}] step {s} ce {loss.item():.3f} ({time.time()-t0:.0f}s)")
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="TF,GRU,SBC"); ap.add_argument("--ds", default="16,32,64")
    ap.add_argument("--seeds", default="1"); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--B", type=int, default=16); ap.add_argument("--L", type=int, default=128)
    a = ap.parse_args(); log = lambda *x: print(*x, flush=True)
    res = {"tag": "ARCH-VET-LM-P48", "protocol": __doc__[:1200], "runs": [], "slopes": {}}
    for arm in a.arms.split(","):
        for d in map(int, a.ds.split(",")):
            for seed in map(int, a.seeds.split(",")):
                m = train(arm, d, seed, a.steps, a.B, a.L, log)
                r = {"arm": arm, "d": d, "seed": seed, "params": P.T.n_params(m), "ce128": val(m, 128, arm=arm), "ce512": val(m, 512, arm=arm), "ce1024": val(m, 1024, n=4, arm=arm)}
                if arm == "SBC":
                    with torch.no_grad():
                        x = batch(random.Random(5), VA, 4, 256); _, g, occ = m(x, tau=0.05, states=True); r["gate_mean"] = round(g.mean().item(), 3); r["gate_frac_on"] = round((g > 0.5).float().mean().item(), 3)
                res["runs"].append(r); log(f"[p48 {arm} d{d} s{seed}] params {r['params']} ce128 {r['ce128']} ce512 {r['ce512']} ce1024 {r['ce1024']} " + (f"gate {r['gate_mean']}/{r['gate_frac_on']}" if arm == "SBC" else ""))
        rs = [r for r in res["runs"] if r["arm"] == arm]
        if len(rs) >= 2:
            xs = [math.log2(r["params"]) for r in rs]
            for met in ["ce128", "ce512", "ce1024"]:
                ys = [r[met] for r in rs]; mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
                res["slopes"][f"{arm}_{met}"] = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9), 4)
    log("[P48] slopes " + json.dumps(res["slopes"]))
    with open("log.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
    log("[P48] DONE")


if __name__ == "__main__":
    main()
