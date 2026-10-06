#!/usr/bin/env python3
"""C94 / P46 — CORE LANE: does a STATE-LEVEL MEANING LOSS make a dense learner
form bindings instead of lookups?  Slope test vs fair micro-TF.

Mechanism (gate-passed, NOVELTY.md B 'what the objective pays for'):
  meaning := what survives distractors and tracks facts.  For every training
  sequence we render the SAME event structure (facts, overwrites, queries)
  twice with different filler (INV pair) and once with one fact value swapped
  (CF pair).  Loss = CE + l_inv * ||n(h_a)-n(h_b)||^2 at aligned anchors
                      + l_cf  * relu(m - ||n(h)-n(h')||^2) at anchors after the swap.
Prior art (searched 2026-10-06): TAR arXiv:1708.01009 penalises adjacent-step
change (opposite intent); CSR arXiv:2509.01544 maximises counterfactual KL on
LLM *outputs* at fine-tune time.  Delta: state-level, from scratch, both
invariance and sensitivity, in a token-prediction learner.  No organs.
Arms: TF (TFMicro, sinusoidal PE), GRU, GRU+INV at d in {16,32,64}; identical
batches (seeded); 1500 steps.  Metrics: answer accuracy in-range / OOD-length /
OOD-count; slope = d(acc)/d(log2 params).
"""
import sys, json, random, time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
sys.argv_p46 = list(sys.argv); sys.argv = ["x"]
import arch_vet_lm as A
sys.argv = sys.argv_p46

K, NV, NF = 8, 10, 16
PAD, Q = 0, 1
KEY0 = 2; VAL0 = KEY0 + K; FIL0 = VAL0 + NV; V = FIL0 + NF   # 36 tokens
torch.set_num_threads(2)


# ---------------- task ----------------
def structure(rng, nfacts, nq, overwrite_p=0.5):
    """events: ('F',k,v) or ('Q',k,ans). Later facts overwrite earlier."""
    keys = rng.sample(range(K), nfacts); ev = []; cur = {}
    for k in keys:
        v = rng.randrange(NV); ev.append(("F", k, v)); cur[k] = v
    # overwrites
    for k in keys:
        if rng.random() < overwrite_p:
            v = rng.randrange(NV); ev.append(("F", k, v)); cur[k] = v
    for _ in range(nq):
        k = rng.choice(keys); ev.append(("Q", k, cur[k]))
    # keep facts order, queries at end (they may also be interleaved: shuffle middle)
    return ev


def render(ev, rng, run_max):
    """tokens + anchor indices (index of last token of each event) + answer positions."""
    toks, anchors, ans = [], [], []
    for e in ev:
        for _ in range(rng.randint(0, run_max)): toks.append(FIL0 + rng.randrange(NF))
        if e[0] == "F":
            toks += [KEY0 + e[1], VAL0 + e[2]]
        else:
            toks += [Q, KEY0 + e[1]]; ans.append(len(toks)); toks.append(VAL0 + e[2])
        anchors.append(len(toks) - 1)
    return toks, anchors, ans


def counterfactual(ev, rng):
    """swap the value of the fact that determines the first query's answer; return (ev', idx_of_swap)."""
    qk = [e for e in ev if e[0] == "Q"][0][1]
    idx = max(i for i, e in enumerate(ev) if e[0] == "F" and e[1] == qk)
    ev2 = list(ev); old = ev2[idx][2]; new = rng.choice([v for v in range(NV) if v != old])
    ev2[idx] = ("F", qk, new)
    cur = {}
    out = []
    for e in ev2:
        if e[0] == "F": cur[e[1]] = e[2]; out.append(e)
        else: out.append(("Q", e[1], cur[e[1]]))
    return out, idx


def pad(seqs, L=None):
    L = L or max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (L - len(s)) for s in seqs])


def make_batch(rng, B, nfacts_rng, nq, run_max, with_pairs):
    evs = [structure(rng, rng.randint(*nfacts_rng), nq) for _ in range(B)]
    a = [render(e, rng, run_max) for e in evs]
    out = {"x": pad([t for t, _, _ in a]), "anch": [an for _, an, _ in a], "ans": [s for _, _, s in a]}
    if with_pairs:
        b = [render(e, rng, run_max) for e in evs]                       # same events, other filler
        cf = [counterfactual(e, rng) for e in evs]
        c = [render(e2, rng, run_max) for e2, _ in cf]
        out.update(xb=pad([t for t, _, _ in b]), anchb=[an for _, an, _ in b],
                   xc=pad([t for t, _, _ in c]), anchc=[an for _, an, _ in c], cfidx=[i for _, i in cf])
    return out


# ---------------- models ----------------
class GRUCore(nn.Module):
    def __init__(self, d):
        super().__init__(); self.E = nn.Embedding(V, d); self.g = nn.GRU(d, d, batch_first=True); self.h = nn.Linear(d, V)

    def forward(self, x, states=False):
        hs, _ = self.g(self.E(x)); return (self.h(hs), hs) if states else self.h(hs)


def n_params(m): return sum(p.numel() for p in m.parameters())


AW = 10.0   # answer-position weight: filler is unpredictable by design; the task lives at the answers
def ce_loss(logits, x, ans=None):
    l = F.cross_entropy(logits[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), ignore_index=PAD, reduction="none").view(x.shape[0], -1)
    w = torch.ones_like(l)
    if ans is not None:
        for i, poss in enumerate(ans):
            for s in poss: w[i, s - 1] = AW
    return (l * w).sum() / w[(x[:, 1:] != PAD)].sum()


def gather(hs, anch):
    return torch.stack([hs[i, torch.tensor(a)] for i, a in enumerate(anch)])      # B x E x d (E equal per batch? no) -> list


def inv_cf_loss(m, bt, margin=1.0):
    logits, hs = m(bt["x"], states=True); _, hb = m(bt["xb"], states=True); _, hc = m(bt["xc"], states=True)
    li, lc, n_i, n_c = 0.0, 0.0, 0, 0
    for i in range(bt["x"].shape[0]):
        ha = F.normalize(hs[i, bt["anch"][i]], dim=-1); hbb = F.normalize(hb[i, bt["anchb"][i]], dim=-1)
        hcc = F.normalize(hc[i, bt["anchc"][i]], dim=-1)
        li = li + ((ha - hbb) ** 2).sum(-1).mean(); n_i += 1
        s = bt["cfidx"][i]
        d = ((ha[s:] - hcc[s:]) ** 2).sum(-1)
        lc = lc + F.relu(margin - d).mean(); n_c += 1
    return ce_loss(logits, bt["x"], bt["ans"]), li / n_i, lc / n_c


@torch.no_grad()
def acc(m, batches):
    m.eval(); c = t = 0
    for bt in batches:
        p = m(bt["x"]).argmax(-1)
        for i, poss in enumerate(bt["ans"]):
            for s in poss: c += int(p[i, s - 1].item() == bt["x"][i, s].item()); t += 1
    m.train(); return c / max(t, 1)


def train(arm, d, seed, steps, B, log):
    torch.manual_seed(seed); rng = random.Random(seed)
    m = A.TFMicro(V, d) if arm == "TF" else GRUCore(d)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time()
    for s in range(steps):
        bt = make_batch(rng, B, (2, 4), 2, 4, with_pairs=(arm == "INV"))
        if arm == "INV":
            ce, li, lc = inv_cf_loss(m, bt); loss = ce + 1.0 * li + 1.0 * lc
        else:
            loss = ce_loss(m(bt["x"]), bt["x"], bt["ans"]); li = lc = torch.tensor(0.0)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 300 == 0 or s == steps - 1:
            log(f"  [{arm} d{d} s{seed}] step {s} loss {loss.item():.3f} inv {float(li):.3f} cf {float(lc):.3f} ({time.time()-t0:.0f}s)")
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="TF,GRU,INV"); ap.add_argument("--ds", default="16,32,64")
    ap.add_argument("--seeds", default="1"); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--B", type=int, default=16)
    a = ap.parse_args(); log = lambda *x: print(*x, flush=True)
    er = random.Random(999); ev_in = [make_batch(er, 16, (2, 4), 2, 4, False) for _ in range(12)]
    er = random.Random(998); ev_len = [make_batch(er, 16, (2, 4), 2, 30, False) for _ in range(12)]
    er = random.Random(997); ev_cnt = [make_batch(er, 16, (8, 8), 2, 4, False) for _ in range(12)]
    er = random.Random(996); ev_both = [make_batch(er, 16, (8, 8), 2, 30, False) for _ in range(12)]
    log(f"[P46] V={V} eval lens in={ev_in[0]['x'].shape[1]} len={ev_len[0]['x'].shape[1]} cnt={ev_cnt[0]['x'].shape[1]} both={ev_both[0]['x'].shape[1]}")
    res = {"tag": "ARCH-VET-LM-P46", "protocol": __doc__[:1500], "runs": []}
    for arm in a.arms.split(","):
        for d in map(int, a.ds.split(",")):
            for seed in map(int, a.seeds.split(",")):
                m = train(arm, d, seed, a.steps, a.B, log)
                r = {"arm": arm, "d": d, "seed": seed, "params": n_params(m), "in": acc(m, ev_in), "len": acc(m, ev_len), "cnt": acc(m, ev_cnt), "both": acc(m, ev_both)}
                res["runs"].append(r); log(f"[p46 {arm} d{d} s{seed}] params {r['params']} in {r['in']:.3f} len {r['len']:.3f} cnt {r['cnt']:.3f} both {r['both']:.3f}")
    # slopes per arm: acc vs log2(params)
    res["slopes"] = {}
    for arm in a.arms.split(","):
        rs = [r for r in res["runs"] if r["arm"] == arm]
        if len(rs) >= 2:
            xs = [math.log2(r["params"]) for r in rs]
            for met in ["in", "len", "cnt", "both"]:
                ys = [r[met] for r in rs]; mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
                sl = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9)
                res["slopes"][f"{arm}_{met}"] = round(sl, 4)
    log("[P46] slopes " + json.dumps(res["slopes"]))
    with open("log.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
    log("[P46] DONE")


if __name__ == "__main__":
    main()
