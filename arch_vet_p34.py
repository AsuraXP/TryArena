"""ARCH-VET P34 (cycle 69) — LEARNED-PREDICATE KRB ("KRB-LEARN"): does the
tagged register bank survive when NOTHING about the grammar is hand-wired?

P33's KRB-TAG hard-codes: (a) which tokens are keys (kid table), (b) the
write phase (pend/after_A logic), (c) the task reset (T_TASK), (d) the
two hash functions h1 = k mod S, h2 = (5k+3) mod S. This is the
"learned-predicate ablation" from the frontier list (threat #3).
KRB-LEARN replaces ALL FOUR with learned components:
  key gate   kg_t = ST-bernoulli(sigmoid(Wk e_t))          [per-token]
  write gate wg_t = ST-bernoulli(sigmoid(Ww [s_t, e_t, has_pend]))
  reset gate rs_t = ST-bernoulli(sigmoid(Wr e_t))
  hash       h1(x), h2(x) = ST-Gumbel-argmax(MLP1(e_x)), (MLP2(e_x)) over S
             cells (two independent learned choice functions = learned
             cuckoo placement; Pagh-Rodler 2001), 1-kick displacement kept.
  read gate  rg_t as in P33 (learned), but the after_A predicate is GONE.
The ONLY exact primitive kept is integer tag equality on read (the
verified-miss mechanism, which is the P33 novelty claim). The tag is the
raw token id (content) — no key numbering is given to the model.
Load-balance aux (mean over batch of cell-usage entropy deficit, weight
0.05) guards the hash against collapse (Raven arXiv 2607.25357 notes
router collapse is guarded only by Gumbel noise; TRIM-KV 2512.03324 is
soft eviction). Straight-through Gumbel-softmax: Jang et al. 2016
(arXiv 1611.01144); Bengio ST estimator 2013 (arXiv 1308.3432).
REGIMES: P32 R1 (8k/8s) and R2 (16k/8s collision), clean token sets,
4000 steps, seed 111. Controls = P33 rows on identical data (KRB-TAG with
oracle predicates; TF-ALiBi 42,672p): R1 KRB-TAG 1/1/1/1|1/.993;
R2 KRB-TAG 1/1/.99/1|.946/.865, TF .944 n4 / .754 n8.
PREDICTION: R1 >= .95 at n4 and >= .9 at n8 (single hash suffices, only
key/write/read timing must be learned); R2 within .10 of KRB-TAG at n8
if the learned hashes spread keys (usage entropy > 2.5 nats of 2.08 max
per hash -> report), else collapse -> verified misses -> VETDCC floor.
Tag ARCH-VET-LM-P34.
"""
import argparse, json, math, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p32 as p32
V = p19.V


def st_bern(p):
    return (p > 0.5).float() + (p - p.detach())


def st_gumbel(logits, tau=1.0, hard_noise=True):
    if hard_noise:
        g = -torch.log(-torch.log(torch.rand_like(logits).clamp_min(1e-9)))
        logits = logits + g
    y = F.softmax(logits / tau, -1); idx = y.argmax(-1)
    oh = F.one_hot(idx, logits.shape[-1]).float()
    return oh + (y - y.detach()), idx


class KRBLearn(p9.VETDCC):
    def __init__(self, V, d, slots, k=8, K=8):
        super().__init__(V, d, k=k, K=K); self.S = slots
        self.Wk = nn.Linear(d, 1); self.Wr = nn.Linear(d, 1); nn.init.constant_(self.Wr.bias, -2.0)
        self.Wwr = nn.Linear(d + k + 1, 1)
        self.H1 = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, slots))
        self.H2 = nn.Sequential(nn.Linear(d, 16), nn.Tanh(), nn.Linear(16, slots))
        self.Wread = nn.Linear(d + k, 1); nn.init.constant_(self.Wread.bias, -1.0)
        self.aux = torch.zeros(())

    def hashes(self, e):
        noise = self.training
        o1, i1 = st_gumbel(self.H1(e), hard_noise=noise); o2, i2 = st_gumbel(self.H2(e), hard_noise=noise)
        return o1, i1, o2, i2

    def forward(self, x):
        B, L = x.shape; e = self.E(x); d = self.d; ar = torch.arange(B); S = self.S
        R = torch.zeros(B, d); s = torch.full((B, self.k), 1.0 / self.k)
        buf = torch.zeros(B, self.K, d); valid = torch.zeros(B, self.K, dtype=torch.bool)
        vals = torch.zeros(B, S, d); tags = torch.full((B, S), -1, dtype=torch.long)
        pend = torch.full((B,), -1, dtype=torch.long); pend_oh = torch.zeros(B, S); pend_oh2 = torch.zeros(B, S)
        pend_i1 = torch.zeros(B, dtype=torch.long); pend_i2 = torch.zeros(B, dtype=torch.long)
        lg = torch.empty(B, L, V); usage1 = torch.zeros(S); usage2 = torch.zeros(S); nkey = 0.0
        mod_oh = torch.zeros(B, p9.M_MOD); depth_oh = torch.zeros(B, p9.D_CLAMP + 1); mod_oh[:, 0] = 1; depth_oh[:, 0] = 1  # DCC counters inert here (no hand-wired predicates)
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            # ---- learned reset
            rs = st_bern(torch.sigmoid(self.Wr(xt))).squeeze(-1)
            keep = (1 - rs).unsqueeze(-1)
            vals = vals * keep.unsqueeze(-1); tags = torch.where(rs.bool().unsqueeze(-1), torch.full_like(tags, -1), tags)
            # ---- learned key gate + hashes
            kg = st_bern(torch.sigmoid(self.Wk(xt))).squeeze(-1)
            o1, i1, o2, i2 = self.hashes(xt)
            if self.training:
                usage1 = usage1 + (o1 * kg.unsqueeze(-1)).sum(0); usage2 = usage2 + (o2 * kg.unsqueeze(-1)).sum(0); nkey = nkey + kg.sum()
            # ---- learned write: previous key pending, current token = value
            has_pend = (pend >= 0).float().unsqueeze(-1)
            wg = st_bern(torch.sigmoid(self.Wwr(torch.cat([s, xt, has_pend], -1)))).squeeze(-1) * has_pend.squeeze(-1)
            wm = wg > 0.5
            if wm.any():
                t1 = tags[ar, pend_i1]; t2 = tags[ar, pend_i2]
                use1 = (t1 == -1) | (t1 == pend); use2 = ~use1 & ((t2 == -1) | (t2 == pend))
                cell_oh = torch.where(use1.unsqueeze(-1), pend_oh, torch.where(use2.unsqueeze(-1), pend_oh2, pend_oh))
                cell = torch.where(use1, pend_i1, torch.where(use2, pend_i2, pend_i1))
                kick = wm & ~use1 & ~use2
                if kick.any():
                    occ = t1.clamp(min=0); eo = self.E(occ)
                    _, a1, _, a2 = self.hashes(eo); alt = torch.where(a1 == pend_i1, a2, a1)
                    mv = kick & (tags[ar, alt] == -1)
                    if mv.any():
                        v2 = vals.clone(); g2 = tags.clone()
                        v2[ar[mv], alt[mv]] = vals[ar[mv], pend_i1[mv]]; g2[ar[mv], alt[mv]] = occ[mv]; vals, tags = v2, g2
                wv = (wg.unsqueeze(-1) * cell_oh).unsqueeze(-1)  # B,S,1 (ST-differentiable in wg and hash)
                vals = vals * (1 - wv.detach()) + wv * xt.unsqueeze(1)
                g2 = tags.clone(); g2[ar[wm], cell[wm]] = pend[wm]; tags = g2
            pend = torch.where(kg > 0.5, xid, torch.full_like(pend, -1))
            pend_oh, pend_oh2, pend_i1, pend_i2 = o1 * kg.unsqueeze(-1), o2 * kg.unsqueeze(-1), i1, i2
            # ---- controller + register
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1); R = a * R + torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            # ---- verified READ (tag == token id)
            m1 = tags[ar, i1] == xid; m2 = tags[ar, i2] == xid
            c1 = (o1.unsqueeze(-1) * vals).sum(1); c2 = (o2.unsqueeze(-1) * vals).sum(1)
            cand = torch.where(m1.unsqueeze(-1), c1, c2); hit = (m1 | m2).float().unsqueeze(-1)
            rg = torch.sigmoid(self.Wread(torch.cat([s, xt], -1))); read_on = hit * rg
            R = (1 - read_on) * R + read_on * cand
            g = torch.sigmoid(self.Wg(torch.cat([s, xt], -1))); push = (g > 0.5) + (g - g.detach())
            buf = torch.roll(buf, 1, dims=1); buf[:, 0] = xt * push; valid = torch.roll(valid, 1, dims=1); valid[:, 0] = (g > 0.5).squeeze(-1)
            y = self.Wo(R + xt); feat = torch.stack([buf[:, j] for j in range(self.K)] + [xt], 1)
            logits = self.head(y) + torch.einsum("bk,bjd,kdv->bv", s, feat, self.M)
            selk = torch.zeros(B, self.K + 1)
            for j in range(self.K):
                newer = sum(valid[:, i].float() for i in range(j)) if j else torch.zeros(B); selk[:, j] = valid[:, j].float() * (newer == 0).float()
            selk[:, self.K] = 1.0
            lg[:, t] = logits + torch.einsum("bs,ksv->bv", selk, self.T) + mod_oh @ self.W_mod + depth_oh @ self.W_depth
        if self.training:
            aux = 0.0
            for u in (usage1, usage2):
                p = u / (nkey + 1e-6) + 1e-6; aux = aux + (math.log(S) + (p * p.log()).sum())
            self.aux = aux
        return lg


def train_arm(name, model, pool, steps, batch=8, lr=3e-3, lam=0.05):
    torch.manual_seed(0); opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train(); t0 = time.time(); n_pool = len(pool)
    for step in range(1, steps + 1):
        sel = [(step * batch + i) % n_pool for i in range(batch)]; x = torch.stack([pool[i] for i in sel]); y = x[:, 1:]
        lg = model(x[:, :256]); ce = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1)); loss = ce + lam * model.aux
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0: print(f"  [{name}] step {step}/{steps} ce {float(ce):.4f} aux {float(model.aux):.3f} ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def diag(m, R):
    """Learned predicates vs truth: key-gate on key tokens / others; hash usage entropy; reset on T."""
    m.eval(); E = m.E(torch.arange(V)); kg = torch.sigmoid(m.Wk(E)).squeeze(-1) > 0.5; rs = torch.sigmoid(m.Wr(E)).squeeze(-1) > 0.5
    keys = set(R["keys"]); kt = torch.tensor([i in keys for i in range(V)])
    _, i1, _, i2 = m.hashes(E)
    h1k = i1[kt]; h2k = i2[kt]; ent = lambda h: float(-(torch.bincount(h, minlength=m.S).float() / len(h) + 1e-9).mul(torch.log(torch.bincount(h, minlength=m.S).float() / len(h) + 1e-9)).sum())
    return {"key_recall": round(float(kg[kt].float().mean()), 3), "key_fp": round(float(kg[~kt].float().mean()), 3),
            "reset_on_T": bool(rs[p19.T_TASK]), "reset_fp": round(float(rs.float().mean()), 3),
            "h1_ent": round(ent(h1k), 3), "h2_ent": round(ent(h2k), 3), "h1_cells": int(h1k.unique().numel()), "h2_cells": int(h2k.unique().numel()),
            "h1_eq_h2": round(float((h1k == h2k).float().mean()), 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", default="R1,R2"); ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lam", type=float, default=0.05); a = ap.parse_args()
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P34", "protocol": __doc__[:1900], "seed": a.seed, "steps": a.steps, "lam": a.lam, "arms": {}}
    for rn in a.jobs.split(","):
        R = p32.regime(rn); rng = random.Random(12345); pool = [torch.tensor(p32.gen_stream(rng, R, 256)) for _ in range(512)]
        torch.manual_seed(a.seed); m = KRBLearn(V, 24, R["slots"]); print(f"[p34] {rn}:KRB-LEARN params {p19.n_params(m)}", flush=True)
        train_arm(f"P34-{rn}-LEARN", m, pool, a.steps, 8, lam=a.lam); m.eval()
        r = {f"n{n}_hard": p32.acc(m, R, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in R["n_eval"]}
        r["n_train_mix_hard"] = p32.acc(m, R, 12, 256, random.Random(700), True, None); r["params"] = p19.n_params(m); r["diag"] = diag(m, R)
        print(f"[p34 {rn}:KRB-LEARN] {r}", flush=True); out["arms"][rn] = r
        torch.save({"sd": m.state_dict()}, f"p21_ckpt/P34_{rn}_LEARN_s{a.seed}.pt")
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P34] DONE", flush=True)


if __name__ == "__main__":
    main()
