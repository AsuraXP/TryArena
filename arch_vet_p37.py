"""ARCH-VET P37 (cycle 74) — CONSUMED-FIRST EVICTION + balanced universal hash
for the learned-predicate KRB: closing the gap to the hand-wired ceiling
WITHOUT any grammar knowledge.

Diagnosis (C74, trained P35 SEEN s111 on hard-gap R2): predicate is
perfect (recall 1.0, fp 0/0) and 0 stale reads; ALL misses are
EVICTIONS (n4 20/160, n8 90/240) — the bank is thrashed because no
per-task reset exists (hand-wired KRB-TAG resets on T_TASK) and the
universal hash of seed 7 covers only 6/8 cells for the 16 keys.
Oracle (perfect predicate, 1-kick cuckoo, no reset):
   hash seed 7  : n4 .884 n8 .694   |  + consumed-first: .977 / .768
   hash seed 85 : n4 .918 n8 .767   |  + consumed-first: .986 / .842
   hand-wired reset ceiling (linear hashes): .975 / .855
Mutations (both grammar-free):
  (a) CONSUMED-FIRST eviction: one "used" bit per slot, set when the
      slot is read via a verified hit, cleared on write; a used slot
      counts as free for placement/kick. Prior art: Attendre arXiv
      2401.04881 (LRA/LFA eviction by attention usage — soft, batch-
      level, for KV caches); here the analogue is exact, 1 bit, O(1).
      A bound answered once is exactly what this stream never needs
      again; the model learns nothing about that — the physics does it.
  (b) hash seed chosen by VOCABULARY-level cell balance only (seed 85:
      min cell load 4/5 of 48 tokens, 35 distinct (h1,h2) pairs, 3
      h1==h2) — no key list consulted.
Everything else = P35e/SEEN. Regimes R2 (and R1), 4000 steps, seeds
111/222/333. Controls: P35 SEEN R2 .919/.925/.919 n4, .726/.733/.729 n8.
PREDICTION: R2 n8 >= .80 on 3/3 seeds, n4 >= .95; R1 unchanged or up.
Tag ARCH-VET-LM-P37. Arms: --arm CF (a only, seed 7) | CF85 (a+b).
"""

import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p32 as p32
V = p19.V


def hindsight_labels(x, W=128):
    """lab[b,t]=1 iff bigram (x_t,x_{t+1}) recurs at some u in (t, t+W]. Grammar-agnostic."""
    B, L = x.shape; a = x[:, :-1]; b = x[:, 1:]; n = L - 1
    eq = (a[:, :, None] == a[:, None, :]) & (b[:, :, None] == b[:, None, :])
    i = torch.arange(n); fut = (i[None, :] > i[:, None]) & (i[None, :] <= i[:, None] + W); past = (i[None, :] < i[:, None]) & (i[None, :] >= i[:, None] - W)
    lab = ((eq & fut).any(-1) & ~(eq & past).any(-1)).float()
    return F.pad(lab, (0, 1))


class KRBHind(p9.VETDCC):
    def __init__(self, V, d, slots, k=8, K=8, seen=True, cf=True, hseed=85):
        super().__init__(V, d, k=k, K=K); self.S = slots; self.seen = seen; self.cf = cf
        g = torch.Generator().manual_seed(hseed); self.register_buffer("HF1", torch.randn(V, slots, generator=g)); self.register_buffer("HF2", torch.randn(V, slots, generator=g))
        self.Wk = nn.Linear(d + k + (1 if seen else 0), 1); self.Wread = nn.Linear(d + k, 1); nn.init.constant_(self.Wread.bias, 0.0)
        self.kg_logits = None

    def forward(self, x):
        B, L = x.shape; e = self.E(x); d = self.d; ar = torch.arange(B); S = self.S
        R = torch.zeros(B, d); s = torch.full((B, self.k), 1.0 / self.k)
        buf = torch.zeros(B, self.K, d); valid = torch.zeros(B, self.K, dtype=torch.bool)
        vals = torch.zeros(B, S, d); tags = torch.full((B, S), -1, dtype=torch.long); used = torch.zeros(B, S, dtype=torch.bool)
        H1 = self.HF1[x].argmax(-1); H2 = self.HF2[x].argmax(-1)          # B,L fixed universal hashes
        kgl_list = []; kg_prev = torch.zeros(B)
        lg = torch.empty(B, L, V)
        mod_oh = torch.zeros(B, p9.M_MOD); depth_oh = torch.zeros(B, p9.D_CLAMP + 1); mod_oh[:, 0] = 1; depth_oh[:, 0] = 1
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            if t > 0:   # write value x_t under key x_{t-1} with soft strength kg_{t-1}
                pk = x[:, t - 1]; w = kg_prev; i1 = H1[:, t - 1]; i2 = H2[:, t - 1]
                t1 = tags[ar, i1]; t2 = tags[ar, i2]
                fr1 = used[ar, i1] if self.cf else torch.zeros_like(t1, dtype=torch.bool); fr2 = used[ar, i2] if self.cf else fr1
                use1 = (t1 == -1) | (t1 == pk) | fr1; use2 = ~use1 & ((t2 == -1) | (t2 == pk) | fr2)
                cell = torch.where(use1, i1, torch.where(use2, i2, i1))
                wm = w > 0.5
                kick = wm & ~use1 & ~use2
                if kick.any():
                    occ = t1.clamp(min=0); a1 = self.HF1[occ].argmax(-1); a2 = self.HF2[occ].argmax(-1); alt = torch.where(a1 == i1, a2, a1)
                    mv = kick & ((tags[ar, alt] == -1) | (used[ar, alt] if self.cf else False))
                    if mv.any():
                        v2 = vals.clone(); g2 = tags.clone(); v2[ar[mv], alt[mv]] = vals[ar[mv], i1[mv]]; g2[ar[mv], alt[mv]] = occ[mv]; vals, tags = v2, g2; u2 = used.clone(); u2[ar[mv], alt[mv]] = used[ar[mv], i1[mv]]; used = u2
                wv = (F.one_hot(cell, S).float() * wm.float().unsqueeze(-1)).unsqueeze(-1)  # P35c: hard overwrite
                vals = vals * (1 - wv) + wv * e[:, t].unsqueeze(1)
                if wm.any():
                    g2 = tags.clone(); g2[ar[wm], cell[wm]] = pk[wm]; tags = g2; u2 = used.clone(); u2[ar[wm], cell[wm]] = False; used = u2
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            i1 = H1[:, t]; i2 = H2[:, t]; seen_t = ((tags[ar, i1] == xid) | (tags[ar, i2] == xid)).float().unsqueeze(-1)
            kl = self.Wk(torch.cat([s, xt] + ([seen_t] if self.seen else []), -1)).squeeze(-1); kgl_list.append(kl); kg_prev = torch.sigmoid(kl)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1); R = a * R + torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            i1 = H1[:, t]; i2 = H2[:, t]; m1 = tags[ar, i1] == xid; m2 = tags[ar, i2] == xid
            cand = torch.where(m1.unsqueeze(-1), vals[ar, i1], vals[ar, i2]); hit = (m1 | m2).float().unsqueeze(-1)
            rg = torch.sigmoid(self.Wread(torch.cat([s, xt], -1))); read_on = hit * (1 - kg_prev).unsqueeze(-1)  # P35e: no learned read gate
            R = (1 - read_on) * R + read_on * cand
            rd = (read_on.squeeze(-1) > 0.5); rc = torch.where(m1, i1, i2)
            if self.cf and rd.any():
                u2 = used.clone(); u2[ar[rd], rc[rd]] = True; used = u2
            g = torch.sigmoid(self.Wg(torch.cat([s, xt], -1))); push = (g > 0.5) + (g - g.detach())
            buf = torch.roll(buf, 1, dims=1); buf[:, 0] = xt * push; valid = torch.roll(valid, 1, dims=1); valid[:, 0] = (g > 0.5).squeeze(-1)
            y = self.Wo(R + xt); feat = torch.stack([buf[:, j] for j in range(self.K)] + [xt], 1)
            logits = self.head(y) + torch.einsum("bk,bjd,kdv->bv", s, feat, self.M)
            selk = torch.zeros(B, self.K + 1)
            for j in range(self.K):
                newer = sum(valid[:, i].float() for i in range(j)) if j else torch.zeros(B); selk[:, j] = valid[:, j].float() * (newer == 0).float()
            selk[:, self.K] = 1.0
            lg[:, t] = logits + torch.einsum("bs,ksv->bv", selk, self.T) + mod_oh @ self.W_mod + depth_oh @ self.W_depth
        self.kg_logits = torch.stack(kgl_list, 1)
        return lg


def train_arm(name, model, pool, steps, batch=8, lr=3e-3, lam=0.5):
    torch.manual_seed(0); opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train(); t0 = time.time(); n_pool = len(pool)
    for step in range(1, steps + 1):
        sel = [(step * batch + i) % n_pool for i in range(batch)]; x = torch.stack([pool[i] for i in sel]); y = x[:, 1:]
        xin = x[:, :256]; lg = model(xin); ce = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        lab = hindsight_labels(xin); bce = F.binary_cross_entropy_with_logits(model.kg_logits, lab)
        loss = ce + lam * bce; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0: print(f"  [{name}] step {step}/{steps} ce {float(ce):.4f} bce {float(bce):.3f} ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def diag(m, R):
    m.eval(); rng = random.Random(900); xs = torch.stack([torch.tensor(p32.gen_stream(rng, R, 256, True, None)[:256]) for _ in range(8)])
    m(xs); kg = torch.sigmoid(m.kg_logits) > 0.5; keys = set(R["keys"]); isk = torch.tensor([[int(t) in keys for t in p] for p in xs]); wk = isk.clone(); wk[:, 1:] &= xs[:, :-1] != p32.A
    return {"key_recall": round(float(kg[wk].float().mean()), 3), "query_key_fp": round(float(kg[isk & ~wk].float().mean()), 3), "nonkey_fp": round(float(kg[~isk].float().mean()), 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", default="R1,R2"); ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--arm", default="CF85"); a = ap.parse_args()
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P37", "protocol": __doc__[:1900], "seed": a.seed, "steps": a.steps, "lam": a.lam, "arms": {}}
    for rn in a.jobs.split(","):
        R = p32.regime(rn); rng = random.Random(12345); pool = [torch.tensor(p32.gen_stream(rng, R, 256)) for _ in range(512)]
        xs = torch.stack(pool)[:, :256]; lab = hindsight_labels(xs); keys = set(R["keys"]); isk = torch.tensor([[int(t) in keys for t in p[:256]] for p in pool]); wk = isk.clone(); wk[:, 1:] &= xs[:, :-1] != p32.A
        print(f"[p37] {rn} label stats: P(lab|write-key)={float(lab[wk].mean()):.3f} P(lab|query-key)={float(lab[isk & ~wk].mean()):.3f} P(lab|nonkey)={float(lab[~isk].mean()):.3f}", flush=True)
        torch.manual_seed(a.seed); m = KRBHind(V, 24, R["slots"], seen=True, cf=a.arm.startswith("CF"), hseed=(85 if a.arm.endswith("85") else 7)); print(f"[p37] {rn}:KRB-HIND params {p19.n_params(m)}", flush=True)
        train_arm(f"P37-{rn}-{a.arm}", m, pool, a.steps, 8, lam=a.lam); m.eval()
        r = {f"n{n}_hard": p32.acc(m, R, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in R["n_eval"]}
        r["n_train_mix_hard"] = p32.acc(m, R, 12, 256, random.Random(700), True, None); r["params"] = p19.n_params(m); r["diag"] = diag(m, R)
        print(f"[p37 {rn}:{a.arm} s{a.seed}] {r}", flush=True); out["arms"][f"{rn}:{a.arm}"] = r
        torch.save({"sd": m.state_dict()}, f"p21_ckpt/P37_{rn}_{a.arm}_s{a.seed}.pt")
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P37] DONE", flush=True)


if __name__ == "__main__":
    main()
