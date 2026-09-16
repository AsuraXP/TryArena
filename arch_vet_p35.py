"""ARCH-VET P35 (cycle 70) — HINDSIGHT-SUPERVISED KEY PREDICATE ("KRB-HIND"):
last attack on frontier threat #3 (learned predicates) after P34/P34b.

P34 law: a hard binary write gate whose payoff arrives only through a
downstream EXACT tag-equality read gets no usable straight-through
gradient (collapses open or closed). Fix tested here: give the ONE
learned predicate ("is this token a key?") a DENSE, GRAMMAR-AGNOSTIC
self-supervised target computed from the stream itself in hindsight:
    lab_t = 1  iff  bigram (x_t,x_{t+1}) recurs at some u in (t, t+128]
                    AND did not occur in [t-128, t)   (first occurrence)
i.e. "this binding is NEW and is later queried". No key list, no phase
markers, no task token is given. Measured label quality on the train
pools: P(lab|write-phase key) .86/.93 (R1/R2), P(lab|query-phase key)
0, P(lab|filler) .20 — NOISY on purpose; the bank must tolerate thrash.
Prior art (searched 2026-09-16): Hindsight Memory-PRM arXiv 2608.29605
(hindsight credit for memory write ops in LLM agents; per-op utility
targets from later retrieval logs) — same principle at agent scale, no
exact bank, no micro-scale result. Universal hashing Carter-Wegman 1979
(frozen random projection of token-id one-hot, as P34b FIXH).
Everything else as P34b: write = soft strength sigmoid(Wk e_{t-1}) (no
binarisation on the value path; tag written when > .5), no reset gate
(P33 no-reset oracle R2 .96/.89/.85/.79/.73/.67 = ceiling here), tag-
verified read with learned soft read gate. Loss = CE + 0.5 BCE(kg, lab).
Regimes P32 R1/R2 clean, 4000 steps, seed 111. Controls: KRB-TAG (hand-
wired predicates) R1 1/1/1/1|1/.993, R2 1/1/.99/1|.946/.865; P34b FIXH
R1 .964/.722/.627/.494|.366/.345; TF-ALiBi R2 .944 n4 / .754 n8.
P35 (first run): gate = f(e_t) only -> key_recall 0, BCE .53: the same key
token is label 1 in write phase and 0 in query phase, so a content-only
gate saturates at p=.5 and never writes (design flaw, logged). P35b: gate
= f(s_t, e_t) (controller state = learned context), otherwise identical.
P35b: predicate LEARNED (write-key recall 1.0, query/filler fp 0; bank
hits 14/16 queries on a probe stream) but the learned READ gate closed
(bias -1.85, rg~0): soft-strength writes (w in (.5,1)) blend old/new
value vectors, so early reads were corrupt and the read gate learned to
distrust the bank (R1 .57 n4). P35c: value write is HARD (full overwrite
when the gate fires; gate trains through the hindsight BCE only), read
gate bias init 0. Nothing else changes.
P35c: predicate still perfect, acc unchanged (R1 .56 n4); read gate rg
= 0 at ALL positions (bias drifted to -.97). Cause: without P33's
after_A mask the bank HITS in the write phase too (the key is still
bound from an earlier task, no reset) and injects the STALE value right
before the NEW value must be predicted -> reads are net-harmful early ->
gate closes. P35d: couple read to the learned predicate: read_on = hit *
(1 - kg_t) * rg — "a key is either being bound or being queried". kg is
the same hindsight-supervised gate; no grammar is added.
P35d: unchanged (.58 n4). Post-mortem on the trained model: bank content
is CORRECT (14/16 query hits, 0 stale) but the learned read gate rg is
0 everywhere and FORCING rg=1 gives .35 — the value decoder never
trained because no reads ever flowed (rg starts <.5, reads are noise
until the decoder exists, so the gate closes: a chicken-and-egg
bootstrap failure; P33 avoided it because after_A made reads exact from
step 0). P35e: DROP the learnable read gate; read_on = hit * (1 - kg)
— both factors are exact-or-hindsight-supervised, nothing to bootstrap.
PREDICTION: R1 >= .90 n4, >= .80 n8 with key_recall >= .9; R2 >= .80 n4.
If R1 < .70 n4 threat #3 is closed as a scale limit. Tag ARCH-VET-LM-P35.
"""
import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p32 as p32, arch_vet_p34 as p34
V = p19.V


def hindsight_labels(x, W=128):
    """lab[b,t]=1 iff bigram (x_t,x_{t+1}) recurs at some u in (t, t+W]. Grammar-agnostic."""
    B, L = x.shape; a = x[:, :-1]; b = x[:, 1:]; n = L - 1
    eq = (a[:, :, None] == a[:, None, :]) & (b[:, :, None] == b[:, None, :])
    i = torch.arange(n); fut = (i[None, :] > i[:, None]) & (i[None, :] <= i[:, None] + W); past = (i[None, :] < i[:, None]) & (i[None, :] >= i[:, None] - W)
    lab = ((eq & fut).any(-1) & ~(eq & past).any(-1)).float()
    return F.pad(lab, (0, 1))


class KRBHind(p9.VETDCC):
    def __init__(self, V, d, slots, k=8, K=8):
        super().__init__(V, d, k=k, K=K); self.S = slots
        g = torch.Generator().manual_seed(7); self.register_buffer("HF1", torch.randn(V, slots, generator=g)); self.register_buffer("HF2", torch.randn(V, slots, generator=g))
        self.Wk = nn.Linear(d + k, 1); self.Wread = nn.Linear(d + k, 1); nn.init.constant_(self.Wread.bias, 0.0)
        self.kg_logits = None

    def forward(self, x):
        B, L = x.shape; e = self.E(x); d = self.d; ar = torch.arange(B); S = self.S
        R = torch.zeros(B, d); s = torch.full((B, self.k), 1.0 / self.k)
        buf = torch.zeros(B, self.K, d); valid = torch.zeros(B, self.K, dtype=torch.bool)
        vals = torch.zeros(B, S, d); tags = torch.full((B, S), -1, dtype=torch.long)
        H1 = self.HF1[x].argmax(-1); H2 = self.HF2[x].argmax(-1)          # B,L fixed universal hashes
        kgl_list = []; kg_prev = torch.zeros(B)
        lg = torch.empty(B, L, V)
        mod_oh = torch.zeros(B, p9.M_MOD); depth_oh = torch.zeros(B, p9.D_CLAMP + 1); mod_oh[:, 0] = 1; depth_oh[:, 0] = 1
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            if t > 0:   # write value x_t under key x_{t-1} with soft strength kg_{t-1}
                pk = x[:, t - 1]; w = kg_prev; i1 = H1[:, t - 1]; i2 = H2[:, t - 1]
                t1 = tags[ar, i1]; t2 = tags[ar, i2]
                use1 = (t1 == -1) | (t1 == pk); use2 = ~use1 & ((t2 == -1) | (t2 == pk))
                cell = torch.where(use1, i1, torch.where(use2, i2, i1))
                wm = w > 0.5
                kick = wm & ~use1 & ~use2
                if kick.any():
                    occ = t1.clamp(min=0); a1 = self.HF1[occ].argmax(-1); a2 = self.HF2[occ].argmax(-1); alt = torch.where(a1 == i1, a2, a1)
                    mv = kick & (tags[ar, alt] == -1)
                    if mv.any():
                        v2 = vals.clone(); g2 = tags.clone(); v2[ar[mv], alt[mv]] = vals[ar[mv], i1[mv]]; g2[ar[mv], alt[mv]] = occ[mv]; vals, tags = v2, g2
                wv = (F.one_hot(cell, S).float() * wm.float().unsqueeze(-1)).unsqueeze(-1)  # P35c: hard overwrite
                vals = vals * (1 - wv) + wv * e[:, t].unsqueeze(1)
                if wm.any():
                    g2 = tags.clone(); g2[ar[wm], cell[wm]] = pk[wm]; tags = g2
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            kl = self.Wk(torch.cat([s, xt], -1)).squeeze(-1); kgl_list.append(kl); kg_prev = torch.sigmoid(kl)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1); R = a * R + torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            i1 = H1[:, t]; i2 = H2[:, t]; m1 = tags[ar, i1] == xid; m2 = tags[ar, i2] == xid
            cand = torch.where(m1.unsqueeze(-1), vals[ar, i1], vals[ar, i2]); hit = (m1 | m2).float().unsqueeze(-1)
            rg = torch.sigmoid(self.Wread(torch.cat([s, xt], -1))); read_on = hit * (1 - kg_prev).unsqueeze(-1)  # P35e: no learned read gate
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
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--lam", type=float, default=0.5); a = ap.parse_args()
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P35", "protocol": __doc__[:1900], "seed": a.seed, "steps": a.steps, "lam": a.lam, "arms": {}}
    for rn in a.jobs.split(","):
        R = p32.regime(rn); rng = random.Random(12345); pool = [torch.tensor(p32.gen_stream(rng, R, 256)) for _ in range(512)]
        xs = torch.stack(pool)[:, :256]; lab = hindsight_labels(xs); keys = set(R["keys"]); isk = torch.tensor([[int(t) in keys for t in p[:256]] for p in pool]); wk = isk.clone(); wk[:, 1:] &= xs[:, :-1] != p32.A
        print(f"[p35] {rn} label stats: P(lab|write-key)={float(lab[wk].mean()):.3f} P(lab|query-key)={float(lab[isk & ~wk].mean()):.3f} P(lab|nonkey)={float(lab[~isk].mean()):.3f}", flush=True)
        torch.manual_seed(a.seed); m = KRBHind(V, 24, R["slots"]); print(f"[p35] {rn}:KRB-HIND params {p19.n_params(m)}", flush=True)
        train_arm(f"P35-{rn}", m, pool, a.steps, 8, lam=a.lam); m.eval()
        r = {f"n{n}_hard": p32.acc(m, R, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in R["n_eval"]}
        r["n_train_mix_hard"] = p32.acc(m, R, 12, 256, random.Random(700), True, None); r["params"] = p19.n_params(m); r["diag"] = diag(m, R)
        print(f"[p35 {rn}:KRB-HIND] {r}", flush=True); out["arms"][f"{rn}:HINDe"] = r
        torch.save({"sd": m.state_dict()}, f"p21_ckpt/P35e_{rn}_HIND_s{a.seed}.pt")
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P35] DONE", flush=True)


if __name__ == "__main__":
    main()
