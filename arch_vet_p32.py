"""ARCH-VET P32 (cycle 67) — KRB under COLLISION / OVERFLOW and CONTENT
(two-token) KEYS: where does exact integer addressing break, and does a
learned overwrite policy or a second hash table recover it?

Prior art (2026-09-11): Raven arXiv 2607.25357 (fixed slots, learned
sparse routing + selective decay — softmax addressing); TRIM-KV arXiv
2512.03324 (learned retention gates for bounded KV eviction); classic
hashing (arXiv 1509.05472) — collisions handled by multiple tables.
Our KRB is exact single-table hashing. P32 tests the three failure modes
the literature predicts for hash-addressed memory and the two cheapest
exact remedies:
  REGIME R1 (baseline)  8 keys / 8 slots (P31 setting; collision-free)
  REGIME R2 (collision) 16 keys / 8 slots, identity mod hash -> pairs
                        of keys share a slot; n<=4 bindings per task
                        (collision inside one task with prob ~1-(7/8)^C(n,2))
  REGIME R3 (overflow)  8 keys / 4 slots, n up to 6 bindings -> guaranteed
                        overflow: more live bindings than slots
  REGIME R4 (content keys) key = ORDERED PAIR of tokens (k_a k_b) from
                        4x4=16 composite keys, 8 slots; the address must
                        be a function of BOTH tokens (the single-token
                        hash is blind to the pair) -> KRB2 variant hashes
                        the (prev, cur) token pair exactly.
VARIANTS: KRB (P31 exact overwrite), KRB-2T (two hash tables with
different exact hashes — write both, read gate picks via learned
2-way soft combine), KRB-PAIR (composite (prev,cur) hash for R4).
Controls: VETDCC and TF-ALiBi (P31's strongest TF) on each regime.
Eval: value acc at hard gap for n in the regime's range + n OOD.
Prediction: R1 1.0 (replicates); R2 degrades exactly by the collision
rate for KRB, near-1.0 for KRB-2T; R3 all exact variants drop to
~slots/n (physics), TF may match; R4 KRB ~1/4 (blind), KRB-PAIR ~1.0.
Tag ARCH-VET-LM-P32.
"""
import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_lm as lm, arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p23 as p23, arch_vet_p31 as p31
V = p19.V; BOS, EOS, A, T = p19.BOS, p19.EOS, p19.A, p19.T_TASK; fill = p19.fill_tok
K16 = list(range(p19.TRACK, p19.TRACK + 8)) + list(range(p19.KEYS, p19.KEYS + 4)) + list(range(p19.VALS, p19.VALS + 4))   # 16 key tokens
V8 = list(range(p19.MANS, p19.MANS + 3)) + list(range(p19.BRK, p19.BRK + 4)) + [p19.ONE]                                  # 8 value tokens (disjoint)
KA = list(range(p19.KEYS, p19.KEYS + 4)); KB = list(range(p19.VALS, p19.VALS + 4))                                          # composite key parts


def regime(name):
    if name == "R1": return dict(keys=K16[:8], vals=V8, n_lo=1, n_hi=4, comp=False, slots=8, n_eval=(1, 2, 3, 4, 6, 8))
    if name == "R2": return dict(keys=K16, vals=V8, n_lo=1, n_hi=4, comp=False, slots=8, n_eval=(1, 2, 3, 4, 6, 8))
    if name == "R3": return dict(keys=K16[:8], vals=V8, n_lo=1, n_hi=6, comp=False, slots=4, n_eval=(1, 2, 3, 4, 5, 6, 8))
    if name == "R4": return dict(keys=[(a, b) for a in KA for b in KB], vals=V8, n_lo=1, n_hi=4, comp=True, slots=8, n_eval=(1, 2, 3, 4, 6, 8))


def gen_stream(rng, R, L=256, hard=False, n_fixed=None):
    glo, ghi = (24, 48) if hard else (4, 12); x = [BOS]
    while len(x) < L:
        n = n_fixed or rng.randrange(R["n_lo"], R["n_hi"] + 1)
        ks = rng.sample(R["keys"], n); vs = [rng.choice(R["vals"]) for _ in ks]
        kt = (lambda k: list(k)) if R["comp"] else (lambda k: [k])
        order = list(range(n)); rng.shuffle(order)
        seg = [T, T] + [t for i in range(n) for t in kt(ks[i]) + [vs[i]]] + [fill(rng) for _ in range(rng.randrange(glo, ghi + 1))] + [t for i in order for t in [A] + kt(ks[i]) + [vs[i]]]
        if len(x) + len(seg) > L: x += [fill(rng)] * (L - len(x)); break
        x += seg
    x = x[:L]; x.append(EOS); return x


@torch.no_grad()
def acc(model, R, n_streams, L, rng, hard, n_fixed):
    ok = tot = 0; klen = 2 if R["comp"] else 1
    for _ in range(n_streams):
        x = gen_stream(rng, R, L, hard, n_fixed); pred = model(torch.tensor(x[:L]).unsqueeze(0)).argmax(-1).squeeze(0)
        for t in range(1, L - klen - 1):
            if x[t] == A: tot += 1; ok += int(int(pred[t + klen]) == x[t + klen + 1])
    return round(ok / max(1, tot), 4)


class KRB(p9.VETDCC):
    """Generalised exact KRB. mode: 'one' (single table), 'two' (two tables,
    hashes h1=id mod S, h2=(id*5+3) mod S, learned soft combine on read),
    'pair' (address = hash of (prev,cur) pair; write/read on the second
    key token)."""
    def __init__(self, V, d, key_toks, slots, mode="one", k=8, K=8):
        super().__init__(V, d, k=k, K=K); self.S, self.mode = slots, mode
        kid = torch.full((V,), -1, dtype=torch.long)
        flat = sorted({t for kk in key_toks for t in (kk if isinstance(kk, tuple) else (kk,))})
        for i, t in enumerate(flat): kid[t] = i
        self.register_buffer("kid", kid); self.nk = len(flat)
        self.Wread = nn.Linear(d + k, 1); nn.init.constant_(self.Wread.bias, -1.0)
        if mode == "two": self.Wsel = nn.Linear(d + k, 2)

    def addr(self, xid, prev_id):
        i = self.kid[xid]
        if self.mode == "pair":
            j = self.kid[prev_id]; valid = (i >= 0) & (j >= 0)
            return torch.where(valid, (j * 7 + i) % self.S, torch.full_like(i, -1)), None
        h1 = torch.where(i >= 0, i % self.S, torch.full_like(i, -1))
        h2 = torch.where(i >= 0, (i * 5 + 3) % self.S, torch.full_like(i, -1)) if self.mode == "two" else None
        return h1, h2

    def forward(self, x):
        B, L = x.shape; e = self.E(x); d = self.d; ar = torch.arange(B)
        R = torch.zeros(B, d); s = torch.full((B, self.k), 1.0 / self.k)
        buf = torch.zeros(B, self.K, d); valid = torch.zeros(B, self.K, dtype=torch.bool)
        c = torch.zeros(B, dtype=torch.long); dep = torch.zeros(B, dtype=torch.long)
        nt = 2 if self.mode == "two" else 1
        slots = torch.zeros(B, nt, self.S, d); pend = torch.full((B, nt), -1, dtype=torch.long)
        after_A = torch.zeros(B, dtype=torch.bool); prev = torch.zeros(B, dtype=torch.long); lg = torch.empty(B, L, V)
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            is_one = xid == p9.ONE; is_task = xid == T
            c = torch.where(is_task, 0, torch.where(is_one, (c + 1) % p9.M_MOD, c))
            is_open = (xid == p9.BRK) | (xid == p9.BRK + 1); is_close = (xid == p9.BRK + 2) | (xid == p9.BRK + 3)
            dep = torch.where(is_task, 0, torch.where(is_open, (dep + 1).clamp(max=p9.D_CLAMP), dep)); dep = torch.where(is_close, (dep - 1).clamp(min=0), dep)
            mod_oh = F.one_hot(c, p9.M_MOD).float(); depth_oh = F.one_hot(dep, p9.D_CLAMP + 1).float()
            h1, h2 = self.addr(xid, prev); hs = [h1] + ([h2] if h2 is not None else [])
            # pending writes (value token arrives now)
            for ti in range(nt):
                wm = pend[:, ti] >= 0
                if wm.any():
                    idx = pend[:, ti].clamp(min=0); upd = slots.clone()
                    upd[ar, ti, idx] = torch.where(wm.unsqueeze(-1), xt, slots[ar, ti, idx]); slots = upd
            is_addr = h1 >= 0
            for ti in range(nt): pend[:, ti] = torch.where(is_addr & ~after_A, hs[ti], torch.full_like(h1, -1))
            after_A = torch.where(xid == A, torch.ones_like(after_A), torch.where(is_addr | (self.kid[xid] >= 0), after_A, torch.zeros_like(after_A)))
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1); w = torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt); R = a * R + w
            rg = torch.sigmoid(self.Wread(torch.cat([s, xt], -1))); read_on = (is_addr & after_A).float().unsqueeze(-1) * rg
            cands = [slots[ar, ti, hs[ti].clamp(min=0)] for ti in range(nt)]
            if nt == 2:
                sel = F.softmax(self.Wsel(torch.cat([s, xt], -1)), -1); cand = sel[:, :1] * cands[0] + sel[:, 1:] * cands[1]
            else: cand = cands[0]
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
            prev = xid
        return lg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--regimes", default="R1,R2,R3,R4"); ap.add_argument("--arms", default="KRB,KRB-2T,KRB-PAIR,VETDCC,TF-alibi")
    ap.add_argument("--seed", type=int, default=111); ap.add_argument("--steps", type=int, default=3000); a = ap.parse_args()
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P32", "protocol": __doc__[:1900], "seed": a.seed, "arms": {}}
    for rn in a.regimes.split(","):
        R = regime(rn); rng = random.Random(12345); pool = [torch.tensor(gen_stream(rng, R, 256)) for _ in range(512)]
        for arm in a.arms.split(","):
            if arm == "KRB-PAIR" and rn != "R4": continue
            if arm == "KRB-2T" and rn == "R4": continue
            torch.manual_seed(a.seed)
            m = {"KRB": lambda: KRB(V, 24, R["keys"], R["slots"], "pair" if R["comp"] else "one"),
                 "KRB-2T": lambda: KRB(V, 24, R["keys"], R["slots"], "two"),
                 "KRB-PAIR": lambda: KRB(V, 24, R["keys"], R["slots"], "pair"),
                 "VETDCC": lambda: p9.VETDCC(V, 24, k=8, K=8), "TF-alibi": lambda: p23.TFCtrl(V, pe="alibi")}[arm]()
            if arm == "KRB" and R["comp"]: continue   # single-token KRB is undefined on composite keys; KRB-PAIR is the variant
            name = f"{rn}/{arm}"; print(f"[p32] {name} params {p19.n_params(m)} slots {R['slots']} keys {len(R['keys'])}", flush=True)
            p19.train_arm(f"P32-{rn}-{arm}", m, pool, a.steps, 8); m.eval()
            r = {f"n{n}_hard": acc(m, R, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in R["n_eval"]}
            r["n_train_mix_hard"] = acc(m, R, 12, 256, random.Random(700), True, None); r["params"] = p19.n_params(m)
            print(f"[p32 {name}] {r}", flush=True); out["arms"][name] = r
            torch.save({"sd": m.state_dict()}, f"p21_ckpt/P32_{rn}_{arm}_s{a.seed}.pt")
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P32] DONE", flush=True)


if __name__ == "__main__":
    main()
