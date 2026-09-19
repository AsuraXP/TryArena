"""ARCH-VET P38 (cycle 77) — BLOCKED CUCKOO for the learned-predicate KRB.
Hypothesis: the R2 n8 ceiling (.855 unified, oracle .835) is the cuckoo
load-1.0 wall (L-CUCKOO-LOAD-CEILING). Blocked cuckoo hashing (b cells per
bucket, S/b buckets, 2 hash choices) raises the achievable load from 1/2 to
>.8 with b=2 (Dietzfelbinger & Weidling 2007, "Balanced allocation and
dictionaries with tightly packed constant size bins"; Wikipedia: "Using just
2 keys per bucket permits a load factor above 80%"). Stash variant:
Kirsch, Mitzenmacher & Wieder 2008. Oracle (oracle_c77.py, S=8 total cells,
consumed-first, 1 kick): b=1 .986/.835; b=2 .998/.904; b=4 1.000/.962;
S16 b=1 .998/.978. Same slot memory, same parameter count, zero grammar.
Mechanism: write -> first empty/consumed cell of bucket H1, else of H2,
else overwrite cell 0 of H1 (1 kick of victim to its alt bucket). Read ->
exact tag scan over the 2b candidate cells. All O(1) per token.
Arms: BK2 (b=2), BK4 (b=4), BK1 (== P37 CF85R, control).
"""
import argparse, json, os, random, sys, time, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p32 as p32, arch_vet_p37 as p37
V = p19.V


class KRBBlocked(p37.KRBHind):
    def __init__(self, V, d, slots, b=2, hseed=85, **kw):
        super().__init__(V, d, slots, hseed=hseed, decouple=True, **kw); self.b = b; self.NB = slots // b
        g = torch.Generator().manual_seed(hseed); self.register_buffer("HF1", torch.randn(V, self.NB, generator=g)); self.register_buffer("HF2", torch.randn(V, self.NB, generator=g))

    def forward(self, x):
        B, L = x.shape; e = self.E(x); d = self.d; ar = torch.arange(B); NB, b = self.NB, self.b
        R = torch.zeros(B, d); s = torch.full((B, self.k), 1.0 / self.k)
        buf = torch.zeros(B, self.K, d); valid = torch.zeros(B, self.K, dtype=torch.bool)
        vals = torch.zeros(B, NB, b, d); tags = torch.full((B, NB, b), -1, dtype=torch.long); used = torch.zeros(B, NB, b, dtype=torch.bool)
        H1 = self.HF1[x].argmax(-1); H2 = self.HF2[x].argmax(-1)
        kgl_list = []; kg_prev = torch.zeros(B); lg = torch.empty(B, L, V)
        mod_oh = torch.zeros(B, p9.M_MOD); depth_oh = torch.zeros(B, p9.D_CLAMP + 1); mod_oh[:, 0] = 1; depth_oh[:, 0] = 1
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            if t > 0:
                pk = x[:, t - 1]; wm = kg_prev > 0.5; i1 = H1[:, t - 1]; i2 = H2[:, t - 1]
                T1 = tags[ar, i1]; T2 = tags[ar, i2]; U1 = used[ar, i1]; U2 = used[ar, i2]          # B,b
                f1 = (T1 == pk.unsqueeze(-1)) | (T1 == -1) | U1; f2 = (T2 == pk.unsqueeze(-1)) | (T2 == -1) | U2
                # priority: same-key cell anywhere, then free in bucket1, then free in bucket2, else cell0 of bucket1 (kick)
                same1 = (T1 == pk.unsqueeze(-1)); same2 = (T2 == pk.unsqueeze(-1))
                def first(mask): return mask.float().argmax(-1), mask.any(-1)
                c_s1, h_s1 = first(same1); c_s2, h_s2 = first(same2); c_f1, h_f1 = first(f1); c_f2, h_f2 = first(f2)
                use1 = h_s1 | (~h_s2 & h_f1); use2 = ~use1 & (h_s2 | h_f2)
                bk = torch.where(use1, i1, torch.where(use2, i2, i1))
                cell = torch.where(h_s1, c_s1, torch.where(use1, c_f1, torch.where(h_s2, c_s2, torch.where(use2, c_f2, torch.zeros_like(c_f1)))))
                kick = wm & ~use1 & ~use2
                if kick.any():
                    occ = tags[ar, i1, 0].clamp(min=0); a1 = self.HF1[occ].argmax(-1); a2 = self.HF2[occ].argmax(-1); alt = torch.where(a1 == i1, a2, a1)
                    Ta = tags[ar, alt]; fa = (Ta == -1) | used[ar, alt]; ca, ha = first(fa); mv = kick & ha
                    if mv.any():
                        v2 = vals.clone(); g2 = tags.clone(); u2 = used.clone()
                        v2[ar[mv], alt[mv], ca[mv]] = vals[ar[mv], i1[mv], 0]; g2[ar[mv], alt[mv], ca[mv]] = occ[mv]; u2[ar[mv], alt[mv], ca[mv]] = used[ar[mv], i1[mv], 0]
                        vals, tags, used = v2, g2, u2
                if wm.any():
                    v2 = vals.clone(); g2 = tags.clone(); u2 = used.clone()
                    v2[ar[wm], bk[wm], cell[wm]] = e[wm, t]; g2[ar[wm], bk[wm], cell[wm]] = pk[wm]; u2[ar[wm], bk[wm], cell[wm]] = False
                    vals, tags, used = v2, g2, u2
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            i1 = H1[:, t]; i2 = H2[:, t]; m1 = tags[ar, i1] == xid.unsqueeze(-1); m2 = tags[ar, i2] == xid.unsqueeze(-1)   # B,b
            seen_t = (m1.any(-1) | m2.any(-1)).float().unsqueeze(-1)
            kl = self.Wk(torch.cat([s, xt] + ([seen_t] if self.seen else []), -1)).squeeze(-1); kgl_list.append(kl); kg_prev = torch.sigmoid(kl)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1); R = a * R + torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            c1, h1 = m1.float().argmax(-1), m1.any(-1); c2, h2 = m2.float().argmax(-1), m2.any(-1)
            cand = torch.where(h1.unsqueeze(-1), vals[ar, i1, c1], vals[ar, i2, c2]); hit = (h1 | h2).float().unsqueeze(-1)
            read_on = hit
            R = (1 - read_on) * R + read_on * cand
            rd = hit.squeeze(-1) > 0.5
            if rd.any():
                rb = torch.where(h1, i1, i2); rc = torch.where(h1, c1, c2); u2 = used.clone(); u2[ar[rd], rb[rd], rc[rd]] = True; used = u2
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", default="R2"); ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--arm", default="BK2"); a = ap.parse_args()
    b = int(a.arm[2:]); t0 = time.time(); out = {"tag": "ARCH-VET-LM-P38", "protocol": __doc__[:1500], "seed": a.seed, "steps": a.steps, "arm": a.arm, "arms": {}}
    for rn in a.jobs.split(","):
        R = p32.regime(rn); rng = random.Random(12345); pool = [torch.tensor(p32.gen_stream(rng, R, 256)) for _ in range(512)]
        torch.manual_seed(a.seed); m = KRBBlocked(V, 24, R["slots"], b=b); print(f"[p38] {rn}:{a.arm} params {p19.n_params(m)}", flush=True)
        p37.train_arm(f"P38-{rn}-{a.arm}", m, pool, a.steps, 8, lam=0.5); m.eval()
        r = {f"n{n}_hard": p32.acc(m, R, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in R["n_eval"]}
        r["n_train_mix_hard"] = p32.acc(m, R, 12, 256, random.Random(700), True, None); r["params"] = p19.n_params(m); r["diag"] = p37.diag(m, R)
        print(f"[p38 {rn}:{a.arm} s{a.seed}] {r}", flush=True); out["arms"][f"{rn}:{a.arm}"] = r
        torch.save({"sd": m.state_dict()}, f"p21_ckpt/P38_{rn}_{a.arm}_s{a.seed}.pt")
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P38] DONE", flush=True)


if __name__ == "__main__":
    main()
