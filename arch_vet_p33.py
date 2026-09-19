"""ARCH-VET P33 (cycle 68) — TAGGED KRB with exact 2-choice (cuckoo-style)
placement: turns hash collisions into verified misses and removes the
P32 negative (soft two-table combine).

MECHANISM (all addressing exact integer, only the read gate is learned):
  each slot stores (value_emb d, key_tag int). Two hashes h1(k)=k mod S,
  h2(k)=(5k+3) mod S over ONE table of S slots (Pagh-Rodler cuckoo
  hashing, 2001; each key has exactly two candidate cells).
  WRITE(k,v): if cell h1 empty or tagged k -> write there; elif cell h2
  empty or tagged k -> write there; else overwrite h1 (bounded 1-kick:
  the displaced key's entry is moved to ITS alternate cell if that cell
  is free, else dropped). READ(k): probe h1 then h2; select the cell
  whose tag == k (exact match); if neither matches -> verified miss
  (register untouched). All deterministic; the controller only learns
  WHEN to read (read gate) and how to use the register (unchanged).
  Prior art: cuckoo hashing (Pagh & Rodler 2001, di.ens.fr Cuckoo.pdf);
  Raven arXiv 2607.25357 / TRIM-KV 2512.03324 use learned soft slot
  routing and eviction — no exact tag verification.
REGIMES (P32 definitions; token sets made DISJOINT from filler/ONE/MANS in this cycle — P32's overlapped, which depressed all P32 arms equally; oracle-verified ceilings: R1 1.0, R2 1.0/1.0/.994/.981/.927/.858, R3 1.0/1.0/.97/.90/.67/.50, R4 1.0/1.0/.956/.896/.741/.596 with task-boundary reset), 4000 steps (= P31 budget, removes the P32
confound), seed 111 (+222 on R2):
  R1 8k/8s, R2 16k/8s collision, R3 8k/4s overflow, R4 composite keys
  (pair hash then tagged with the pair id).
ARMS: KRB-TAG (this), KRB (P32 single-table, 4000 steps), TF-ALiBi
(4000 steps) on R2 as the strongest control; VETDCC skipped (banked).
PREDICTION: R2 KRB-TAG ~R1 level (collision -> alternate cell; 16 keys
in 8 slots at n<=4 live bindings is load .5 = cuckoo threshold, so
some kicks drop); R3 unchanged (physics); R4 ~R1.
Tag ARCH-VET-LM-P33.
"""
import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p23 as p23, arch_vet_p32 as p32
V = p19.V; A, T = p19.A, p19.T_TASK


class KRBTag(p9.VETDCC):
    def __init__(self, V, d, key_toks, slots, comp=False, k=8, K=8):
        super().__init__(V, d, k=k, K=K); self.S, self.comp = slots, comp
        kid = torch.full((V,), -1, dtype=torch.long)
        flat = sorted({t for kk in key_toks for t in (kk if isinstance(kk, tuple) else (kk,))})
        for i, t in enumerate(flat): kid[t] = i
        self.register_buffer("kid", kid); self.nk = len(flat)
        self.Wread = nn.Linear(d + k, 1); nn.init.constant_(self.Wread.bias, -1.0)

    def key_id(self, xid, prev):
        i = self.kid[xid]
        if not self.comp: return i
        j = self.kid[prev]; return torch.where((i >= 0) & (j >= 0), j * self.nk + i, torch.full_like(i, -1))

    def forward(self, x):
        B, L = x.shape; e = self.E(x); d = self.d; ar = torch.arange(B); S = self.S
        R = torch.zeros(B, d); s = torch.full((B, self.k), 1.0 / self.k)
        buf = torch.zeros(B, self.K, d); valid = torch.zeros(B, self.K, dtype=torch.bool)
        c = torch.zeros(B, dtype=torch.long); dep = torch.zeros(B, dtype=torch.long)
        vals = torch.zeros(B, S, d); tags = torch.full((B, S), -1, dtype=torch.long)
        pend = torch.full((B,), -1, dtype=torch.long); after_A = torch.zeros(B, dtype=torch.bool); prev = torch.zeros(B, dtype=torch.long)
        lg = torch.empty(B, L, V)
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            is_one = xid == p9.ONE; is_task = xid == T
            c = torch.where(is_task, 0, torch.where(is_one, (c + 1) % p9.M_MOD, c))
            is_open = (xid == p9.BRK) | (xid == p9.BRK + 1); is_close = (xid == p9.BRK + 2) | (xid == p9.BRK + 3)
            dep = torch.where(is_task, 0, torch.where(is_open, (dep + 1).clamp(max=p9.D_CLAMP), dep)); dep = torch.where(is_close, (dep - 1).clamp(min=0), dep)
            mod_oh = F.one_hot(c, p9.M_MOD).float(); depth_oh = F.one_hot(dep, p9.D_CLAMP + 1).float()
            # ---- per-task RESET of the bank (same convention as the DCC counters: T_TASK clears state)
            if is_task.any():
                tags = torch.where(is_task.unsqueeze(-1), torch.full_like(tags, -1), tags)
                vals = torch.where(is_task.view(B, 1, 1), torch.zeros_like(vals), vals)
            # ---- pending WRITE of value (this token) for key `pend` (exact cuckoo placement, 1 kick)
            wm = pend >= 0
            if wm.any():
                kk = pend.clamp(min=0); h1 = kk % S; h2 = (5 * kk + 3) % S
                t1 = tags[ar, h1]; t2 = tags[ar, h2]
                use1 = (t1 == -1) | (t1 == kk); use2 = ~use1 & ((t2 == -1) | (t2 == kk))
                cell = torch.where(use1, h1, torch.where(use2, h2, h1))
                # 1-kick: displaced occupant of h1 (when neither free) moves to its alt cell if free
                kick = ~use1 & ~use2
                if kick.any():
                    occ = t1.clamp(min=0); alt = torch.where((occ % S) == h1, (5 * occ + 3) % S, occ % S)
                    free_alt = tags[ar, alt] == -1; mv = kick & free_alt
                    if mv.any():
                        v2 = vals.clone(); g2 = tags.clone()
                        v2[ar[mv], alt[mv]] = vals[ar[mv], h1[mv]]; g2[ar[mv], alt[mv]] = occ[mv]; vals, tags = v2, g2
                v2 = vals.clone(); g2 = tags.clone()
                v2[ar[wm], cell[wm]] = xt[wm]; g2[ar[wm], cell[wm]] = kk[wm]; vals, tags = v2, g2
            kidt = self.key_id(xid, prev); is_key = kidt >= 0
            pend = torch.where(is_key & ~after_A, kidt, torch.full_like(pend, -1))
            after_A = torch.where(xid == A, torch.ones_like(after_A), torch.where(is_key | (self.kid[xid] >= 0), after_A, torch.zeros_like(after_A)))
            # ---- controller + register
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1); R = a * R + torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            # ---- verified READ
            kk = kidt.clamp(min=0); h1 = kk % S; h2 = (5 * kk + 3) % S
            m1 = tags[ar, h1] == kk; m2 = tags[ar, h2] == kk
            cand = torch.where(m1.unsqueeze(-1), vals[ar, h1], vals[ar, h2]); hit = (m1 | m2) & is_key & after_A
            rg = torch.sigmoid(self.Wread(torch.cat([s, xt], -1))); read_on = hit.float().unsqueeze(-1) * rg
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
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", default="R1:KRB-TAG,R2:KRB-TAG,R2:KRB,R2:TF-alibi,R3:KRB-TAG,R4:KRB-TAG")
    ap.add_argument("--seed", type=int, default=111); ap.add_argument("--steps", type=int, default=4000); a = ap.parse_args()
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P33", "protocol": __doc__[:1900], "seed": a.seed, "steps": a.steps, "arms": {}}
    pools = {}
    for job in a.jobs.split(","):
        rn, arm = job.split(":"); R = p32.regime(rn)
        if rn not in pools: rng = random.Random(12345); pools[rn] = [torch.tensor(p32.gen_stream(rng, R, 256)) for _ in range(512)]
        torch.manual_seed(a.seed)
        m = {"KRB-TAG": lambda: KRBTag(V, 24, R["keys"], R["slots"], comp=R["comp"]),
             "KRB": lambda: p32.KRB(V, 24, R["keys"], R["slots"], "pair" if R["comp"] else "one"),
             "TF-alibi": lambda: p23.TFCtrl(V, pe="alibi")}[arm]()
        print(f"[p33] {job} params {p19.n_params(m)}", flush=True)
        p19.train_arm(f"P33-{rn}-{arm}", m, pools[rn], a.steps, 8); m.eval()
        r = {f"n{n}_hard": p32.acc(m, R, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in R["n_eval"]}
        r["n_train_mix_hard"] = p32.acc(m, R, 12, 256, random.Random(700), True, None); r["params"] = p19.n_params(m)
        print(f"[p33 {job}] {r}", flush=True); out["arms"][job] = r
        torch.save({"sd": m.state_dict()}, f"p21_ckpt/P33_{rn}_{arm}_s{a.seed}.pt")
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P33] DONE", flush=True)


if __name__ == "__main__":
    main()
