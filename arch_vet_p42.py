"""ARCH-VET P42 (cycle 83) — LEARNED PREDICATES for expert A (VETDCC-LP).
Claim-integrity gap: expert A's exact organs (mod-3 counter, clamped depth counter) are driven by four HAND-WIRED
token predicates: is_one (count token), is_task (reset), is_open, is_close. Expert K's predicate was already made
learnable (P35/P37, hindsight). Here the four predicates become a learned V x 4 logit table P, with NO supervision
other than next-token CE:
  SOFT mode (training): the counters are run as DISTRIBUTIONS over their states (mod in Delta^3, depth in Delta^7),
    shifted by the predicate probabilities p = sigmoid(P[x_t] / tau) — reset mixes toward e0, count rolls the mod
    distribution, open/close shift depth up/down with mass clamped at the ends. The controller and the zero-init
    readouts (W_mod, W_depth) consume the soft one-hots. tau anneals 1 -> 0.2 (DeepDFA-style) so the soft automaton
    approaches the discrete one as training proceeds.
  HARD mode (deployment / all evals): p > 0.5 -> the ORIGINAL exact integer counters. Length invariance is therefore
    inherited exactly from VETDCC; only the token->predicate map was learned.
Prior art (header citations): Differentiable FSMs (Google Research self-organising-systems blog, 2022: softmax train,
hardmax decode); DeepDFA (arXiv 2408.08622, 2024: temperature-annealed probabilistic-automaton relaxation);
Grefenstette et al. 2015 (neural stacks with learned push/pop); Suzgun et al. 2019 (memory-augmented RNNs on
generalized Dyck). Difference: those learn whole transition tables from labelled traces / stack actions; here the
arithmetic organs are fixed and EXACT, only the 4 x V predicate map is learned, under the LM objective, inside the
certified VET substrate, and deployment is the exact integer machine.
Arms: LP (learned predicates, soft->hard) vs HW (hand-wired VETDCC, == certified A). Same pool, 4000 st, seeds.
Report: eval_full bars (pair/modk/ratio) in HARD mode + the recovered predicate sets vs truth."""
import argparse, json, os, random, time, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p13d as p13d
V = p19.V; ONE, BRK, T_TASK = p9.ONE, p9.BRK, p9.T_TASK; M_MOD, D_CLAMP = p9.M_MOD, p9.D_CLAMP
VETDCC = p13d.VETDCC


class VETDCC_LP(VETDCC):
    def __init__(self, V, d, k=8, K=8, p0=-2.0, live=False, st=False):
        super().__init__(V, d, k=k, K=K)
        self.P = nn.Parameter(torch.zeros(V, 4) + p0)     # predicate logits; 4 x V = 192 params
        self.tau = 1.0; self.hard = False; self.st = st
        if live:   # C84 FIX: P13d zero-inits every consumer of the counter state (Ws counter block, W_mod, W_depth) so dL/dP == 0
                   # exactly at init (measured: grad-norm 0.0). Give the consumers a small random init so P receives gradient.
            with torch.no_grad():
                nc = M_MOD + D_CLAMP + 1; self.Ws.weight[:, -nc:].normal_(0, 0.1); self.W_mod.normal_(0, 0.1); self.W_depth.normal_(0, 0.1)

    def truth_table(self):
        t = torch.zeros(V, 4, dtype=torch.bool); t[ONE, 0] = True; t[T_TASK, 1] = True
        t[BRK, 2] = t[BRK + 1, 2] = True; t[BRK + 2, 3] = t[BRK + 3, 3] = True; return t

    def forward(self, x):
        B, L = x.shape; e = self.E(x); dev = x.device
        R = torch.zeros(B, self.d, device=dev); s = torch.full((B, self.k), 1.0 / self.k, device=dev)
        buf = torch.zeros(B, self.K, self.d, device=dev); valid = torch.zeros(B, self.K, dtype=torch.bool, device=dev)
        lg = torch.empty(B, L, V, device=dev)
        if self.hard:
            Pb = self.P > 0; c = torch.zeros(B, dtype=torch.long, device=dev); dep = torch.zeros(B, dtype=torch.long, device=dev)
        else:
            Pp = torch.sigmoid(self.P / self.tau)
            if self.st: Pp = (self.P > 0).float() + (Pp - Pp.detach())   # straight-through: hard forward, sigmoid gradient
            m = torch.zeros(B, M_MOD, device=dev); m[:, 0] = 1; dv = torch.zeros(B, D_CLAMP + 1, device=dev); dv[:, 0] = 1
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            if self.hard:
                pr = Pb[xid]; is_one, is_task, is_open, is_close = pr[:, 0], pr[:, 1], pr[:, 2], pr[:, 3]
                c = torch.where(is_task, 0, torch.where(is_one, (c + 1) % M_MOD, c))
                dep = torch.where(is_task, 0, torch.where(is_open, (dep + 1).clamp(max=D_CLAMP), dep))
                dep = torch.where(is_close, (dep - 1).clamp(min=0), dep)
                mod_oh = F.one_hot(c, M_MOD).float(); depth_oh = F.one_hot(dep, D_CLAMP + 1).float()
            else:
                pr = Pp[xid]; p1, pt, po, pc = pr[:, 0:1], pr[:, 1:2], pr[:, 2:3], pr[:, 3:4]
                e0m = torch.zeros_like(m); e0m[:, 0] = 1; e0d = torch.zeros_like(dv); e0d[:, 0] = 1
                m = (1 - pt) * ((1 - p1) * m + p1 * torch.roll(m, 1, dims=1)) + pt * e0m
                up = torch.cat([torch.zeros_like(dv[:, :1]), dv[:, :-1]], 1); up[:, -1] += dv[:, -1]          # shift up, clamp mass at D
                d2 = (1 - pt) * ((1 - po) * dv + po * up) + pt * e0d
                dn = torch.cat([d2[:, 1:], torch.zeros_like(d2[:, :1])], 1); dn[:, 0] += d2[:, 0]              # shift down, clamp at 0
                dv = (1 - pc) * d2 + pc * dn
                mod_oh, depth_oh = m, dv
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1)
            R = a * R + torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            g = torch.sigmoid(self.Wg(torch.cat([s, xt], -1))); push = (g > 0.5) + (g - g.detach())
            buf = torch.roll(buf, 1, dims=1); buf[:, 0] = xt * push; valid = torch.roll(valid, 1, dims=1); valid[:, 0] = (g > 0.5).squeeze(-1)
            y = self.Wo(R + xt); feat = torch.stack([buf[:, j] for j in range(self.K)] + [xt], 1)
            logits = self.head(y) + torch.einsum("bk,bjd,kdv->bv", s, feat, self.M)
            sel = torch.zeros(B, self.K + 1, device=dev)
            for j in range(self.K):
                newer = sum(valid[:, i].float() for i in range(j)) if j else torch.zeros(B, device=dev); sel[:, j] = valid[:, j].float() * (newer == 0).float()
            sel[:, self.K] = 1.0
            logits = logits + torch.einsum("bs,ksv->bv", sel, self.T) + mod_oh @ self.W_mod + depth_oh @ self.W_depth
            lg[:, t] = logits
        return lg


def train_lp(name, m, pool, steps, batch=8, lr=3e-3, tau0=1.0, tau1=0.2, l1=0.0):
    """p19.train_arm recipe (AdamW 3e-3, step-indexed batches, clip 1) + tau annealing for the predicate relaxation."""
    torch.manual_seed(0); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr); m.train(); t0 = time.time(); n_pool = len(pool)
    for step in range(1, steps + 1):
        m.tau = tau0 * (tau1 / tau0) ** ((step - 1) / max(1, steps - 1)); m.hard = False
        sel = [(step * batch + i) % n_pool for i in range(batch)]; x = torch.stack([pool[i] for i in sel])
        lg = m(x[:, :256]); loss = F.cross_entropy(lg.reshape(-1, V), x[:, 1:257].reshape(-1))
        if l1 > 0: loss = loss + l1 * torch.sigmoid(m.P).sum()   # sparsity prior: predicates are few-token sets
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            with torch.no_grad():
                tt = m.truth_table(); Pb = m.P > 0; rec = [(int((Pb[:, j] & tt[:, j]).sum()), int(Pb[:, j].sum()), int(tt[:, j].sum())) for j in range(4)]
            print(f"  [{name}] step {step}/{steps} ce {loss.item():.4f} tau {m.tau:.2f} pred(hit/on/true) one{rec[0]} task{rec[1]} open{rec[2]} close{rec[3]} ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="LP"); ap.add_argument("--seed", type=int, default=111); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--p0", type=float, default=-2.0); ap.add_argument("--live", action="store_true"); ap.add_argument("--st", action="store_true")
    ap.add_argument("--tau1", type=float, default=0.2); ap.add_argument("--l1", type=float, default=0.0); ap.add_argument("--tagsuffix", default="")
    ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--freeze_ctrl", default="", help="C85 diagnostic: load an HW checkpoint, freeze everything but P, learn only the predicates")
    a = ap.parse_args()
    pool = p19.make_pool(512, 256, 12345); out = {"tag": "ARCH-VET-LM-P42" + a.tagsuffix, "protocol": __doc__[:1800], "seed": a.seed, "steps": a.steps, "cfg": vars(a), "arms": {}}
    for arm in a.arms.split(","):
        torch.manual_seed(a.seed)
        if arm == "LP":
            m = VETDCC_LP(V, 24, k=8, K=8, p0=a.p0, live=a.live, st=a.st)
            if a.freeze_ctrl:
                sd = torch.load(a.freeze_ctrl)["sd"]; missing = m.load_state_dict(sd, strict=False); assert missing.missing_keys == ["P"], missing
                for n_, p_ in m.named_parameters(): p_.requires_grad_(n_ == "P")
                print(f"[p42{a.tagsuffix}] controller frozen from {a.freeze_ctrl}; trainable = P only", flush=True)
            print(f"[p42{a.tagsuffix}] LP params {p19.n_params(m)} cfg p0={a.p0} live={a.live} st={a.st} tau1={a.tau1} l1={a.l1}", flush=True); train_lp(f"P42{a.tagsuffix}-LP-s{a.seed}", m, pool, a.steps, lr=a.lr, tau1=a.tau1, l1=a.l1); m.hard = True; m.eval()
            with torch.no_grad():
                tt = m.truth_table(); Pb = m.P > 0
                pred = {n: {"learned_on": sorted(torch.nonzero(Pb[:, j]).flatten().tolist()), "truth": sorted(torch.nonzero(tt[:, j]).flatten().tolist())} for j, n in enumerate(["one", "task", "open", "close"])}
                exact = bool((Pb == tt).all())
        else:
            m = VETDCC(V, 24, k=8, K=8); print(f"[p42] HW params {p19.n_params(m)}", flush=True); p19.train_arm(f"P42-HW-s{a.seed}", m, pool, a.steps, 8); m.eval(); pred = None; exact = None
        r = p19.eval_full(m, f"P42-{arm}", a.seed); r["predicates"] = pred; r["predicates_exact"] = exact; r["params"] = p19.n_params(m)
        acc = r["acc_eval_interval"]; r["bars"] = {"pair": acc["pair"] >= .717, "modk": acc["modk"] >= 1, "ratio": r["len_ratio_1024_over_256hard"] <= .6}
        print(f"[p42 {arm} s{a.seed}] bars={sum(r['bars'].values())}/3 pair {acc['pair']} modk {acc['modk']} ratio {r['len_ratio_1024_over_256hard']} predicates_exact={exact} {pred}", flush=True)
        out["arms"][arm] = r; torch.save({"sd": m.state_dict()}, f"p21_ckpt/P42{a.tagsuffix}_{arm}_s{a.seed}.pt")
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P42] DONE", flush=True)


if __name__ == "__main__":
    main()
