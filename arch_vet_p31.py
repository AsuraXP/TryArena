"""ARCH-VET P31 (cycle 66) — MULTI-QUERY ASSOCIATIVE RECALL (MQAR) capacity
probe + a NEW mechanism: the KEYED REGISTER BANK (KRB).

WHY: the certified VET substrate holds ONE soft register R (d dims). Its
pair task = 1 binding. Zoology (arXiv 2312.04927) shows recall capacity
is the axis where fixed-state models lose to attention; arXiv
2607.09889 (2026) frames it as "recall capped at ~state dimension" and
proposes a Dirichlet-process item cache between SSMs and attention;
arXiv 2507.00449 (joint recall) adds context-dependent keys. GAP: none
of these is an EXACT, token-addressed, O(1)-per-token write/read whose
addressing is discrete (no softmax over slots) and learned only in the
value path. KRB: n_slots slots, each slot owns a KEY id decided by a
hard, exact hash of the key token (slot = key_id mod n_slots: content-
addressed, no learning needed, collision-free when n_keys <= n_slots);
on a write (key followed by value) the value embedding is stored in
slot[key]; on a query (key after the answer marker) the register R is
REPLACED by slot[key] so the existing certified readout path (Wo, head)
answers. The controller decides write/read via the Mealy state (learned)
and the DCC-style exact integer machinery does the addressing. Cost
O(n_slots*d) state, O(d) per token — no attention, no growth with L.
TASK (MQAR-stream, V=48 alphabet reused): T T k1 v1 k2 v2 ... kn vn
<gap fill> A kq vq A kq' vq' ...  with n bindings in {1..4} at train,
queries in RANDOM order; eval n in {1,2,3,4,6,8} (6,8 = OOD count),
gaps 4-12 train / 24-48 hard. Keys 8 (KEYS..+4 and TRACK..+4 reused),
values 8. Answer acc = exact value at each query.
ARMS (matched ~21-24k params, 4000 steps, seeds 111/222):
  VETDCC-big (certified A, 1 register)   — the capacity-limited baseline
  VET-KRB   (VETDCC + KRB, 8 slots)      — the new mechanism
  TF-NoPE d48 (P23 arm, best in-range TF) — attention control
  TF-ALiBi d48                            — length-generalizing TF
Bars: n<=4 acc >=.9 at hard gap; n in {6,8} acc (capacity extrapolation);
CE ratio L1024/L256hard. Tag ARCH-VET-LM-P31.
"""
import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_lm as lm, arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p23 as p23
V = p19.V; BOS, EOS, A, T = p19.BOS, p19.EOS, p19.A, p19.T_TASK
KEYS8 = list(range(p19.KEYS, p19.KEYS + 4)) + list(range(p19.TRACK, p19.TRACK + 4))   # 8 keys
VALS8 = list(range(p19.VALS, p19.VALS + 4)) + list(range(p19.TRACK + 4, p19.TRACK + 8))  # 8 values
KEY_ID = {k: i for i, k in enumerate(KEYS8)}
fill = p19.fill_tok


def gen_mqar_stream(rng, L=256, hard=False, n_fixed=None):
    glo, ghi = (24, 48) if hard else (4, 12)
    x = [BOS]
    while len(x) < L:
        n = n_fixed or rng.randrange(1, 5)
        ks = rng.sample(KEYS8, n); vs = [rng.choice(VALS8) for _ in ks]
        gap = rng.randrange(glo, ghi + 1)
        order = list(range(n)); rng.shuffle(order)
        seg = [T, T] + [t for i in range(n) for t in (ks[i], vs[i])] + [fill(rng) for _ in range(gap)] + [t for i in order for t in (A, ks[i], vs[i])]
        if len(x) + len(seg) > L:
            x += [fill(rng)] * (L - len(x)); break
        x += seg
    x = x[:L]; x.append(EOS); return x


def mqar_pool(n, L, seed):
    rng = random.Random(seed); return [torch.tensor(gen_mqar_stream(rng, L)) for _ in range(n)]


@torch.no_grad()
def mqar_acc(model, n_streams, L, rng, hard, n_fixed):
    ok = tot = 0
    for _ in range(n_streams):
        x = gen_mqar_stream(rng, L, hard, n_fixed); xt = torch.tensor(x[:L + 1]).unsqueeze(0)
        pred = model(xt[:, :L]).argmax(-1).squeeze(0)
        for t in range(1, L - 1):
            if x[t] == A and x[t + 1] in KEY_ID:          # position t+1 (key) predicts value at t+2
                tot += 1; ok += int(int(pred[t + 1]) == x[t + 2])
    return round(ok / max(1, tot), 4), tot


class VETKRB(p9.VETDCC):
    """VETDCC + keyed register bank: exact content-addressed slots."""
    def __init__(self, V, d, k=8, K=8, n_slots=8):
        super().__init__(V, d, k=k, K=K)
        self.n_slots = n_slots
        key_map = torch.full((V,), -1, dtype=torch.long)
        for tok, i in KEY_ID.items(): key_map[tok] = i % n_slots
        self.register_buffer("key_map", key_map)
        self.Wread = nn.Linear(d + k, 1); nn.init.constant_(self.Wread.bias, -1.0)   # learned read gate (state-conditioned)

    def forward(self, x):
        B, L = x.shape; e = self.E(x); d = self.d
        R = torch.zeros(B, d); s = torch.full((B, self.k), 1.0 / self.k)
        buf = torch.zeros(B, self.K, d); valid = torch.zeros(B, self.K, dtype=torch.bool)
        c = torch.zeros(B, dtype=torch.long); dep = torch.zeros(B, dtype=torch.long)
        slots = torch.zeros(B, self.n_slots, d); pend = torch.full((B,), -1, dtype=torch.long)   # pending write slot
        after_A = torch.zeros(B, dtype=torch.bool)
        lg = torch.empty(B, L, V)
        for t in range(L):
            xt = e[:, t]; xid = x[:, t]
            is_one = xid == p9.ONE; is_task = xid == T
            c = torch.where(is_task, 0, torch.where(is_one, (c + 1) % p9.M_MOD, c))
            is_open = (xid == p9.BRK) | (xid == p9.BRK + 1); is_close = (xid == p9.BRK + 2) | (xid == p9.BRK + 3)
            dep = torch.where(is_task, 0, torch.where(is_open, (dep + 1).clamp(max=p9.D_CLAMP), dep)); dep = torch.where(is_close, (dep - 1).clamp(min=0), dep)
            mod_oh = F.one_hot(c, p9.M_MOD).float(); depth_oh = F.one_hot(dep, p9.D_CLAMP + 1).float()
            # ---- KRB exact addressing (integer, no gradient) ----
            kslot = self.key_map[xid]                                   # -1 if not a key
            is_key = kslot >= 0
            # write: previous token was a key in binding phase (not after A) -> store this value into pending slot
            wmask = (pend >= 0)
            if wmask.any():
                idx = pend.clamp(min=0)
                upd = slots.clone(); upd[torch.arange(B), idx] = torch.where(wmask.unsqueeze(-1), xt, slots[torch.arange(B), idx]); slots = upd
            pend = torch.where(is_key & ~after_A, kslot, torch.full_like(pend, -1))
            after_A = torch.where(xid == A, torch.ones_like(after_A), torch.where(is_key, after_A, torch.zeros_like(after_A)))
            # read: key right after A -> candidate = slots[key]
            s = F.softmax(self.Ws(torch.cat([xt, mod_oh, depth_oh], -1)) + self.Wss(s), -1)
            a = (s.unsqueeze(-1) * torch.exp(-F.softplus(self.Alog))).sum(1)
            w = torch.einsum("bk,ksd,bd->bd", s, self.Ww, xt)
            R = a * R + w
            rgate = torch.sigmoid(self.Wread(torch.cat([s, xt], -1)))            # (B,1) learned
            read_on = (is_key & after_A).float().unsqueeze(-1) * rgate
            cand = slots[torch.arange(B), kslot.clamp(min=0)]
            R = (1 - read_on) * R + read_on * cand                                # replace register on read
            g = torch.sigmoid(self.Wg(torch.cat([s, xt], -1))); push = (g > 0.5) + (g - g.detach())
            buf = torch.roll(buf, 1, dims=1); buf[:, 0] = xt * push
            valid = torch.roll(valid, 1, dims=1); valid[:, 0] = (g > 0.5).squeeze(-1)
            y = self.Wo(R + xt); feat = torch.stack([buf[:, j] for j in range(self.K)] + [xt], 1)
            logits = self.head(y) + torch.einsum("bk,bjd,kdv->bv", s, feat, self.M)
            sel = torch.zeros(B, self.K + 1)
            for j in range(self.K):
                newer = sum(valid[:, i].float() for i in range(j)) if j else torch.zeros(B)
                sel[:, j] = valid[:, j].float() * (newer == 0).float()
            sel[:, self.K] = 1.0
            logits = logits + torch.einsum("bs,ksv->bv", sel, self.T) + mod_oh @ self.W_mod + depth_oh @ self.W_depth
            lg[:, t] = logits
        return lg


def evaluate(m, name, seed):
    m.eval(); r = {}
    for hard in (False, True):
        for n in (1, 2, 3, 4, 6, 8):
            acc, tot = mqar_acc(m, 12, 256 if n <= 4 else 320, random.Random(600 + n), hard, n)
            r[f"n{n}_{'hard' if hard else 'train'}"] = acc
    rng = random.Random(777); vha = torch.stack([torch.tensor(gen_mqar_stream(rng, 256, True)) for _ in range(8)])
    rng = random.Random(31415); v1024 = torch.stack([torch.tensor(gen_mqar_stream(rng, 1024, False)) for _ in range(2)])
    r["ce_256hard"] = lm.val_ce(m, vha, 256); r["ce_1024"] = lm.val_ce(m, v1024, 1024); r["ratio"] = round(r["ce_1024"] / r["ce_256hard"], 3)
    print(f"[p31 {name} s{seed}] {r}", flush=True); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="VETDCC,VETKRB,TF-nope,TF-alibi"); ap.add_argument("--seeds", default="111,222"); ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args(); t0 = time.time(); out = {"tag": "ARCH-VET-LM-P31", "protocol": __doc__[:1800], "arms": {}}
    pool = mqar_pool(512, 256, 12345)
    for arm in a.arms.split(","):
        per = {}
        for seed in map(int, a.seeds.split(",")):
            torch.manual_seed(seed)
            m = {"VETDCC": lambda: p9.VETDCC(V, 24, k=8, K=8), "VETKRB": lambda: VETKRB(V, 24, k=8, K=8),
                 "TF-nope": lambda: p23.TFCtrl(V, pe="nope"), "TF-alibi": lambda: p23.TFCtrl(V, pe="alibi")}[arm]()
            npar = p19.n_params(m); print(f"[p31] {arm} s{seed} params {npar}", flush=True)
            hist = p19.train_arm(f"P31-{arm}-s{seed}", m, pool, a.steps, 8)
            r = evaluate(m, arm, seed); r["params"] = npar; r["hist"] = hist; per[str(seed)] = r
            torch.save({"sd": m.state_dict()}, f"p21_ckpt/P31_{arm}_s{seed}.pt")
        out["arms"][arm] = {"per_seed": per, "summary": {k: [per[s][k] for s in per] for k in per[list(per)[0]] if k not in ("hist",)}}
        print(f"[P31 {arm}] SUMMARY {json.dumps(out['arms'][arm]['summary'])}", flush=True)
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P31] DONE", flush=True)


if __name__ == "__main__":
    main()
