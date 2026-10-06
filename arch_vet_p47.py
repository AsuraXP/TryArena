#!/usr/bin/env python3
"""C95 / P47 — CORE LANE: SLOT-BINDING CORE (SBC).
State = M slots (key, value, occupancy) that the learner itself writes.  Every
component is learned (controller GRU, q/k/v projections, write gate, match
threshold); the slots are inductive bias, not an organ — delete the learned
part and nothing works.
Per token t (controller state h_t):
  read : a = softmax((q.K_j/sqrt(d) + log occ_j)/tau)   r = sum a_j V_j
         out = head([h_t, r, conf]) with conf = total occupied attention
  write: g = sigmoid(w_g.h_t) (sparse: L1 penalty)   m = softmax((k.K_j/sqrt(d) + log occ_j)/tau)
         p = sigmoid(max_j k.K_j/sqrt(d) - theta)   (match exists?)
         addr = p*m + (1-p)*softmax(-occ/tau)       (overwrite match, else emptiest slot)
         alpha_j = g*addr_j ; K_j,V_j <- (1-alpha_j)(K_j,V_j) + alpha_j (k,v) ; occ_j <- occ_j + alpha_j(1-occ_j)
tau anneals 1.0 -> 0.1 during training; eval tau 0.05.  No positional signal.
Prior art (searched 2026-10-06): Sparse Delta Memory (Cabannes+ 2026-07, writes
every token to top-k slots, soft superposition), Lattice arXiv:2504.05646,
NTM/DNC (location/usage addressing).  Delta: write as learned sparse EVENT,
annealed-hard one-binding-per-slot with overwrite, readable emptiness.
Task/eval identical to arch_vet_p46.py (same eval seeds -> directly comparable
to the banked TF/GRU/INV table in log.md C94).
"""
import sys, json, random, time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
import arch_vet_p46 as T

V, PAD = T.V, T.PAD
torch.set_num_threads(2)


TIE, ST, GATE_INIT, GMIN0 = False, False, 2.0, 0.2


def hard(p):
    """forward one-hot argmax, backward soft (straight-through)."""
    if not ST: return p
    oh = F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype); return oh + p - p.detach()


class SBC(nn.Module):
    def __init__(self, d, M=8):
        super().__init__(); self.d, self.M = d, M
        self.E = nn.Embedding(V, d); self.ctrl = nn.GRUCell(d, d)
        # q,k in the SHARED embedding space with learned per-tap mixing (alignment = 4d params, not two d^2 maps)
        self.ka = nn.Parameter(torch.full((2, d), 0.5)); self.qa = nn.Parameter(torch.full((2, d), 0.5))
        self.v = nn.Linear(2 * d, d)
        self.K0 = nn.Parameter(torch.randn(16, d) * 0.5)   # distinct learned initial slot keys: breaks slot symmetry (up to 16 slots)
        self.g = nn.Linear(d, 1); self.theta = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Linear(2 * d + 1, V)
        nn.init.constant_(self.g.bias, GATE_INIT)   # +2: start writing everything (like a KV cache); learn to NOT write

    def forward(self, x, tau=0.1, states=False, force_write=False, M=None, gmin=0.0):
        B, L = x.shape; d = self.d; M = M or self.M; e = self.E(x)
        h = torch.zeros(B, d); K = self.K0[:M].unsqueeze(0).expand(B, M, d).clone(); Vm = torch.zeros(B, M, d); occ = torch.zeros(B, M)
        outs, gates = [], []
        prev = torch.zeros(B, d)
        for t in range(L):
            h = self.ctrl(e[:, t], h); win = torch.cat([prev, e[:, t]], -1); prev0 = prev; prev = e[:, t]
            logocc = torch.log(occ + 1e-4)
            # read
            q = self.qa[0] * e[:, t] + self.qa[1] * prev0; sim = torch.einsum("bd,bmd->bm", q, K) / math.sqrt(d)
            a = hard(torch.softmax((sim + logocc) / tau, -1)); r = torch.einsum("bm,bmd->bd", a, Vm)
            conf = (a * occ).sum(-1, keepdim=True)
            outs.append(self.head(torch.cat([h, r, conf], -1)))
            # write
            k = self.ka[0] * e[:, t] + self.ka[1] * prev0; v = self.v(win); g = torch.ones(B, 1) if force_write else gmin + (1 - gmin) * torch.sigmoid(self.g(h)); gates.append(g)   # leaky: 'off' is never absorbing
            wsim = torch.einsum("bd,bmd->bm", k, K) / math.sqrt(d)
            m = hard(torch.softmax((wsim + logocc) / tau, -1))
            p = torch.sigmoid(wsim.max(-1, keepdim=True).values - self.theta)
            empty = hard(torch.softmax(-occ / tau, -1))
            addr = p * m + (1 - p) * empty; alpha = (g * addr).unsqueeze(-1)
            K = (1 - alpha) * K + alpha * k.unsqueeze(1); Vm = (1 - alpha) * Vm + alpha * v.unsqueeze(1)
            occ = occ + alpha.squeeze(-1) * (1 - occ)
        logits = torch.stack(outs, 1); gates = torch.cat(gates, 1)
        return (logits, gates, occ) if states else logits


def train(d, M, seed, steps, B, log, l1=0.0, warm=0):
    torch.manual_seed(seed); rng = random.Random(seed)
    m = SBC(d, M); opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time()
    for s in range(steps):
        tau = 1.0 * (0.1 / 1.0) ** (s / max(steps - 1, 1))
        bt = T.make_batch(rng, B, (2, 4), 2, 4, with_pairs=False)
        gmin = GMIN0 * (1 - s / max(steps - 1, 1))
        logits, gates, occ = m(bt["x"], tau=tau, states=True, force_write=(s < warm), gmin=gmin)
        ce = T.ce_loss(logits, bt["x"], bt["ans"]); mask = (bt["x"] != PAD).float()
        gl = (gates * mask).sum() / mask.sum(); loss = ce + l1 * gl
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 500 == 0 or s == steps - 1:
            log(f"  [SBC d{d} M{M} s{seed}] step {s} tau {tau:.2f} ce {ce.item():.3f} gate {gl.item():.3f} occ {occ.mean().item():.2f} ({time.time()-t0:.0f}s)")
    return m


@torch.no_grad()
def acc(m, batches, tau=0.05, M=None):
    m.eval(); c = t = 0
    for bt in batches:
        p = m(bt["x"], tau=tau, M=M).argmax(-1)
        for i, poss in enumerate(bt["ans"]):
            for s in poss: c += int(p[i, s - 1].item() == bt["x"][i, s].item()); t += 1
    m.train(); return c / max(t, 1)


@torch.no_grad()
def write_profile(m, batches, tau=0.05):
    """fraction of write-gate mass spent on fact-value tokens vs filler (is the write an event?)"""
    m.eval(); fact = fil = 0.0; nf = nl = 0
    for bt in batches:
        _, g, _ = m(bt["x"], tau=tau, states=True); x = bt["x"]
        isval = (x >= T.VAL0) & (x < T.FIL0); isfil = x >= T.FIL0
        fact += g[isval].sum().item(); nf += isval.sum().item(); fil += g[isfil].sum().item(); nl += isfil.sum().item()
    m.train(); return round(fact / max(nf, 1), 3), round(fil / max(nl, 1), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", default="16,32,64"); ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--seeds", default="1"); ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--B", type=int, default=16); ap.add_argument("--warm", type=int, default=0); ap.add_argument("--Meval", type=int, default=0)
    a = ap.parse_args(); log = lambda *x: print(*x, flush=True)
    er = random.Random(999); ev_in = [T.make_batch(er, 16, (2, 4), 2, 4, False) for _ in range(12)]
    er = random.Random(998); ev_len = [T.make_batch(er, 16, (2, 4), 2, 30, False) for _ in range(12)]
    er = random.Random(997); ev_cnt = [T.make_batch(er, 16, (8, 8), 2, 4, False) for _ in range(12)]
    er = random.Random(996); ev_both = [T.make_batch(er, 16, (8, 8), 2, 30, False) for _ in range(12)]
    er = random.Random(995); ev_far = [T.make_batch(er, 16, (8, 8), 2, 100, False) for _ in range(6)]
    log(f"[P47] eval lens in={ev_in[0]['x'].shape[1]} len={ev_len[0]['x'].shape[1]} cnt={ev_cnt[0]['x'].shape[1]} both={ev_both[0]['x'].shape[1]} far={ev_far[0]['x'].shape[1]}")
    res = {"tag": "ARCH-VET-LM-P47", "protocol": __doc__[:1500], "runs": []}
    for d in map(int, a.ds.split(",")):
        for seed in map(int, a.seeds.split(",")):
            m = train(d, a.M, seed, a.steps, a.B, log, warm=a.warm)
            wp = write_profile(m, ev_in)
            Me = a.Meval or None
            r = {"arm": "SBC", "warm": a.warm, "d": d, "M": a.M, "Meval": a.Meval or a.M, "seed": seed, "params": T.n_params(m), "in": acc(m, ev_in, M=Me), "len": acc(m, ev_len, M=Me), "cnt": acc(m, ev_cnt, M=Me), "both": acc(m, ev_both, M=Me), "far": acc(m, ev_far, M=Me), "gate_on_values": wp[0], "gate_on_filler": wp[1]}
            res["runs"].append(r); log(f"[p47 SBC d{d} M{a.M} s{seed}] params {r['params']} in {r['in']:.3f} len {r['len']:.3f} cnt {r['cnt']:.3f} both {r['both']:.3f} far {r['far']:.3f} gate val/fil {wp}")
    xs = [math.log2(r["params"]) for r in res["runs"]]
    if len(xs) >= 2:
        res["slopes"] = {}
        for met in ["in", "len", "cnt", "both", "far"]:
            ys = [r[met] for r in res["runs"]]; mx = sum(xs) / len(xs); my = sum(ys) / len(ys)
            res["slopes"][met] = round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9), 4)
        log("[P47] slopes " + json.dumps(res["slopes"]))
    with open("log.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
    log("[P47] DONE")


if __name__ == "__main__":
    main()
