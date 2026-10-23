"""C127 — LIN: a learner whose KEY FUNCTION is re-executable by the reader (REPRESENTATION axis; built from scratch, not p49).
Question it starts from: what would a learner look like in which the second hop is an OPERATION on the representation, not a
second read that must guess a context-laden key?  Answer tried here: make the writer's key a LINEAR, decaying bound sum of the
unit's symbol embeddings, k_t = lam*k_{t-1} + P e_t.  Then the key the writer used for prefix "b0 b1 =" is an algebraic
function of (b0, b1, =) that the reader can RE-EXECUTE on recalled symbols: q = lam^2 P r(b0) + lam P r(b1) + P s, where s is
a learned shape vector.  Memory is a fast-weight matrix W += k_{t-1} (x) e_t (outer product, written every step).
Reads are dense linear (r = W k), so every hop passes gradient from step 0 — the starvation loop of a gated chain (C116–C126)
cannot form.  The slow GRU is only a controller: it mixes hop outputs and feeds the head; it never makes keys (--keyfn lin).
CONTROL for the stated reason: --keyfn ctrl makes the key from the controller state (k = P h), as in Fast Weight Memory
(Schlag, Munkhdalai, Schmidhuber, ICLR 2021, arXiv:2011.07831 — retrieved values become keys of later queries with LSTM-made
TPR keys; walked, succeeded at ~1M steps on concatenated-story tasks).  If LIN composes and CTRL does not, the reason
"re-executable key function" is tested, not asserted.  Prior art also: Ba et al. 2016 fast weights (arXiv:1610.06258),
Schlag et al. 2021 linear transformers as fast weight programmers (arXiv:2102.11174), FREKI (github thevoidwolf) hop-with-
cleanup.  None fixes the key to a controller-free linear recurrence so that re-posing is exact algebra; that is the delta.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class LIN(nn.Module):
    def __init__(self, V, d, keyfn="lin", hops=2, kmult=2, roles=0, hopce=0.0, sparse=0.0, kcons=0.0, nodecay=0, wgate=0.0, delta=0, pgru=0, clean=0):
        super().__init__()
        dk = d * kmult; self.d, self.dk, self.keyfn, self.hops = d, dk, keyfn, hops
        self.sparse = sparse; self.kcons = kcons; self.nodecay = nodecay; self.wgate = wgate; self.delta = delta
        if wgate: self.wg = nn.Linear(2 * d, 1)   # C136: the learner decides what to store (write gate from token+controller), L1 pressure wgate
        self.E = nn.Embedding(V, d); self.P = nn.Linear(d, dk, bias=False);
        if sparse:   # C131: Kanerva-style address — FIXED random projection (no parameters) + k-WTA; crosstalk falls with dk at zero parameter cost
            self.P.weight.requires_grad_(False); nn.init.normal_(self.P.weight, std=1.0 / math.sqrt(d))
        self.Ph = nn.Linear(d, dk, bias=False) if keyfn == 'ctrl' else None
        self.lam = nn.Parameter(torch.full((dk,), 1.0)); self.dec = nn.Parameter(torch.tensor(4.0))   # W decay sigmoid(4)=.98   # sigmoid -> ~.73 start
        self.shape = nn.Parameter(torch.randn(d) * 0.1); self.roles = roles
        if roles: self.R = nn.Parameter(torch.randn(roles, dk) * 0.5 + 1.0)   # C128: learned role per offset (TPR with recency roles)
        self.clean = clean   # M2: power-iteration cleanup of each read on the fast weights themselves: q <- norm(W^T (W q)) repeated `clean` times
        self.pgru = pgru   # speed form: controller GRU reads the three raw reads of t-1 (no feedback from the mix) -> fused nn.GRU, no python loop
        self.g = nn.GRU(4 * d, d, batch_first=True) if pgru else nn.GRUCell(2 * d, d)
        self.mix = nn.Linear(d, 3)   # r0 (1-hop), r1 (continuation), rq (re-posed)
        self.head = nn.Linear(2 * d, V); self.hopce = hopce; self.head2 = nn.Linear(2 * d, V)   # C129: t+2 head for the continuation read

    def key(self, k, e, h):
        lam = torch.sigmoid(self.lam)
        if self.keyfn == "lin": return lam * k + self.P(e)
        return self.Ph(h)   # controller-made key (FWM-style control)

    def rolekey(self, win):   # win: list of last `roles` projected tokens, most recent last
        k = sum(self.R[i] * win[-1 - i] for i in range(min(self.roles, len(win))))
        if self.sparse:   # k-WTA on the bound key, straight-through
            kk = max(1, int(self.sparse * k.shape[-1])); th = k.abs().topk(kk, dim=-1).values[:, -1:]; mask = (k.abs() >= th).float()
            k = k * mask + (k - k.detach()) * (1 - mask) * 0
        return k

    def forward_loop(self, x, **kw):
        B, L = x.shape; d, dk = self.d, self.dk; e = self.E(x)
        h = torch.zeros(B, d); k = torch.zeros(B, dk); W = torch.zeros(B, dk, d); r = torch.zeros(B, d); outs = []; self._mix = []; self._reads = []; self._hs = []; self._kw = []; self._qq = []; self._pr = []; win = []; l0, l1 = [], []; pr0, pr1 = [], []; gs = []
        lam = torch.sigmoid(self.lam); sP = self.P(self.shape).unsqueeze(0)
        for t in range(L):
            et = e[:, t]
            # write: previous key -> this token (normalised key so W stays bounded)
            kn = F.normalize(k, dim=-1); self._kw.append(kn.detach())   # heard key used for the write at step t
            vt = et - torch.einsum("bk,bkd->bd", kn, W) if self.delta else et   # C136b control: delta-rule (error-correcting) write, DeltaNet-style
            if self.wgate:
                g = torch.sigmoid(self.wg(torch.cat([et, h], -1))); gs.append(g); vt = vt * g
            W = (W if self.nodecay else torch.sigmoid(self.dec) * W) + kn.unsqueeze(2) * vt.unsqueeze(1)
            # advance key and controller
            if self.roles:
                win.append(self.P(et)); win = win[-self.roles:]; k = self.rolekey(win)
            else: k = self.key(k, et, h)
            h = self.g(torch.cat([et, r], -1), h)
            def rd(q):
                r = torch.einsum("bk,bkd->bd", q, W)
                for _ in range(self.clean): q = F.normalize(torch.einsum("bd,bkd->bk", r, W), dim=-1); r = torch.einsum("bk,bkd->bd", q, W)
                return r
            q0 = F.normalize(k, dim=-1); r0 = rd(q0)           # 1-hop: what followed this prefix
            if self.roles: k1 = self.rolekey(win + [self.P(r0)])
            else: k1 = self.key(k, r0, h)
            q1 = F.normalize(k1, dim=-1); r1 = rd(q1)   # continuation (next symbol)
            if self.roles: qq = self.rolekey([self.P(r0), self.P(r1), sP.expand(B, -1)])   # RE-POSE with the same roles
            elif self.keyfn == "lin": qq = lam * lam * self.P(r0) + lam * self.P(r1) + sP   # RE-POSE: writer's key function on recalled symbols
            else: qq = self.Ph(self.g(torch.cat([r1, r0], -1), h))
            rq = rd(F.normalize(qq, dim=-1))
            w = torch.softmax(self.mix(h), -1); self._mix.append(w.detach()); self._reads.append(torch.stack([r0, r1, rq], 1).detach()); self._hs.append(h.detach()); self._qq.append(F.normalize(qq, dim=-1).detach()); self._pr.append(torch.stack([self.P(r0), self.P(r1)], 1).detach())
            r = w[:, 0:1] * r0 + w[:, 1:2] * r1 + w[:, 2:3] * rq
            outs.append(self.head(torch.cat([h, r], -1)))
            if self.kcons: pr0.append(self.P(r0)); pr1.append(self.P(r1))
            if self.hopce: l0.append(self.head(torch.cat([h, r0], -1))); l1.append(self.head2(torch.cat([h, r1], -1)))
        self.aux = torch.zeros(()); self.gmean = None
        if self.wgate: G = torch.stack(gs, 1).squeeze(-1); self.aux = self.aux + self.wgate * G.mean(); self.gmean = G.detach()
        if self.kcons:   # C132 LEARNING RULE: reads must live where heard tokens live IN ADDRESS SPACE — P r0_t ~ P e_{t+1}, P r1_t ~ P e_{t+2}
            PE = self.P(e).detach()   # adjust the read toward the heard address, never the reverse (C117 lesson)
            A0 = torch.stack(pr0, 1)[:, :-1]; A1 = torch.stack(pr1, 1)[:, :-2]; m0 = (x[:, 1:] != 0).float(); m1 = (x[:, 2:] != 0).float()
            c0 = 1 - F.cosine_similarity(A0, PE[:, 1:], dim=-1); c1 = 1 - F.cosine_similarity(A1, PE[:, 2:], dim=-1)
            self.aux = self.aux + self.kcons * ((c0 * m0).sum() / m0.sum() + (c1 * m1).sum() / m1.sum())
        if self.hopce:   # C129 LEARNING RULE: each hop read must be a standalone predictor of a real stream target (t+1 for r0, t+2 for r1); no oracle
            L0 = torch.stack(l0, 1); L1 = torch.stack(l1, 1)
            self.aux = self.hopce * (F.cross_entropy(L0[:, :-1].reshape(-1, L0.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)
                                     + F.cross_entropy(L1[:, :-2].reshape(-1, L1.shape[-1]), x[:, 2:].reshape(-1), ignore_index=0))
        return torch.stack(outs, 1)

    # ---- exact parallel form (added after T3): W_t = sum_{s<=t} dec^{t-s} kn_s (x) v_s, so q W_t = sum_s dec^{t-s} (q.kn_s) v_s.
    # Reads are computed for all t at once in chunks (decayed causal linear attention with a carried W state); only the small
    # controller GRU stays sequential.  Same function as the loop (checked to 1e-4); used when roles>0 and no wgate/delta.
    def _readall(self, Q, KN, Vv, dec, C=512):
        B, L, dk = KN.shape; d = Vv.shape[-1]; out = []; Wst = torch.zeros(B, dk, d); i = torch.arange(C)
        for s0 in range(0, L, C):
            q = Q[:, s0:s0 + C]; k = KN[:, s0:s0 + C]; v = Vv[:, s0:s0 + C]; n = q.shape[1]
            dist = (i[:n, None] - i[None, :n]).float(); mask = torch.where(dist >= 0, dec ** dist.clamp(min=0), torch.zeros(()))
            S = torch.einsum("bik,bjk->bij", q, k) * mask
            rr = torch.einsum("bij,bjd->bid", S, v) + torch.einsum("bik,bkd->bid", q, Wst) * (dec ** (i[:n] + 1).float())[None, :, None]
            out.append(rr)
            wdec = dec ** (n - 1 - i[:n]).float()
            Wst = (dec ** n) * Wst + torch.einsum("bjk,bjd->bkd", k * wdec[None, :, None], v)
        return torch.cat(out, 1)

    def _rolekey_seq(self, PE, extra=None):   # PE: (B,L,dk) projected tokens; key_t = sum_i R[i] * PE[t-i]
        k = 0
        for i in range(self.roles):
            sh = F.pad(PE, (0, 0, i, 0))[:, :PE.shape[1]] if i else PE
            k = k + self.R[i] * sh
        if self.sparse:
            kk = max(1, int(self.sparse * k.shape[-1])); th = k.abs().topk(kk, dim=-1).values[..., -1:]; k = k * (k.abs() >= th).float()
        return k

    def forward_fast(self, x, **kw):
        B, L = x.shape; d, dk = self.d, self.dk; e = self.E(x); dec = torch.ones(()) if self.nodecay else torch.sigmoid(self.dec)
        PE = self.P(e); K = self._rolekey_seq(PE)                                   # key after seeing token t
        Kprev = F.pad(K, (0, 0, 1, 0))[:, :L]; KN = F.normalize(Kprev, dim=-1)       # write key at t = key of prefix up to t-1 (zero at t=0)
        def RD(Q):   # read with optional cleanup: the match weights become ~(q.k_s)^(2n+1) instead of (q.k_s) — exact recall without softmax
            r = self._readall(Q, KN, e, dec)
            for _ in range(self.clean):
                Q = F.normalize(self._readall(r, e, KN, dec), dim=-1); r = self._readall(Q, KN, e, dec)
            return r
        Q0 = F.normalize(K, dim=-1); R0 = RD(Q0)
        PR0 = self.P(R0)                                  # k1 = rolekey(win + [P r0]): P r0 at offset 0, tokens shift by one
        K1 = self.R[0] * PR0 + sum(self.R[i] * (F.pad(PE, (0, 0, i - 1, 0))[:, :L] if i > 1 else PE) for i in range(1, self.roles))
        if self.sparse:
            kk = max(1, int(self.sparse * dk)); th = K1.abs().topk(kk, dim=-1).values[..., -1:]; K1 = K1 * (K1.abs() >= th).float()
        R1 = RD(F.normalize(K1, dim=-1)); PR1 = self.P(R1)
        sP = self.P(self.shape)
        QQ = self.R[0] * sP + self.R[1] * PR1 + self.R[2] * PR0 if self.roles >= 3 else self.R[0] * sP + self.R[1] * PR1
        if self.sparse:
            kk = max(1, int(self.sparse * dk)); th = QQ.abs().topk(kk, dim=-1).values[..., -1:]; QQ = QQ * (QQ.abs() >= th).float()
        RQ = RD(F.normalize(QQ, dim=-1))
        reads = torch.stack([R0, R1, RQ], 2)                                          # (B,L,3,d)
        if self.pgru:
            prev = F.pad(reads.reshape(B, L, 3 * d), (0, 0, 1, 0))[:, :L]
            H, _ = self.g(torch.cat([e, prev], -1)); Wm = torch.softmax(self.mix(H), -1)
        else:
            h = torch.zeros(B, d); r = torch.zeros(B, d); hs = []; ws = []
            for t in range(L):                                                            # controller only: tiny per-step ops
                h = self.g(torch.cat([e[:, t], r], -1), h); w = torch.softmax(self.mix(h), -1)
                r = (w.unsqueeze(-1) * reads[:, t]).sum(1); hs.append(h); ws.append(w)
            H = torch.stack(hs, 1); Wm = torch.stack(ws, 1)
        Rm = (Wm.unsqueeze(-1) * reads).sum(2)
        self._mix = list(Wm.detach().unbind(1)); self._reads = list(reads.detach().unbind(1)); self._hs = list(H.detach().unbind(1))
        self._kw = list(KN.detach().unbind(1)); self._qq = list(F.normalize(QQ, dim=-1).detach().unbind(1)); self._pr = list(torch.stack([PR0, PR1], 2).detach().unbind(1))
        self.aux = torch.zeros(()); self.gmean = None
        if self.kcons:
            PEd = PE.detach(); A0 = PR0[:, :-1]; A1 = PR1[:, :-2]; m0 = (x[:, 1:] != 0).float(); m1 = (x[:, 2:] != 0).float()
            c0 = 1 - F.cosine_similarity(A0, PEd[:, 1:], dim=-1); c1 = 1 - F.cosine_similarity(A1, PEd[:, 2:], dim=-1)
            self.aux = self.aux + self.kcons * ((c0 * m0).sum() / m0.sum() + (c1 * m1).sum() / m1.sum())
        if self.hopce:
            L0 = self.head(torch.cat([H, R0], -1)); L1 = self.head2(torch.cat([H, R1], -1))
            self.aux = self.hopce * (F.cross_entropy(L0[:, :-1].reshape(-1, L0.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)
                                     + F.cross_entropy(L1[:, :-2].reshape(-1, L1.shape[-1]), x[:, 2:].reshape(-1), ignore_index=0))
        return self.head(torch.cat([H, Rm], -1))

    def forward(self, x, **kw):
        if self.pgru or (self.roles and not self.wgate and not self.delta and not getattr(self, "slow", False)): return self.forward_fast(x, **kw)
        return self.forward_loop(x, **kw)
