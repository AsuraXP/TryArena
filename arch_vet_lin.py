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
    def __init__(self, V, d, keyfn="lin", hops=2, kmult=2, roles=0, hopce=0.0, sparse=0.0, kcons=0.0, nodecay=0, wgate=0.0, delta=0, pgru=0, clean=0, mres=0, rnorm=0, sgate=0, gkey=0, sohead=0.0):
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
        self.sohead = sohead   # O2 OBJECTIVE: store-only auxiliary target — predict byte t+1 from the 1-hop READ ALONE (no controller state), weight
        # `sohead`.  Dense on every byte, stream-only, no oracle: it makes the store a consumer-driven structure (the loss creates the need,
        # NOVELTY §D) so copying from context is trained on every byte of ordinary text rather than once per sample.  hopce (C129) lineage,
        # but the controller is excluded so the gradient cannot route around the store.
        if sohead: self.heads = nn.Linear(d, V)
        self.gkey = gkey   # M6 REPRESENTATION: the address is a gated bound sum of the stream, k_t = g(e_t) k_{t-1} + P e_t, with a LEARNED per-byte
        # forget gate g(e_t) = sigmoid(u.e_t + b).  If the learner sets g ~ 0 on separators the key becomes "the word so far", identical at every
        # mention of a name regardless of what surrounds it -> exact word-aligned copy from a Hebbian store; nothing is hard-coded (no boundary
        # oracle; the byte value is in the stream).  Prior art: data-dependent decay in linear recurrences (GLA, Yang et al. 2023,
        # arXiv:2312.06635; Mamba, Gu & Dao 2023) for the hidden STATE; here it forms the ADDRESS of a fast-weight store (C127 lineage).
        if gkey: self.gu = nn.Linear(d, 1); nn.init.constant_(self.gu.bias, 2.0)
        self.sgate = sgate   # M4 LEARNING RULE: write by SURPRISE — pass 1 reads an ungated store and predicts; the realised next-byte nll of byte t
        # (model-level, detached) gates the write of byte t in pass 2: v_t = sigmoid(s*(nll_t - tau)) e_t.  The store then holds what the
        # learner could not predict (names, facts), not every byte -> capacity is spent on the unpredictable, forgetting is not needed.
        # Prior art: Titans (Behrouz et al. 2024, arXiv:2501.00663) gates/updates a neural memory by the gradient of the MEMORY's own
        # associative loss (delta-rule family) with momentum + decay.  Delta here: the gate is the whole model's realised prediction error
        # on the stream (no inner loss, no gradient), two-pass, onto a Hebbian sparse-address store; O(1)/token, exact parallel form.
        if sgate: self.stau = nn.Parameter(torch.tensor(2.0)); self.ssc = nn.Parameter(torch.tensor(2.0))
        self.rnorm = rnorm   # M3b: scale-free reads (unit-norm * sqrt(d)) so a growing store (nodecay) does not change the read's magnitude
        self.mres = mres   # M3: multi-resolution addresses — separate stores keyed by the last 1, 2 and `roles` bytes (sparse k-WTA per store), so a
        # partial context match (induction copy: same byte, different neighbours) still hits; block weights learned (softmax alpha)
        if mres: self.widths = [w for w in (1, 2, roles) if w <= roles]; self.alpha = nn.Parameter(torch.zeros(len(self.widths)))
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

    def _gscan(self, X, lam, C=64):   # K_t = sum_{s<=t} prod_{u=s+1..t} lam_u X_s, chunked, log-space (stable for any lam in (0,1])
        B, L, dk = X.shape; out = []; carry = torch.zeros(B, dk); Ll = torch.log(lam.clamp(min=1e-6)).squeeze(-1)   # (B,L)
        for s0 in range(0, L, C):
            x = X[:, s0:s0 + C]; ll = Ll[:, s0:s0 + C]; n = x.shape[1]; cs = ll.cumsum(1)   # cs_t = sum_{u<=t} log lam_u (within chunk)
            diff = cs.unsqueeze(2) - cs.unsqueeze(1)                                           # [t,s] = cs_t - cs_s = log prod_{u=s+1..t} lam_u
            D = torch.exp(diff.masked_fill(~torch.tril(torch.ones(n, n, dtype=torch.bool)), float("-inf")))   # s <= t only (mask BEFORE exp: no inf*0 = NaN)
            k = torch.einsum("bts,bsk->btk", D, x) + torch.exp(cs).unsqueeze(-1) * carry.unsqueeze(1)   # carry decays by prod_{u<=t} lam_u
            out.append(k); carry = k[:, -1]
        return torch.cat(out, 1)

    def _forward_gkey(self, x, gate):
        B, L = x.shape; d, dk = self.d, self.dk; e = self.E(x); dec = torch.ones(()) if self.nodecay else torch.sigmoid(self.dec)
        v = e * gate if gate is not None else e
        PE = self.P(e); lam = torch.sigmoid(self.gu(e)); self.lam_seq = lam.detach()
        K = self._kwta(self._gscan(PE, lam)); KN = F.normalize(F.pad(K, (0, 0, 1, 0))[:, :L], dim=-1)
        R0 = self._readall(F.normalize(K, dim=-1), KN, v, dec); PR0 = self.P(R0); g0 = torch.sigmoid(self.gu(R0))
        K1 = self._kwta(g0 * K + PR0); R1 = self._readall(F.normalize(K1, dim=-1), KN, v, dec); PR1 = self.P(R1); g1 = torch.sigmoid(self.gu(R1))
        sP = self.P(self.shape); QQ = self._kwta(g1 * (g0 * sP + PR0) + PR1); RQ = self._readall(F.normalize(QQ, dim=-1), KN, v, dec)
        return self._finish(x, e, PE, R0, R1, RQ, PR0, PR1, KN, QQ)

    def _kwta(self, k):
        if not self.sparse: return k
        kk = max(1, int(self.sparse * k.shape[-1])); th = k.abs().topk(kk, dim=-1).values[..., -1:]; return k * (k.abs() >= th).float()

    def forward_fast(self, x, **kw):
        if not self.sgate: return self._fwd(x, None)
        lg1 = self._fwd(x, None); self.logits1 = lg1
        nll = F.cross_entropy(lg1[:, :-1].reshape(-1, lg1.shape[-1]), x[:, 1:].reshape(-1), reduction="none").view(x.shape[0], -1).detach()
        g = torch.sigmoid(self.ssc * (F.pad(nll, (1, 0), value=5.0) - self.stau)).unsqueeze(-1)   # (B,L,1); byte 0 always written
        self.gmean = g.detach().mean(); self.gate = g.detach()
        out = self._fwd(x, g)
        self.aux = self.aux + 0.5 * F.cross_entropy(lg1[:, :-1].reshape(-1, lg1.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)   # pass 1 must stay a predictor (it defines surprise)
        return out

    def _fwd(self, x, gate):
        if self.gkey: return self._forward_gkey(x, gate)
        if self.mres: return self._forward_mres(x, gate)
        B, L = x.shape; d, dk = self.d, self.dk; e = self.E(x); dec = torch.ones(()) if self.nodecay else torch.sigmoid(self.dec)
        v = e * gate if gate is not None else e
        PE = self.P(e); K = self._rolekey_seq(PE)                                   # key after seeing token t
        Kprev = F.pad(K, (0, 0, 1, 0))[:, :L]; KN = F.normalize(Kprev, dim=-1)       # write key at t = key of prefix up to t-1 (zero at t=0)
        def RD(Q):   # read with optional cleanup: the match weights become ~(q.k_s)^(2n+1) instead of (q.k_s) — exact recall without softmax
            r = self._readall(Q, KN, v, dec)
            for _ in range(self.clean):
                Q = F.normalize(self._readall(r, v, KN, dec), dim=-1); r = self._readall(Q, KN, v, dec)
            return r
        Q0 = F.normalize(K, dim=-1); R0 = RD(Q0)
        PR0 = self.P(R0)                                  # k1 = rolekey(win + [P r0]): P r0 at offset 0, tokens shift by one
        K1 = self.R[0] * PR0 + sum(self.R[i] * (F.pad(PE, (0, 0, i - 1, 0))[:, :L] if i > 1 else PE) for i in range(1, self.roles))
        K1 = self._kwta(K1)
        R1 = RD(F.normalize(K1, dim=-1)); PR1 = self.P(R1)
        sP = self.P(self.shape)
        QQ = self.R[0] * sP + self.R[1] * PR1 + self.R[2] * PR0 if self.roles >= 3 else self.R[0] * sP + self.R[1] * PR1
        QQ = self._kwta(QQ)
        RQ = RD(F.normalize(QQ, dim=-1))
        return self._finish(x, e, PE, R0, R1, RQ, PR0, PR1, KN, QQ)

    def _forward_mres(self, x, gate=None):
        B, L = x.shape; d, dk = self.d, self.dk; e = self.E(x); dec = torch.ones(()) if self.nodecay else torch.sigmoid(self.dec)
        v = e * gate if gate is not None else e
        PE = self.P(e); a = torch.softmax(self.alpha, 0); sP = self.P(self.shape)
        def keyw(parts, w):   # parts: list of (B,L,dk) most-recent-first; key of width w
            return self._kwta(sum(self.R[i] * parts[i] for i in range(min(w, len(parts)))))
        shifted = [F.pad(PE, (0, 0, i, 0))[:, :L] if i else PE for i in range(self.roles)]
        R0s, R1s, RQs, KNs, QQs = [], [], [], [], []
        for bi, w in enumerate(self.widths):
            K = keyw(shifted, w); KN = F.normalize(F.pad(K, (0, 0, 1, 0))[:, :L], dim=-1)
            R0 = self._readall(F.normalize(K, dim=-1), KN, v, dec); PR0 = self.P(R0)
            K1 = keyw([PR0] + shifted[:-1], w); R1 = self._readall(F.normalize(K1, dim=-1), KN, v, dec); PR1 = self.P(R1)
            QQ = keyw([sP.expand(B, L, -1), PR1, PR0], w); RQ = self._readall(F.normalize(QQ, dim=-1), KN, v, dec)
            R0s.append(a[bi] * R0); R1s.append(a[bi] * R1); RQs.append(a[bi] * RQ); KNs.append(KN); QQs.append(QQ)
        R0 = sum(R0s); R1 = sum(R1s); RQ = sum(RQs)
        return self._finish(x, e, PE, R0, R1, RQ, self.P(R0), self.P(R1), KNs[-1], QQs[-1])

    def _finish(self, x, e, PE, R0, R1, RQ, PR0, PR1, KN, QQ):
        B, L = x.shape; d = self.d
        if self.rnorm: R0, R1, RQ = (F.normalize(R, dim=-1) * math.sqrt(d) ** 0.5 for R in (R0, R1, RQ))
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
        self.aux = torch.zeros(()); self.gmean = None if not self.sgate else getattr(self, 'gmean', None)
        if self.kcons:
            PEd = PE.detach(); A0 = PR0[:, :-1]; A1 = PR1[:, :-2]; m0 = (x[:, 1:] != 0).float(); m1 = (x[:, 2:] != 0).float()
            c0 = 1 - F.cosine_similarity(A0, PEd[:, 1:], dim=-1); c1 = 1 - F.cosine_similarity(A1, PEd[:, 2:], dim=-1)
            self.aux = self.aux + self.kcons * ((c0 * m0).sum() / m0.sum() + (c1 * m1).sum() / m1.sum())
        if self.sohead:
            Ls = self.heads(R0); self.aux = self.aux + self.sohead * F.cross_entropy(Ls[:, :-1].reshape(-1, Ls.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)
        if self.hopce:
            L0 = self.head(torch.cat([H, R0], -1)); L1 = self.head2(torch.cat([H, R1], -1))
            self.aux = self.hopce * (F.cross_entropy(L0[:, :-1].reshape(-1, L0.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)
                                     + F.cross_entropy(L1[:, :-2].reshape(-1, L1.shape[-1]), x[:, 2:].reshape(-1), ignore_index=0))
        return self.head(torch.cat([H, Rm], -1))

    def forward(self, x, **kw):
        if self.pgru or self.sgate or self.mres or self.gkey or (self.roles and not self.wgate and not self.delta and not getattr(self, "slow", False)): return self.forward_fast(x, **kw)
        return self.forward_loop(x, **kw)
