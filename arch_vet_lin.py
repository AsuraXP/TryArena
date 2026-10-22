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
    def __init__(self, V, d, keyfn="lin", hops=2, kmult=2, roles=0, hopce=0.0, sparse=0.0, kcons=0.0, nodecay=0):
        super().__init__()
        dk = d * kmult; self.d, self.dk, self.keyfn, self.hops = d, dk, keyfn, hops
        self.sparse = sparse; self.kcons = kcons; self.nodecay = nodecay
        self.E = nn.Embedding(V, d); self.P = nn.Linear(d, dk, bias=False);
        if sparse:   # C131: Kanerva-style address — FIXED random projection (no parameters) + k-WTA; crosstalk falls with dk at zero parameter cost
            self.P.weight.requires_grad_(False); nn.init.normal_(self.P.weight, std=1.0 / math.sqrt(d))
        self.Ph = nn.Linear(d, dk, bias=False) if keyfn == 'ctrl' else None
        self.lam = nn.Parameter(torch.full((dk,), 1.0)); self.dec = nn.Parameter(torch.tensor(4.0))   # W decay sigmoid(4)=.98   # sigmoid -> ~.73 start
        self.shape = nn.Parameter(torch.randn(d) * 0.1); self.roles = roles
        if roles: self.R = nn.Parameter(torch.randn(roles, dk) * 0.5 + 1.0)   # C128: learned role per offset (TPR with recency roles)
        self.g = nn.GRUCell(2 * d, d)
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

    def forward(self, x, **kw):
        B, L = x.shape; d, dk = self.d, self.dk; e = self.E(x)
        h = torch.zeros(B, d); k = torch.zeros(B, dk); W = torch.zeros(B, dk, d); r = torch.zeros(B, d); outs = []; self._mix = []; self._reads = []; self._hs = []; self._kw = []; self._qq = []; self._pr = []; win = []; l0, l1 = [], []; pr0, pr1 = [], []
        lam = torch.sigmoid(self.lam); sP = self.P(self.shape).unsqueeze(0)
        for t in range(L):
            et = e[:, t]
            # write: previous key -> this token (normalised key so W stays bounded)
            kn = F.normalize(k, dim=-1); self._kw.append(kn.detach())   # heard key used for the write at step t
            W = (W if self.nodecay else torch.sigmoid(self.dec) * W) + kn.unsqueeze(2) * et.unsqueeze(1)
            # advance key and controller
            if self.roles:
                win.append(self.P(et)); win = win[-self.roles:]; k = self.rolekey(win)
            else: k = self.key(k, et, h)
            h = self.g(torch.cat([et, r], -1), h)
            q0 = F.normalize(k, dim=-1); r0 = torch.einsum("bk,bkd->bd", q0, W)           # 1-hop: what followed this prefix
            if self.roles: k1 = self.rolekey(win + [self.P(r0)])
            else: k1 = self.key(k, r0, h)
            q1 = F.normalize(k1, dim=-1); r1 = torch.einsum("bk,bkd->bd", q1, W)   # continuation (next symbol)
            if self.roles: qq = self.rolekey([self.P(r0), self.P(r1), sP.expand(B, -1)])   # RE-POSE with the same roles
            elif self.keyfn == "lin": qq = lam * lam * self.P(r0) + lam * self.P(r1) + sP   # RE-POSE: writer's key function on recalled symbols
            else: qq = self.Ph(self.g(torch.cat([r1, r0], -1), h))
            rq = torch.einsum("bk,bkd->bd", F.normalize(qq, dim=-1), W)
            w = torch.softmax(self.mix(h), -1); self._mix.append(w.detach()); self._reads.append(torch.stack([r0, r1, rq], 1).detach()); self._hs.append(h.detach()); self._qq.append(F.normalize(qq, dim=-1).detach()); self._pr.append(torch.stack([self.P(r0), self.P(r1)], 1).detach())
            r = w[:, 0:1] * r0 + w[:, 1:2] * r1 + w[:, 2:3] * rq
            outs.append(self.head(torch.cat([h, r], -1)))
            if self.kcons: pr0.append(self.P(r0)); pr1.append(self.P(r1))
            if self.hopce: l0.append(self.head(torch.cat([h, r0], -1))); l1.append(self.head2(torch.cat([h, r1], -1)))
        self.aux = torch.zeros(())
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
