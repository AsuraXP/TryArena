#!/usr/bin/env python3
"""C97 / P49 — CORE LANE: CHUNK-KEYED BINDING (CKB) — the unit the learner makes.
P47's slot-binding core learned key<-previous token, value<-current token, exact and
length-free, but on bytes nothing is a unit (C96).  Here the unit is learned:
  chunker  c_t = GRU_c(e_t, c_{t-1} * (1 - b_{t-1}))      resets at boundaries -> c is context-free
  boundary b_t = gmin + (1-gmin) * sigmoid(W_b [h_t, c_t])  (leaky; the boundary IS the write event)
  write    at boundary: key = c of the PREVIOUS chunk, value = c of THIS chunk   (chunk bigram binding)
           addr = p*match + (1-p)*emptiest  (as P47), alpha = b_t * addr
  read     at boundary: q = c_t -> r = sum a_j V_j   ("what followed this chunk last time");
           latched r_l <- (1-b) r_l + b r so the head sees it while spelling the next chunk
  output   head([h_t, r_l, conf]) ; controller h_t = GRU_h(e_t)
C116 (--inner S): INNER RECALL. At each position the key-state re-consumes its own read (the recalled
next-token embedding) through the same recurrence gc, reads the slots again, S times; a learned mixture over
hops feeds the head. Prior art: DiscoLoop (arXiv 2607.00341) loops a Transformer with a decoded discrete
embedding channel over PARAMETRIC memory; memory networks / looped transformers do fixed hops with separate
layers. Delta: hops run the recurrent key-state itself on recalled tokens from the EPISODIC slot state (facts
of this context), the same embedding path as heard tokens (keys stay invariant), O(S) per token at any length.
C118 (--look D): LOOK-AHEAD VALUES. Besides the next-token embedding, each slot stores (written D steps later,
from heard data) the learner's own key-state D steps after the write. A recalled look-ahead state IS a key, so
the second hop is one ordinary read; a 2-way learned mixture (from h) blends hop-0 and hop-1 reads. Prior art:
MARCH (arXiv 2608.12435) banks PAST recurrent states as retrievable anchors; KV-MemNNs do multi-hop with
separate hop layers over token-typed values. Delta: value channel typed as the learner's FUTURE key-state,
delayed write, hop = read-with-recalled-key, no imagination, no hop supervision.
Everything learned; loss = CE only; no position signal; slots are state (grown at eval).
C106 (--bmode util): the UNIT IS DEFINED BY ITS CONSUMER, BACKWARDS. The reset/boundary b_t is a
sampled hard action that receives NO gradient from CE. Its only learning signal is a local reward:
a cut is rewarded when the keys it started (a) later RECUR as a hard slot match and (b) that read
lowered CE against a counterfactual no-read head. REINFORCE on b with that reward. Prior art and
delta: REBORN (Tseng et al. 2024, arXiv 2402.03988) trains speech segment boundaries by REINFORCE
with a perplexity-difference reward from a separate phoneme LM; MGPO (arXiv 2609.37930) credits
textual-memory rewrites by counterfactual utility under a frozen reader; LiB (Yang 2022) selects
units by recurrence with counterfactual evaluation, no neural learner. Delta: byte-level, online,
inside one learner; reward = key recurrence x read gain of the learner's own slot state; no frozen
reader, no separate LM, trained jointly with CE on everything else.
Prior art (searched 2026-10-06): HM-RNN Chung+ 2016 (learned binary boundaries via ST,
feeding a layer hierarchy, no memory, no cardinality control); Gated DeltaNet-2
arXiv:2605.22791 (erase/write gates on a dense state, no unit).  Delta: boundaries
define the KEY of a slot binding and the write event in a bounded set-structured
state; keys are context-free by reset; read/write agreement holds by construction.
Tests: (a) synthetic P46 binding task (must hold); (b) entity re-mention on text with
UNSEEN names and x4 distance: CE on the name bytes at re-mention vs GRU / TF.
"""
import sys, json, random, time, math, argparse, re
import torch, torch.nn as nn, torch.nn.functional as F
sys.argv_p49 = list(sys.argv); sys.argv = ["x"]
import arch_vet_lm as A
import arch_vet_p46 as T
sys.argv = sys.argv_p49
torch.set_num_threads(2)
GMIN0, BINIT = 0.2, 0.0
BMODE = "soft"
SURPRISE = False
PKB = False
CGW = False   # C101: content-gated window taps (learned per-byte pass factor cuts context)
DENSE = False
PUREADDR = False   # C103b: address = softmax(k.K/tau) only (no empty branch, no p)
# C103b pure content addressing = Tyulmankov et al. NeurIPS 2021 (softmax key match, slot overwrite), ALER arXiv 2610.00592 (one-slot Gumbel overwrite); here framed as a fix, not a novelty claim.
ROLE = 0   # C103: dim of role bottleneck z = tanh(W_z h) appended to the prefix key (0 = off)   # C102: train on dense-reuse dialogues
WKB = 0   # C100: key = learned per-tap mix of the last WKB byte embeddings (no boundary, no gate)
IMCONS = 0.0   # C117: imagination-consistency weight — gc(read_{t-1}, c_{t-1}) must land on the heard state c_t (self-target, no supervision)
LOOKH = 3   # C118 number of chained look-ahead hops
LOOK = 0   # C118: look-ahead values — each slot also stores the learner's own key-state LOOK steps after the write (delayed write from heard data); hop 2 = ordinary read with the recalled key
HOPX = 0.0   # C121: hop-exploration credit. Loop: read-space hop mixture gives hop k gradient ∝ w_k; w_k≈0 ⇒ no gradient ⇒ w_k stays 0 (rich-get-richer in the
# hop gate), so intermediate hops never learn to be good intermediates (C116–C120, weights frozen [.4,0,0,.6]). Removal (learning rule, not wiring): HARD hop
# choice per position; with prob HOPX the hop is drawn uniformly (exploration), else the gate's argmax; the chosen hop alone feeds the head, so each hop gets
# the FULL CE gradient on its own path whenever drawn; the gate is trained separately toward the hop with the lowest per-hop CE (detached target).
# Related but different: ε-greedy / noisy-top-k expert routing (Shazeer et al. 2017 arXiv:1701.06538) adds noise to a soft gate for load balance; expert-
# choice routing (Zhou et al. 2022 arXiv:2202.09368) routes tokens by expert; neither trains a gate on a per-expert loss oracle over READ DEPTH in a memory.
REPOSE = 0   # C123b: RE-POSE. Diagnostic C123a showed the inner chain chases the first fact's continuation (hop1 -> b1 slot .62, hop2 -> DOT slot .5,
# hop1 -> goal slot 0.0): consuming a recalled token into the CURRENT state yields the key "a0 a1 = b0", never "b0 b1 =". Removal: a second, FRESH
# state track consumes the recalled tokens from zero and then a learned query-shape vector; the read with that key is offered to the hop gate.
# Nothing is given: the shape vector, the fresh-track GRU (shared gc) and the gate are all learned. Prior art checked: latent query reformulation
# for multi-hop QA (IP-LQR, 2022) updates a question encoding with retrieved context in a Transformer QA reader; KV-MemNN hops add the read to
# the query vector. Neither re-runs the sequence model from a blank state over the recalled tokens to mint a fresh prefix key.
INNER = 0   # C116: inner recall — the state re-consumes its own read (a next-token embedding) through gc, reads again, S times; learned hop mixture
LOWMEM = 0   # hard index write (memory O(B d)/step); drops the straight-through address gradient
KNORM = 0   # C111: size-free keys — the c-part of query and stored keys is L2-normalised to norm 4 before distance match/read
DSCALE = 0   # C109: scale the hard-match threshold with sqrt(d/32) so the state keys' match rule is size-free
MTHR = 1.0
CCRED = 0   # C108c complementary credit: gradient into the h/head path scaled by (1 - read confidence) per position
HCUT = 0.0   # C108d control: random hard reset of predictor state h in training with this prob
RANDP = 0.1   # C106b fixed reset probability
UTILW = 1.0   # C106 weight of the boundary REINFORCE loss
MSLOTS = 8   # C99: key = prefix state c_t, value = next byte embedding, written every step   # C98: boundary also sees the learner's own surprise -log p(x_t) and entropy H_{t-1}   # soft | st (straight-through hard boundary) | force (b=1 every token: wiring control)


class CKB(nn.Module):
    def __init__(self, V, d, M=8):
        super().__init__(); self.V, self.d, self.M = V, d, M
        self.E = nn.Embedding(V, d); self.gh = nn.GRUCell(d, d); self.gc = nn.GRUCell(d, d)
        self.b = nn.Linear(2 * d + 2, 1); nn.init.constant_(self.b.bias, BINIT)
        self.theta = nn.Parameter(torch.tensor(1.0)); self.K0 = nn.Parameter(torch.randn(max(32, M), d + ROLE) * 0.5); self.register_buffer('ramp', torch.arange(max(32, M)).float()[:M] * 1e-3); self.beta = nn.Parameter(torch.tensor(0.1))
        self.head = nn.Linear(2 * d + 1, V)
        self.hopw = nn.Linear(d, max(INNER, 0) + 1 + (INNER if REPOSE else 0)); self.qshape = nn.Parameter(torch.randn(d) * 0.1); self.hop2 = nn.Linear(d, LOOKH + 1); self.z = nn.Linear(d, max(ROLE, 1)); self.taps = nn.Parameter(torch.full((max(WKB, 1), d), 0.5)); self.ws = nn.Linear(d, 1); nn.init.constant_(self.ws.bias, 2.2)

    def wkey(self, e, t, B, d):
        win = torch.stack([e[:, t - i] if t - i >= 0 else torch.zeros(B, d) for i in range(WKB)], 1)   # B x W x d (i=0 newest)
        if CGW:
            sp = torch.sigmoid(self.ws(win)).squeeze(-1)                       # B x W pass factors
            gate = torch.cumprod(torch.cat([torch.ones(B, 1), sp[:, :-1]], 1), 1)   # tap i passes through bytes j<i
            return (self.taps.unsqueeze(0) * gate.unsqueeze(-1) * win).sum(1)
        return (self.taps.unsqueeze(0) * win).sum(1)

    def forward(self, x, tau=0.1, gmin=0.0, M=None, states=False):
        B, L = x.shape; d = self.d; M = M or self.M; e = self.E(x)
        h = torch.zeros(B, d); c = torch.zeros(B, d); rl = torch.zeros(B, d); conf = torch.zeros(B, 1); pk = torch.zeros(B, d)
        dk = d + ROLE; K = self.K0[:M].unsqueeze(0).expand(B, M, dk).clone(); Vm = torch.zeros(B, M, d); occ = torch.zeros(B, M); pk = torch.zeros(B, dk); usage = torch.zeros(B, M)
        outs, bs = [], []; surp = torch.zeros(B, 2)
        UTIL = BMODE == "util"; cutpos = torch.zeros(B, dtype=torch.long); slot_cut = torch.zeros(B, M, dtype=torch.long); Rw = torch.zeros(B, L); logps = []
        im_loss = torch.zeros(()); hx_loss = torch.zeros(()); r0_prev = None; c_prev = None; self._w = []; self._ah = []; self._jw = []; V2 = torch.zeros(B, M, d); jhist = []; self._w2 = []
        for t in range(L):
            c_prev = c; h = self.gh(e[:, t], h); c = self.gc(e[:, t], c)
            if IMCONS and r0_prev is not None and self.training:
                c_im = self.gc(r0_prev, c_prev); im_loss = im_loss + ((c_im - c.detach()) ** 2).mean() / L
            b = gmin + (1 - gmin) * torch.sigmoid(self.b(torch.cat([h, c, surp if SURPRISE else torch.zeros(B, 2)], -1)))
            if BMODE == "st": b = (b > 0.5).float() + b - b.detach()
            elif BMODE == "rand": b = (torch.bernoulli(torch.full_like(b, RANDP)) if self.training else torch.zeros_like(b)).detach()   # C106b ablation: fixed-rate random hard resets, no learning of b
            elif UTIL:
                pb = b; b = (torch.bernoulli(pb) if self.training else (pb > 0.5).float()).detach()
                logps.append(torch.log(torch.where(b > 0.5, pb, 1 - pb) + 1e-6).squeeze(-1))
            elif BMODE == "force": b = torch.ones_like(b)
            elif BMODE == "oracle":   # CONTROL ONLY: unit given = separator bytes (space, newline, , . : ? !)
                xt = x[:, t]; b = ((xt == 32) | (xt == 10) | (xt == 44) | (xt == 46) | (xt == 58) | (xt == 63) | (xt == 33)).float().unsqueeze(-1)
            bs.append(b)
            logocc = torch.log(occ + 1e-4)
            # read with the chunk so far (at a boundary: the whole chunk)
            if WKB:
                qv = self.wkey(e, t, B, d)
            else: qv = torch.cat([c, torch.tanh(self.z(h))], -1) if ROLE else c
            if KNORM: qv = torch.cat([F.normalize(qv[:, :d], dim=-1) * 4.0, qv[:, d:]], -1)
            if PUREADDR == 3: sim = -torch.cdist(qv[:, :d].unsqueeze(1), K[:, :, :d]).squeeze(1); a = torch.softmax(sim / tau + logocc, -1)
            else: sim = torch.einsum("bd,bmd->bm", qv, K) / math.sqrt(d); a = torch.softmax((sim + (0 if PUREADDR else logocc)) / tau, -1)
            r = torch.einsum("bm,bmd->bd", a, Vm); cf = (a * occ).sum(-1, keepdim=True)
            r0_prev = r
            if LOOK and PUREADDR == 3:   # C118: recalled key-typed value -> ordinary reads, chained LOOKH times
                rs2, cfs2, kk = [r], [cf], torch.einsum("bm,bmd->bd", a, V2)
                for _h in range(LOOKH):
                    s2 = -torch.cdist(kk.unsqueeze(1), K[:, :, :d]).squeeze(1); a2 = torch.softmax(s2 / tau + logocc, -1)
                    rs2.append(torch.einsum("bm,bmd->bd", a2, Vm)); cfs2.append((a2 * occ).sum(-1, keepdim=True)); kk = torch.einsum("bm,bmd->bd", a2, V2)
                w2 = torch.softmax(self.hop2(h), -1); self._w2.append(w2.detach())
                r = sum(w2[:, k:k + 1] * rs2[k] for k in range(LOOKH + 1)); cf = sum(w2[:, k:k + 1] * cfs2[k] for k in range(LOOKH + 1))
            if INNER and PUREADDR == 3:   # C116 inner recall: imagine the recalled token, read again (S hops); learner mixes hops
                rs, cfs, cs, rk = [r], [cf], c, r; ahs = [a.argmax(-1)]
                for _k in range(INNER):
                    cs = self.gc(rk, cs); qs = cs
                    if KNORM: qs = F.normalize(qs, dim=-1) * 4.0
                    sk = -torch.cdist(qs.unsqueeze(1), K[:, :, :d]).squeeze(1); ak = torch.softmax(sk / tau + logocc, -1)
                    rk = torch.einsum("bm,bmd->bd", ak, Vm); rs.append(rk); cfs.append((ak * occ).sum(-1, keepdim=True)); ahs.append(ak.argmax(-1))
                if REPOSE:   # C123b fresh track: blank state consumes the recalled tokens, then the learned query shape; read with that key
                    cq = torch.zeros(B, d)
                    for _k in range(INNER):
                        rin = rs[_k]
                        if REPOSE == 2:   # C123c: sharp recalled token — snap the soft read to its nearest embedding (straight-through)
                            idx = torch.cdist(rin.unsqueeze(1), self.E.weight.unsqueeze(0)).squeeze(1).argmin(-1); rin = rin + (self.E.weight[idx] - rin).detach()
                        cq = self.gc(rin, cq); qq = self.gc(self.qshape.unsqueeze(0).expand(B, d), cq)
                        if KNORM: qq = F.normalize(qq, dim=-1) * 4.0
                        sq = -torch.cdist(qq.unsqueeze(1), K[:, :, :d]).squeeze(1); aq = torch.softmax(sq / tau + logocc, -1)
                        rs.append(torch.einsum("bm,bmd->bd", aq, Vm)); cfs.append((aq * occ).sum(-1, keepdim=True)); ahs.append(aq.argmax(-1))
                w = torch.softmax(self.hopw(h), -1); self._w.append(w.detach()); self._ah.append(torch.stack(ahs, 1))
                if HOPX > 0:   # C121 hard hop choice with exploration; gate trained toward the per-hop-CE winner
                    nh = INNER + 1; j = w.argmax(-1)
                    if self.training:
                        ex = torch.rand(B) < HOPX; j = torch.where(ex, torch.randint(0, nh, (B,)), j)
                        if t + 1 < L:
                            with torch.no_grad():
                                y = x[:, t + 1]; ces = torch.stack([F.cross_entropy(self.head(torch.cat([h, rs[k], cfs[k]], -1)), y, reduction="none") for k in range(nh)], 1)
                            hx_loss = hx_loss + F.cross_entropy(self.hopw(h), ces.argmin(-1)) / L
                    oh = F.one_hot(j, nh).float()
                    r = sum(oh[:, k:k + 1] * rs[k] for k in range(nh)); cf = sum(oh[:, k:k + 1] * cfs[k] for k in range(nh))
                else:
                    r = sum(w[:, k:k + 1] * rs[k] for k in range(len(rs))); cf = sum(w[:, k:k + 1] * cfs[k] for k in range(len(rs)))
            if PKB or WKB: rl, conf = r, cf
            else: rl = (1 - b) * rl + b * r; conf = (1 - b) * conf + b * cf
            if CCRED:
                W = self.head.weight; lg_h = h @ W[:, :d].t() + self.head.bias; lg_r = torch.cat([rl, conf], -1) @ W[:, d:].t()
                g = conf.detach().clamp(0, 1); lg = lg_h.detach() + (lg_h - lg_h.detach()) * (1 - g) + lg_r
            else: lg = self.head(torch.cat([h, rl, conf], -1))
            if HCUT > 0 and self.training: h = h * (torch.rand(B, 1) > HCUT).float()
            if TIE: lg = lg + self.beta * (rl @ self.E.weight.t())
            outs.append(lg)
            if UTIL and t + 1 < L:   # reward: hard recurrence x counterfactual read gain
                with torch.no_grad():
                    lg0 = self.head(torch.cat([h, torch.zeros_like(rl), torch.zeros_like(conf)], -1)); y = x[:, t + 1]
                    gain = (F.cross_entropy(lg0, y, reduction="none") - F.cross_entropy(lg.detach(), y, reduction="none")).clamp(min=0)
                    amax, jstar = a.max(-1); hit = (amax > 0.5) & (gain > 0.05); g = gain * hit.float()
                    Rw.scatter_add_(1, slot_cut.gather(1, jstar[:, None]), g[:, None]); Rw.scatter_add_(1, cutpos[:, None], g[:, None])
            if SURPRISE and t + 1 < L:
                lp = F.log_softmax(lg.detach(), -1); surp = torch.stack([-lp.gather(-1, x[:, t + 1:t + 2]).squeeze(-1) / 5.0, -(lp.exp() * lp).sum(-1) / 5.0], -1)
            if WKB:
                kw = qv
                wsim = torch.einsum("bd,bmd->bm", pk, K) / math.sqrt(d); m = torch.softmax((wsim + logocc) / tau, -1)
                p = torch.sigmoid(wsim.max(-1, keepdim=True).values - self.theta); empty = torch.softmax(-occ / tau, -1)
                alpha = (p * m + (1 - p) * empty).unsqueeze(-1)
                K = (1 - alpha) * K + alpha * pk.unsqueeze(1); Vm = (1 - alpha) * Vm + alpha * e[:, t].unsqueeze(1); occ = occ + alpha.squeeze(-1) * (1 - occ)
                pk = kw; continue
            if PKB:
                # read: what followed this prefix last time?  (q = c_t; r already computed above with q = c)
                # write: key = previous prefix state (pk), value = this byte's embedding
                wsim = torch.einsum("bd,bmd->bm", pk, K) / math.sqrt(d)
                if PUREADDR == 3:   # C103d: HARD ST allocation: match if c-part key dist<1 else LRU(+ramp); read is distance-based
                    dmat = torch.cdist(pk[:, :d].unsqueeze(1), K[:, :, :d]).squeeze(1); jm = dmat.argmin(-1); ok = dmat.gather(1, jm[:, None]).squeeze(1) < (MTHR * math.sqrt(d / 32) if DSCALE else MTHR)
                    jf = (usage + self.ramp).argmin(-1); j = torch.where(ok, jm, jf); hard = F.one_hot(j, M).float()
                    soft = torch.softmax(-dmat / tau, -1) * ok[:, None].float() + torch.softmax(-(usage + self.ramp) / tau, -1) * (~ok)[:, None].float()
                    alpha = (hard + soft - soft.detach()).unsqueeze(-1); usage = 0.98 * usage + hard + a.detach()
                elif PUREADDR == 2:   # C103c: match else LRU-allocate (usage + index ramp breaks ties)
                    m = torch.softmax(wsim / tau, -1); p = torch.sigmoid((wsim.max(-1, keepdim=True).values - self.theta) / 0.1)
                    free = torch.softmax((-usage - self.ramp) / tau, -1); alpha = (p * m + (1 - p) * free).unsqueeze(-1)
                    usage = 0.98 * usage + alpha.squeeze(-1) + a
                elif PUREADDR: alpha = torch.softmax(wsim / tau, -1).unsqueeze(-1)
                else:
                    m = torch.softmax((wsim + logocc) / tau, -1); p = torch.sigmoid(wsim.max(-1, keepdim=True).values - self.theta); empty = torch.softmax(-occ / tau, -1)
                    alpha = (p * m + (1 - p) * empty).unsqueeze(-1)
                if LOWMEM and PUREADDR == 3:   # hard row write: O(B d) saved per step instead of O(B M d); no ST address gradient
                    bi = torch.arange(B); self._jw.append(j); K = K.clone(); K[bi, j] = pk; Vm = Vm.clone(); Vm[bi, j] = e[:, t]; occ = occ + hard * (1 - occ)
                    if LOOK:
                        jhist.append(j)
                        if len(jhist) > LOOK: jo = jhist.pop(0); V2 = V2.clone(); V2[bi, jo] = qv[:, :d]
                else:
                    K = (1 - alpha) * K + alpha * pk.unsqueeze(1); Vm = (1 - alpha) * Vm + alpha * e[:, t].unsqueeze(1); occ = occ + alpha.squeeze(-1) * (1 - occ)
                if UTIL:
                    if PUREADDR == 3: slot_cut.scatter_(1, j[:, None], cutpos[:, None])
                    cutpos = torch.where(b.squeeze(-1) > 0.5, torch.full_like(cutpos, min(t + 1, L - 1)), cutpos)
                pk = qv; c = c * (1 - b); continue
            # write (previous chunk -> this chunk) at the boundary
            wsim = torch.einsum("bd,bmd->bm", pk, K) / math.sqrt(d); m = torch.softmax((wsim + logocc) / tau, -1)
            p = torch.sigmoid(wsim.max(-1, keepdim=True).values - self.theta); empty = torch.softmax(-occ / tau, -1)
            alpha = (b * (p * m + (1 - p) * empty)).unsqueeze(-1)
            K = (1 - alpha) * K + alpha * pk.unsqueeze(1); Vm = (1 - alpha) * Vm + alpha * c.unsqueeze(1); occ = occ + alpha.squeeze(-1) * (1 - occ)
            pk = (1 - b) * pk + b * c; c = c * (1 - b)
        logits = torch.stack(outs, 1); bs = torch.cat(bs, 1); self.im_loss = im_loss; self.hx_loss = hx_loss
        if UTIL and self.training:
            lp = torch.stack(logps, 1); adv = (Rw - Rw.mean()) / (Rw.std() + 1e-3); self.aux_loss = -(adv.detach() * lp).mean(); self.aux_R = Rw.mean().item()
        return (logits, bs, occ) if states else logits


class GRUCore(nn.Module):
    def __init__(self, V, d):
        super().__init__(); self.E = nn.Embedding(V, d); self.g = nn.GRU(d, d, batch_first=True); self.h = nn.Linear(d, V)

    def forward(self, x, **kw): return self.h(self.g(self.E(x))[0])


def n_params(m): return sum(p.numel() for p in m.parameters())


# ---------------- entity re-mention text ----------------
RAW = open("corpus/chat_dialogues_big.txt", "rb").read().decode("utf8", "ignore").lower()
LINES = [l.strip() for l in RAW.split("\n") if 30 <= len(l.strip()) <= 110 and not any(ch.isupper() for ch in l)]
NAMES_TR = "priya dana tomas lena arjun mina felix nora ivan sara leo maya omar zoe hugo iris kian ruby nico vera idris jade finn luna oscar tara rafi elsa milo nadia".split()
NAMES_OOD = "yuki bram cleo dax ewan faye gil hana ike juno kofi lior mehdi nell otto pia quin rosa sven tove umar wren xiu yara zane bea cyd dov eli fay".split()
INTRO = ["a: my name is {n}.", "a: hi, i am {n}.", "a: people call me {n}.", "a: {n} here, nice to meet you."]
REMEN = ["b: nice to meet you {n}.", "b: {n}, are you ok?", "b: so {n}, what do you do?", "b: good night {n}.", "a: as i said, i am {n}."]


def dialogue(rng, names, nfill):
    n = rng.choice(names); lines = [rng.choice(INTRO).format(n=n)]
    for _ in range(nfill): lines.append(rng.choice("ab") + ": " + rng.choice(LINES))
    rm = rng.choice(REMEN); pre, _ = rm.split("{n}"); lines.append(rm.format(n=n))
    txt = "\n".join(lines) + "\n"; start = len("\n".join(lines[:-1])) + 1 + len(pre)
    return txt.encode(), start, start + len(n)


TIE = False   # C103f: logits += beta * E r (copy logit through the embedding table; one learned scalar)
RANDNAMES = False   # C103e: training names are fresh random CV strings each dialogue -> un-memorizable, copying is the only route
def rand_name(rng):
    C, V = "bcdfghjklmnprstvwz", "aeiou"; n = rng.randint(2, 3)
    return "".join(rng.choice(C) + rng.choice(V) for _ in range(n)) + (rng.choice(C) if rng.random() < 0.4 else "")


def dialogue_dense(rng, names, nlines):
    """3 names recur on nearly every line; returns bytes and spans of every name occurrence after the first per name."""
    ns = [rand_name(rng) for _ in range(3)] if names is None else rng.sample(names, 3); seen = set(); lines = []; spans = []; pos = 0
    tpl = ["{a}: hey {b}, " , "{a}: ", "{a}: {b}, ", "{a}: i told {b} that "]
    for _ in range(nlines):
        a, b = rng.sample(ns, 2); t = rng.choice(tpl); fill = rng.choice(LINES)[:60]
        line = t.format(a=a, b=b) + fill + "\n"
        for nm in (a, b):
            i = line.find(nm)
            while i >= 0:
                if nm in seen: spans.append((pos + i, pos + i + len(nm)))
                i = line.find(nm, i + 1)
            seen.add(nm)
        lines.append(line); pos += len(line)
    return "".join(lines).encode(), spans


def dense_batch(rng, B, names, nlines, L):
    xs, spans = [], []
    for _ in range(B):
        t, sp = dialogue_dense(rng, names, nlines); t = list(t[:L + 1]); spans.append([(s0, min(s1, L)) for s0, s1 in sp if s0 < L - 1])
        xs.append(t + [0] * (L + 1 - len(t)))
    return torch.tensor(xs), spans


def text_batch(rng, B, names, nfill, L):
    xs, spans = [], []
    for _ in range(B):
        t, s0, s1 = dialogue(rng, names, nfill); t = list(t[:L + 1]); spans.append((s0, min(s1, L)))
        xs.append(t + [0] * (L + 1 - len(t)))
    return torch.tensor(xs), spans


def ce_all(logits, x): return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), ignore_index=0)


@torch.no_grad()
def name_ce(m, batches, arm):
    m.eval(); tot = n = 0.0; bg = nb = 0.0
    for x, spans in batches:
        lg = m(x, tau=0.05) if arm == "CKB" else m(x)
        lp = F.log_softmax(lg[:, :-1], -1); tgt = x[:, 1:]
        nll = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        for i, sp in enumerate(spans):
            sp = sp if isinstance(sp, list) else [sp]; msk = (tgt[i] != 0).clone()
            for (s0, s1) in sp:
                if s1 - 1 > s0: tot += nll[i, s0:s1 - 1].sum().item(); n += (s1 - 1 - s0)   # name bytes after the first
                msk[max(s0 - 1, 0):s1 - 1] = False
            bg += nll[i][msk].sum().item(); nb += msk.sum().item()
    m.train(); return round(tot / max(n, 1), 4), round(bg / max(nb, 1), 4)


def train_text(arm, d, seed, steps, B, L, log):
    torch.manual_seed(seed); rng = random.Random(seed)
    m = A.TFMicro(256, d) if arm == "TF" else GRUCore(256, d) if arm == "GRU" else CKB(256, d, MSLOTS)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time()
    for s in range(steps):
        x, _ = dense_batch(rng, B, None if RANDNAMES else NAMES_TR, 6, L) if DENSE else text_batch(rng, B, NAMES_TR, rng.randint(1, 3), L)
        if arm == "CKB":
            tau = 0.1 ** (s / max(steps - 1, 1)); gmin = GMIN0 * (1 - s / max(steps - 1, 1)); lg, bs, _ = m(x, tau=tau, gmin=gmin, states=True)
        else:
            lg = m(x); bs = None
        loss = ce_all(lg, x)
        if BMODE == "util" and arm == "CKB": loss = loss + UTILW * m.aux_loss
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 250 == 0 or s == steps - 1:
            log(f"  [{arm} d{d} s{seed}] step {s} ce {loss.item():.3f}" + (f" b {bs[x != 0].mean().item():.3f}" if bs is not None else "") + (f" R {m.aux_R:.3f}" if BMODE == "util" and arm == "CKB" else "") + f" ({time.time()-t0:.0f}s)")
    return m


@torch.no_grad()
def boundary_profile(m, x):
    """where does the learned boundary fire? fraction at space/newline/punct vs inside words"""
    _, bs, _ = m(x, tau=0.05, states=True); b = bs > 0.5
    toks = x; sp = (toks == 32) | (toks == 10) | (toks == 46) | (toks == 44) | (toks == 58) | (toks == 63)
    return round(b[sp].float().mean().item(), 3), round(b[~sp & (toks != 0)].float().mean().item(), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="text"); ap.add_argument("--arms", default="CKB")
    ap.add_argument("--ds", default="32"); ap.add_argument("--seeds", default="1"); ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--B", type=int, default=8); ap.add_argument("--bmode", default="soft"); ap.add_argument("--ccred", type=int, default=0); ap.add_argument("--dscale", type=int, default=0); ap.add_argument("--knorm", type=int, default=0); ap.add_argument("--lowmem", type=int, default=0); ap.add_argument("--inner", type=int, default=0); ap.add_argument("--look", type=int, default=0); ap.add_argument("--lookh", type=int, default=3); ap.add_argument("--imcons", type=float, default=0.0); ap.add_argument("--threads", type=int, default=1); ap.add_argument("--hcut", type=float, default=0.0); ap.add_argument("--surprise", type=int, default=0); ap.add_argument("--pkb", type=int, default=0); ap.add_argument("--wkb", type=int, default=0); ap.add_argument("--cgw", type=int, default=0); ap.add_argument("--dense", type=int, default=0); ap.add_argument("--role", type=int, default=0); ap.add_argument("--pure", type=int, default=0); ap.add_argument("--randnames", type=int, default=0); ap.add_argument("--tie", type=int, default=0); ap.add_argument("--save", default=""); ap.add_argument("--M", type=int, default=8); ap.add_argument("--L", type=int, default=320)
    a = ap.parse_args(); log = lambda *x: print(*x, flush=True)
    global CCRED, HCUT, DSCALE, KNORM; CCRED = a.ccred; HCUT = a.hcut; DSCALE = a.dscale; KNORM = a.knorm; global LOWMEM, INNER; LOWMEM = a.lowmem; INNER = a.inner; global LOOK, LOOKH; LOOK = a.look; LOOKH = a.lookh; global IMCONS; IMCONS = a.imcons; torch.set_num_threads(a.threads)
    global BMODE, SURPRISE, PKB, MSLOTS, WKB, CGW, DENSE, ROLE, PUREADDR, RANDNAMES, TIE; ROLE = a.role; DENSE = bool(a.dense); PUREADDR = a.pure; RANDNAMES = bool(a.randnames); TIE = bool(a.tie); MSLOTS = a.M; WKB = a.wkb; CGW = bool(a.cgw); BMODE = a.bmode; SURPRISE = bool(a.surprise); PKB = bool(a.pkb)
    res = {"tag": "ARCH-VET-LM-P49", "ccred": a.ccred, "dscale": a.dscale, "knorm": a.knorm, "lowmem": a.lowmem, "inner": a.inner, "hcut": a.hcut, "bmode": a.bmode, "surprise": a.surprise, "pkb": a.pkb, "wkb": a.wkb, "cgw": a.cgw, "dense": a.dense, "role": a.role, "pure": a.pure, "randnames": a.randnames, "tie": a.tie, "M": a.M, "task": a.task, "protocol": __doc__[:1500], "runs": []}
    if a.task == "synthetic":
        er = random.Random(999); ev_in = [T.make_batch(er, 16, (2, 4), 2, 4, False) for _ in range(12)]
        er = random.Random(998); ev_len = [T.make_batch(er, 16, (2, 4), 2, 30, False) for _ in range(12)]
        for d in map(int, a.ds.split(",")):
            for seed in map(int, a.seeds.split(",")):
                torch.manual_seed(seed); rng = random.Random(seed); m = CKB(T.V, d, 8); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
                for s in range(a.steps):
                    tau = 0.1 ** (s / max(a.steps - 1, 1)); gmin = GMIN0 * (1 - s / max(a.steps - 1, 1))
                    bt = T.make_batch(rng, 16, (2, 4), 2, 4, False); lg, bs, _ = m(bt["x"], tau=tau, gmin=gmin, states=True)
                    loss = T.ce_loss(lg, bt["x"], bt["ans"]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                    if s % 500 == 0: log(f"  [CKB-syn d{d} s{seed}] step {s} loss {loss.item():.3f} b {bs.mean().item():.3f}")
                def acc(bat):
                    m.eval(); c = t = 0
                    with torch.no_grad():
                        for bt in bat:
                            p = m(bt["x"], tau=0.05).argmax(-1)
                            for i, poss in enumerate(bt["ans"]):
                                for q in poss: c += int(p[i, q - 1].item() == bt["x"][i, q].item()); t += 1
                    m.train(); return c / t
                r = {"arm": "CKB-syn", "d": d, "seed": seed, "params": n_params(m), "in": acc(ev_in), "len": acc(ev_len)}
                res["runs"].append(r); log(f"[p49 syn d{d} s{seed}] params {r['params']} in {r['in']:.3f} len {r['len']:.3f}")
    else:
        er = random.Random(999); ev_in = [text_batch(er, 8, NAMES_TR, 2, a.L) for _ in range(8)]
        er = random.Random(998); ev_ood = [text_batch(er, 8, NAMES_OOD, 2, a.L) for _ in range(8)]
        er = random.Random(997); ev_far = [text_batch(er, 8, NAMES_OOD, 10, 4 * a.L) for _ in range(6)]
        er = random.Random(996); ev_dense = [dense_batch(er, 8, NAMES_OOD, 6, a.L) for _ in range(8)]
        for arm in a.arms.split(","):
            for d in map(int, a.ds.split(",")):
                for seed in map(int, a.seeds.split(",")):
                    m = train_text(arm, d, seed, a.steps, a.B, a.L, log)
                    r = {"arm": arm, "d": d, "seed": seed, "params": n_params(m)}
                    for nm, ev in [("in", ev_in), ("ood_names", ev_ood), ("far_ood", ev_far), ("dense_ood", ev_dense)]:
                        r[nm + "_name_ce"], r[nm + "_bg_ce"] = name_ce(m, ev, arm)
                    if arm == "CKB": r["b_at_sep"], r["b_in_word"] = boundary_profile(m, ev_in[0][0])
                    if arm == "CKB" and CGW:
                        with torch.no_grad():
                            sp = torch.sigmoid(m.ws(m.E(torch.tensor([32, 10, 46, 97, 101, 112])))).squeeze(-1).tolist()
                        r["pass_sp_nl_dot_a_e_p"] = [round(v, 2) for v in sp]; log(f"   pass factors space/nl/./a/e/p {r['pass_sp_nl_dot_a_e_p']}")
                    if a.save: torch.save(m.state_dict(), f"p21_ckpt/{a.save}_{arm}_d{d}_s{seed}.pt")
                    res["runs"].append(r); log(f"[p49 {arm} d{d} s{seed}] params {r['params']} name-CE in {r['in_name_ce']} ood {r['ood_names_name_ce']} far {r['far_ood_name_ce']} DENSE-unseen {r['dense_ood_name_ce']} | bg {r['in_bg_ce']} {r['ood_names_bg_ce']} {r['far_ood_bg_ce']}" + (f" | b sep/word {r['b_at_sep']}/{r['b_in_word']}" if arm == "CKB" else ""))
    with open("log.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
    log("[P49] DONE")


if __name__ == "__main__":
    main()
