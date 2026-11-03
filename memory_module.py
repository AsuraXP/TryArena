"""memory_module.py — token-level ADD-ON memory for a frozen pretrained Transformer (our core, re-cut as a component).

Goal (STANDING_PROMPT, goal update 2026-11): a small pretrained model that, given what it was told, never distorts
or forgets it, at constant memory over conversation length.  The base model is frozen; only this module trains.

Mechanism (each piece has a measured result behind it in log.md; nothing new is introduced here):
  * WRITE, novelty-gated (F1 "sgate"): at position t the association (key_{t-1} -> value_t) is written with strength
    beta_t = sigmoid((surprise_t - theta) / s), surprise_t = -log p_base(x_t | x_<t).  Tokens the base predicted
    easily are not written; names, numbers, paths, decisions are.  Cleaner store, fewer writes.
  * STORE, O(1) state (M2/M3/P-series): sparse k-WTA keys (dk >> d, top-k fraction `sparse`) from the base's hidden
    state; Hebbian outer-product fast weights W (dk x d), no decay; one or two key resolutions (`mres`: the current
    hidden state and a short EMA of past hidden states).  Chunked causal linear-attention form for training.
  * READ, tied copy readout with confidence gate (P-series "tied 2" + "gconf"): query = current hidden state's sparse
    key; v_hat = q W; copy logits = E v_hat (E = the base's own input embedding matrix, frozen -> the readout can only
    propose real tokens; it cannot invent); confidence g from (match mass, |v_hat|, gate MLP on hidden).  Output
    log p = log((1-g) p_base + g softmax(copy / tau)).  With g -> 0 the base is untouched (no-harm by construction).
  * SEGMENTED streaming: the base sees only a window of `seg` tokens (its KV cost is bounded); the module state
    (W, EMA) carries across segments.  That is what "constant memory" means for the combined system.

Related work (searched; cite, do not claim): fast-weight programmers (Schmidhuber 1992; Schlag et al. 2021),
DeltaNet (Yang et al. 2024), Titans surprise-gated memory (Behrouz et al. 2024), pointer-generator copy readout
(See et al. 2017), sparse distributed memory (Kanerva 1988), fast-weight product-key memory (FwPKM 2026).  Ours is the
combination as a frozen-base add-on with the readout tied to the base embedding and a novelty-gated write; the
evidence base is the byte-level ledger in log.md (S1/F1/P7).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class TokenMemory(nn.Module):
    def __init__(self, d, E, dk=4096, sparse=0.02, mres=1, chunk=128, tau=1.0, ckey=1):
        """d: base hidden size; E: frozen base input-embedding weight (V x d_emb) -- d_emb must equal d (tied readout);
        dk: store address width; sparse: k-WTA fraction; mres: 1 or 2 key resolutions."""
        super().__init__(); self.d, self.dk, self.k, self.mres, self.chunk, self.ckey = d, dk, max(1, int(sparse * dk)), mres, chunk, ckey
        din = d * (1 + ckey)   # ckey (P7): the key also sees the raw embedding of the current token, so mid-name tokens address by identity
        self.register_buffer("E", E.detach().float(), persistent=False)   # tied copy readout: proposals are real tokens only
        self.hn = nn.LayerNorm(d)   # real LLM hidden states carry a few massive-activation dims; without normalisation every sparse key is
                                    # dominated by the same dims and the store blurs (suspected cause of the identical mem==window runs 1-2)
        self.kp = nn.ModuleList([nn.Linear(din, dk, bias=False) for _ in range(mres)])
        for p in self.kp: nn.init.orthogonal_(p.weight)                                   # write key == read key (same code; content addressing)
        self.vp = nn.Linear(d, d, bias=False); nn.init.eye_(self.vp.weight)                 # value = (near-)identity of the embedding
        self.theta = nn.Parameter(torch.tensor(2.0)); self.ssc = nn.Parameter(torch.tensor(1.0))   # novelty gate: surprise threshold / scale
        self.gate = nn.Sequential(nn.LayerNorm(d + 3 * mres), nn.Linear(d + 3 * mres, 64), nn.GELU(), nn.Linear(64, 1)); nn.init.constant_(self.gate[-1].bias, -2.0)
        # gate inputs are normalised (LayerNorm; match mass enters as log1p): the 135M run showed a raw match mass in the hundreds
        # saturating the gate at 0 from step 0 -> memory never consulted (null run, logged).
        self.ltau = nn.Parameter(torch.tensor(math.log(tau))); self.ema_a = 0.75
        self.aux = torch.zeros(()); self.last = {}

    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_state(self, B, device):
        return {"W": [torch.zeros(B, self.dk, self.d, device=device) for _ in range(self.mres)], "ema": torch.zeros(B, self.d, device=device),
                "n": torch.zeros(B, device=device)}   # ema sized lazily

    def _code(self, x, proj):
        z = proj(x); top = z.topk(self.k, -1); out = torch.zeros_like(z).scatter(-1, top.indices, F.relu(top.values) + 1e-6)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-6)

    def _keys(self, h, state):
        """keys for each resolution from hidden states h (B,L,d); EMA resolution uses a causal EMA carried across segments."""
        hs = [h]
        if self.mres > 1:
            B, L, d = h.shape; a = self.ema_a; e = state["ema"] if state["ema"].shape[-1] == d else torch.zeros(B, d, device=h.device); outs = []
            for t in range(L):   # short python loop over L (L = segment length <= 2048); cheap relative to the base forward
                e = a * e + (1 - a) * h[:, t]; outs.append(e)
            hs.append(torch.stack(outs, 1)); state["ema"] = e.detach()
        return hs

    def forward(self, h, base_logp, x, state, targets=None, gfloor=0.0, force_g=None):
        """force_g: if set (e.g. 1.0), the gate is overridden -- diagnostic 'copy' arm (pure retrieval).
        h: (B,L,d) base hidden states for tokens x (B,L); base_logp: (B,L,V) base log-probs for the NEXT token;
        x: token ids (B,L).  Returns mixed log-probs (B,L,V) and the updated state.  Causal: the prediction at t uses writes
        up to (key_{t-1} -> value_t) i.e. only tokens <= t."""
        B, L, d = h.shape; dev = h.device; h = self.hn(h)
        # surprise of each observed token under the base (needs base log-prob of x_t from position t-1; first token of a
        # segment uses the carried value if available -> approximate with the mean surprise)
        with torch.no_grad():
            lp_prev = base_logp[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1)             # log p(x_t | <t) for t>=1
            surpr = torch.cat([torch.full((B, 1), 2.0, device=dev), -lp_prev], 1)         # (B,L)
        beta = torch.sigmoid((surpr - self.theta) / F.softplus(self.ssc))               # write strength per token (B,L)
        v = self.vp(self.E[x])                                                            # value: embedding of the observed token (B,L,d)
        ex = self.E[x]; hin0 = torch.cat([h, ex], -1) if self.ckey else h
        hs = self._keys(hin0, state); outs = []; matches = []                            # resolution 2 = causal EMA of (hidden, token emb): base-independent context
        for r in range(self.mres):
            K = self._code(hs[r], self.kp[r]); Q = K                                        # (B,L,dk) sparse codes, shared write/read key
            # write (K_{t-1} -> v_t) with beta_t ; read at t with Q_t over everything written up to t (inclusive of the write at t)
            Kw = torch.cat([torch.zeros(B, 1, self.dk, device=dev), K[:, :-1]], 1) * beta[..., None]   # key shifted: association from previous key
            W = state["W"][r]; o = []; m = []
            for c0 in range(0, L, self.chunk):
                c1 = min(L, c0 + self.chunk); q, kw, vv = Q[:, c0:c1], Kw[:, c0:c1], v[:, c0:c1]
                S = torch.tril(torch.einsum("bik,bjk->bij", q, kw))                       # within-chunk causal (inclusive)
                o.append(torch.einsum("bij,bjd->bid", S, vv) + torch.einsum("bik,bkd->bid", q, W))
                m.append(S.sum(-1) + torch.einsum("bik,bk->bi", q, W.abs().sum(-1)))       # match mass proxy
                W = W + torch.einsum("bjk,bjd->bkd", kw, vv)
            state["W"][r] = W.detach(); outs.append(torch.cat(o, 1)); matches.append(torch.cat(m, 1))
        vhat = sum(outs) / self.mres                                                     # (B,L,d)
        copy_logits = (vhat @ self.E.t()) / vhat.norm(dim=-1, keepdim=True).clamp(min=1e-3) * torch.exp(self.ltau)   # cosine-ish tied readout
        feats = torch.cat([h] + [torch.log1p(mm.clamp(min=0))[..., None] for mm in matches] + [(mm > 0.1).float()[..., None] for mm in matches]
                          + [F.cosine_similarity(o, vhat, dim=-1)[..., None] for o in outs], -1)   # agreement between key resolutions
        g = torch.sigmoid(self.gate(feats)) * (1 - gfloor) + gfloor                        # (B,L,1) confidence of the copy path (gfloor>0 only in training: keeps gradient alive)
        if force_g is not None: g = torch.full_like(g, force_g)
        copy_lp = F.log_softmax(copy_logits, -1)
        logp = torch.logaddexp(torch.log1p(-g + 1e-6) + base_logp, torch.log(g + 1e-6) + copy_lp)   # (B,L,V), no stack copy
        # auxiliary: copy-path NLL of the next token (B,L-1), so the store/readout learn WHAT to retrieve independently of the gate
        self.aux_nll = -copy_lp[:, :-1].gather(-1, targets[:, :, None]).squeeze(-1) if targets is not None else None
        state["n"] = state["n"] + beta.sum(1).detach()
        self.last = {"g": g.detach().mean().item(), "beta": beta.detach().mean().item(), "graw": torch.sigmoid(self.gate(feats)).detach().mean().item(),
                     "wfrac": (beta > 0.5).float().mean().item()}
        return logp, state


class LearnedKeyMemory(nn.Module):
    """Same store and readout as TokenMemory, with the KEY LEARNED (key diagnostic 2026-11: fixed features top out at 64 % NN top-1
    on same-kind facts; a contrastively trained key on an 8-token window of mid-layer states reaches 99-100 % held-out).
      key_t  = normalize(MLP([LN(h_mid[t-7..t]), LN(E[x_{t-7..t}])]))        (dkey-dim, dense, unit norm)
      write  : W += beta_t * key_{t-1} (x) v_t          beta_t = novelty gate (base surprise of x_t)
      read   : vhat_t = key_t W ;  copy logits = E vhat (tied readout, real tokens only) ; gate g from (LN h, log1p match, |vhat|)
    Training signals (all in the main run): LM loss on the mixture, copy-path NLL on answer tokens, and InfoNCE between the key at the
    question's pre-answer position and the key at the same fact's statement pre-answer position (in-batch negatives)."""
    def __init__(self, d, E, dkey=256, win=8, hid=512, chunk=128, tau=1.0, dk=4096, sparse=0.015, theta0=6.0):
        """Run-4 lesson (arithmetic, logged): dense 256-d keys + writing 76 % of tokens = ~1500 writes/conversation, noise ~2.4x signal.
        So: (1) store address = k-WTA sparse code (dk, top `sparse`) of the learned key -> unrelated addresses overlap ~0;
        (2) write only unpredictable tokens: theta0 = 6 nats (p < .25 %); random names/numbers are 8+, prose mostly below."""
        super().__init__(); self.d, self.dkey, self.win, self.chunk, self.dk, self.k = d, dkey, win, chunk, dk, max(1, int(sparse * dk))
        self.register_buffer("E", E.detach().float(), persistent=False)
        self.hn = nn.LayerNorm(d); self.en = nn.LayerNorm(d)
        self.key = nn.Sequential(nn.Linear(2 * d * win, hid), nn.GELU(), nn.Linear(hid, dkey))
        self.vp = nn.Linear(d, d, bias=False); nn.init.eye_(self.vp.weight)
        self.theta = nn.Parameter(torch.tensor(theta0)); self.ssc = nn.Parameter(torch.tensor(0.5))
        self.addr = nn.Linear(dkey, dk, bias=False); nn.init.orthogonal_(self.addr.weight)     # learned key -> sparse address (k-WTA)
        # LEARNED write gate (run 5 finding: surprise cannot select facts — random prose filler is as surprising as names, digits are cheap).
        # Trained with an explicit target (statement answer tokens) because truncated BPTT across segments gives writes no retrieval gradient.
        self.wgate = nn.Sequential(nn.Linear(2 * d * win + 1, 128), nn.GELU(), nn.Linear(128, 1)); nn.init.constant_(self.wgate[-1].bias, -3.0)
        self.gate = nn.Sequential(nn.LayerNorm(d + 2), nn.Linear(d + 2, 64), nn.GELU(), nn.Linear(64, 1)); nn.init.constant_(self.gate[-1].bias, -2.0)
        self.ltau = nn.Parameter(torch.tensor(math.log(tau))); self.last = {}; self.aux_nll = None; self.keys = None

    def n_params(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
    def init_state(self, B, device):
        return {"W": torch.zeros(B, self.dk, self.d, device=device), "hist": None}

    def _windows(self, h, ex, state):
        """(B,L,2d) per-token features -> (B,L,win*2d) causal windows, carrying the last win-1 tokens across segments."""
        f = torch.cat([self.hn(h), self.en(ex)], -1); B, L, D = f.shape
        prev = state["hist"] if state["hist"] is not None else torch.zeros(B, self.win - 1, D, device=f.device)
        full = torch.cat([prev, f], 1); state["hist"] = full[:, -(self.win - 1):].detach()
        return torch.cat([full[:, i:i + L] for i in range(self.win)], -1)        # window ends at t: tokens t-win+1 .. t

    def forward(self, h, base_logp, x, state, targets=None, gfloor=0.0, force_g=None):
        B, L, d = h.shape; dev = h.device; ex = self.E[x]
        with torch.no_grad():
            lp_prev = base_logp[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1); surpr = torch.cat([torch.full((B, 1), 2.0, device=dev), -lp_prev], 1)
        wf = self._windows(h, ex, state); self.win_feats = wf.detach()                             # exposed: the driver re-keys pair positions for the contrastive loss
        self.wlogit = self.wgate(torch.cat([wf, ((surpr - self.theta) / F.softplus(self.ssc))[..., None]], -1)).squeeze(-1)   # (B,L); surprise is one input
        beta = torch.sigmoid(self.wlogit)
        Kd = F.normalize(self.key(wf), dim=-1); self.keys = Kd                                     # (B,L,dkey) unit dense keys (contrastive target)
        z = self.addr(Kd); top = z.topk(self.k, -1); Kf = torch.zeros_like(z).scatter(-1, top.indices, F.relu(top.values) + 1e-6)
        Kf = Kf / (Kf.norm(dim=-1, keepdim=True) + 1e-6)                                            # (B,L,dk) sparse unit address
        v = self.vp(ex); Kw = torch.cat([torch.zeros(B, 1, self.dk, device=dev), Kf[:, :-1]], 1) * beta[..., None]
        W = state["W"]; outs = []; ms = []
        for c0 in range(0, L, self.chunk):
            c1 = min(L, c0 + self.chunk); q, kw, vv = Kf[:, c0:c1], Kw[:, c0:c1], v[:, c0:c1]
            S = torch.tril(torch.einsum("bik,bjk->bij", q, kw)); outs.append(torch.einsum("bij,bjd->bid", S, vv) + torch.einsum("bik,bkd->bid", q, W))
            ms.append(S.abs().sum(-1) + torch.einsum("bik,bk->bi", q.abs(), W.abs().sum(-1))); W = W + torch.einsum("bjk,bjd->bkd", kw, vv)
        state["W"] = W.detach(); vhat = torch.cat(outs, 1); match = torch.cat(ms, 1)
        copy_logits = (vhat @ self.E.t()) / vhat.norm(dim=-1, keepdim=True).clamp(min=1e-3) * torch.exp(self.ltau)
        feats = torch.cat([self.hn(h), torch.log1p(match.clamp(min=0))[..., None], torch.log1p(vhat.norm(dim=-1))[..., None]], -1)
        g = torch.sigmoid(self.gate(feats)) * (1 - gfloor) + gfloor
        if force_g is not None: g = torch.full_like(g, force_g)
        copy_lp = F.log_softmax(copy_logits, -1)
        logp = torch.logaddexp(torch.log1p(-g + 1e-6) + base_logp, torch.log(g + 1e-6) + copy_lp)
        self.aux_nll = -copy_lp[:, :-1].gather(-1, targets[:, :, None]).squeeze(-1) if targets is not None else None
        self.last = {"g": g.detach().mean().item(), "beta": beta.detach().mean().item(), "graw": torch.sigmoid(self.gate(feats)).detach().mean().item(),
                     "wfrac": (beta > 0.5).float().mean().item()}
        return logp, state
