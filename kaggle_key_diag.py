"""kaggle_key_diag.py — DIAGNOSTIC (instrument-first): is the fact identity even present in the frozen base's hidden state at
the positions our memory keys on?  No training; ~2 minutes on a T4.

For N conversations (same generator as kaggle_mem_test.py): take the hidden state at the LAST token of each fact statement's
prefix ("...my dog's name is" -> position of " is") and at the last token of the matching question's answer prefix
("Your dog's name is" -> " is").  Exact nearest-neighbour: for each question vector, is the closest statement vector (cosine)
the one for the SAME fact, among all facts of the conversation?  Reported for several candidate key features and layers:
  raw last layer | LayerNorm | causal EMA (a=.75) | EMA(a=.9) | mean of last 8 tokens | mid-layer variants | concat with token emb
and for the two hardest cases: facts of the same kind (e.g. 4 'name' facts) only.
If NN top-1 is high (>80 %), the information is there and the store's addressing/learning is what fails.  If it is near chance
(1/nfacts), no key function of these states can separate the facts and the add-on must read elsewhere.
usage: BASE=HuggingFaceTB/SmolLM2-135M python TryArena/kaggle_key_diag.py
"""
import os, sys, random, re, math
BASE = os.environ.get("BASE", "HuggingFaceTB/SmolLM2-135M"); N = int(os.environ.get("N", 30)); NF = int(os.environ.get("NF", 16)); T = int(os.environ.get("T", 2048))
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")
sys.path.insert(0, f"{WORK}/TryArena"); import kaggle_mem_test as K

def main():
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dt = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE); base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dt).to(dev).eval()
    E = base.get_input_embeddings().weight.float(); sents = K.load_filler(); r = random.Random(11)
    nl = base.config.num_hidden_layers; layers = sorted(set([nl, nl // 2, (3 * nl) // 4]))
    feats = {}   # name -> list of (stmt_vecs [nf,d], q_vecs [nf,d], kinds [nf])
    def ema(h, a):
        out = torch.zeros_like(h); e = torch.zeros(h.shape[-1], device=h.device)
        for t in range(h.shape[0]): e = a * e + (1 - a) * h[t]; out[t] = e
        return out
    def add(name, s, q, kinds): feats.setdefault(name, []).append((s, q, kinds))
    for i in range(N):
        ids, spans, facts = K.conversation(r, tok, NF, T, sents)
        text = tok.decode(ids); enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); offs = enc["offset_mapping"]
        x = torch.tensor([enc["input_ids"]], device=dev)   # re-encoded text: offsets and ids are consistent by construction
        with torch.no_grad(): out = base(input_ids=x, output_hidden_states=True)
        def tok_before(char_pos):   # index of last token ending at or before char_pos
            j = max(j for j, (a, b) in enumerate(offs) if b <= char_pos); return j
        spos, qpos, kinds, ok = [], [], [], True
        for fi, f in enumerate(facts):
            st = text.find(f"By the way, {f[3]}."); qa = text.find(f"{f[5]} {f[2]}")
            if st < 0 or qa < 0: ok = False; break
            spos.append(tok_before(st + len("By the way, ") + len(f[3]) - len(f[2]) - 1)); qpos.append(tok_before(qa + len(f[5]))); kinds.append(f[0])
        if not ok or min(spos) < 0: continue
        spos, qpos = torch.tensor(spos, device=dev), torch.tensor(qpos, device=dev); kinds = torch.tensor(kinds)
        # surprise of each token under the base (-log p(x_t | <t)); first token gets the mean
        lp = F.log_softmax(out.logits[0].float(), -1); surp = torch.cat([torch.tensor([3.0], device=dev), -lp[:-1].gather(-1, x[0, 1:, None]).squeeze(-1)])
        def swin(v, w, win, pw=1.0):   # surprise-weighted mean over the last `win` tokens (weights = surprise^pw)
            ww = surp.clamp(min=0) ** pw; out_ = torch.zeros_like(v)
            for p in range(v.shape[0]):
                a = max(0, p - win + 1); wt = ww[a:p + 1]; out_[p] = (v[a:p + 1] * wt[:, None]).sum(0) / (wt.sum() + 1e-6)
            return out_
        exn = F.layer_norm(E[x[0]], (E.shape[1],))
        for win in (6, 10):
            sv = swin(exn, None, win); add(f"tokemb surp{win}", sv[spos], sv[qpos], kinds)
            sv2 = swin(exn, None, win, pw=2.0); add(f"tokemb surp^2 w{win}", sv2[spos], sv2[qpos], kinds)
        for L in layers:
            h = out.hidden_states[L][0].float(); tag = f"L{L}"
            hn_ = F.layer_norm(h, (h.shape[-1],))
            for win in (6, 10):
                sv = swin(hn_, None, win); add(f"{tag} surp{win}", sv[spos], sv[qpos], kinds)
            sv = swin(hn_, None, 8, pw=2.0); add(f"{tag} surp^2 w8", sv[spos], sv[qpos], kinds)
            add(f"{tag} raw", h[spos], h[qpos], kinds); hn = F.layer_norm(h, (h.shape[-1],)); add(f"{tag} layernorm", hn[spos], hn[qpos], kinds)
            e75 = ema(hn, 0.75); add(f"{tag} ema.75", e75[spos], e75[qpos], kinds); e9 = ema(hn, 0.9); add(f"{tag} ema.90", e9[spos], e9[qpos], kinds)
            m8 = torch.stack([hn[max(0, p - 7):p + 1].mean(0) for p in range(hn.shape[0])]); add(f"{tag} mean8", m8[spos], m8[qpos], kinds)
            ex = F.layer_norm(E[x[0]], (E.shape[1],)); cat = torch.cat([hn, ex], -1); add(f"{tag} ln+tokemb", cat[spos], cat[qpos], kinds)
            cat2 = torch.cat([e75, hn], -1); add(f"{tag} ema.75+ln", cat2[spos], cat2[qpos], kinds)
    print(f"[DIAG {BASE}] conversations used {len(next(iter(feats.values())))} nfacts {NF}")
    for name, lst in feats.items():
        top1 = top1_same = n = n_same = 0; margin = []
        for s, q, kinds in lst:
            sim = F.normalize(q, dim=-1) @ F.normalize(s, dim=-1).t()   # [nf_q, nf_s]
            pred = sim.argmax(1); top1 += (pred == torch.arange(len(s), device=dev)).sum().item(); n += len(s)
            for i in range(len(s)):   # same-kind only
                mask = (kinds == kinds[i]).to(dev); 
                if mask.sum() < 2: continue
                sub = sim[i].masked_fill(~mask, -2); top1_same += int(sub.argmax().item() == i); n_same += 1
                others = sim[i][mask & (torch.arange(len(s), device=dev) != i)]; margin.append((sim[i, i] - others.max()).item())
        print(f"  {name:22s} NN top-1 all-facts {100*top1/max(1,n):5.1f} %  | same-kind only {100*top1_same/max(1,n_same):5.1f} %  (chance {100/NF:.0f} % / 25 %)  mean margin same-kind {sum(margin)/max(1,len(margin)):+.3f}")

if __name__ == "__main__": main()


# ---------------------------------------------------------------- LEARNED KEY (contrastive) — the decisive test for the add-on form
def learned_key(train_n=int(os.environ.get("KTRAIN", 160)), test_n=30, win=8, steps=600):
    """Train f(window of mid-layer LN states) -> key with InfoNCE on (question, statement) pairs of the same fact; evaluate exact
    NN top-1 on held-out conversations.  Negatives = all other statements in the batch (same-kind facts of the same conversation included)."""
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dt = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE); base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dt).to(dev).eval()
    E = base.get_input_embeddings().weight.float(); sents = K.load_filler(); nl = base.config.num_hidden_layers; L = nl // 2
    def extract(seed, n):
        r = random.Random(seed); S, Q, KD, CID = [], [], [], []
        for i in range(n):
            ids, spans, facts = K.conversation(r, tok, NF, T, sents); text = tok.decode(ids)
            enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); offs = enc["offset_mapping"]; x = torch.tensor([enc["input_ids"]], device=dev)
            with torch.no_grad(): hs = base(input_ids=x, output_hidden_states=True).hidden_states
            h = F.layer_norm(hs[L][0].float(), (hs[L].shape[-1],)); ex = F.layer_norm(E[x[0]], (E.shape[1],))
            def tok_before(c): return max(j for j, (a, b) in enumerate(offs) if b <= c)
            def window(p): 
                a = max(0, p - win + 1); v = torch.cat([h[a:p + 1], ex[a:p + 1]], -1)
                return F.pad(v, (0, 0, win - v.shape[0], 0)).reshape(-1)
            ok = True; cs, cq, ck = [], [], []
            for f in facts:
                st = text.find(f"By the way, {f[3]}."); qa = text.find(f"{f[5]} {f[2]}")
                if st < 0 or qa < 0: ok = False; break
                cs.append(window(tok_before(st + len("By the way, ") + len(f[3]) - len(f[2]) - 1))); cq.append(window(tok_before(qa + len(f[5])))); ck.append(f[0])
            if not ok: continue
            S += cs; Q += cq; KD += ck; CID += [i] * len(cs)
        return torch.stack(S), torch.stack(Q), torch.tensor(KD), torch.tensor(CID)
    S, Q, KD, CID = extract(100, train_n); St, Qt, KDt, CIDt = extract(900, test_n)
    din = S.shape[1]; key = nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.Linear(512, 256)).to(dev); opt = torch.optim.Adam(key.parameters(), lr=1e-3)
    print(f"[LEARNED-KEY {BASE}] layer {L} window {win} train pairs {len(S)} test pairs {len(St)} input dim {din}")
    def evalnn(S_, Q_, KD_, CID_):
        with torch.no_grad():
            ks, kq = F.normalize(key(S_), dim=-1), F.normalize(key(Q_), dim=-1); top = topk = n = 0
            for c in CID_.unique():
                m = (CID_ == c).nonzero().squeeze(-1); sim = kq[m] @ ks[m].t(); pred = sim.argmax(1); top += (pred == torch.arange(len(m), device=dev)).sum().item(); n += len(m)
                kd = KD_[m]
                for i in range(len(m)):
                    mask = (kd == kd[i]).to(dev); sub = sim[i].masked_fill(~mask, -2); topk += int(sub.argmax().item() == i)
            return 100 * top / n, 100 * topk / n
    for s in range(steps):
        idx = torch.randint(0, len(S), (256,)); ks, kq = F.normalize(key(S[idx]), dim=-1), F.normalize(key(Q[idx]), dim=-1)
        logits = kq @ ks.t() / 0.07; loss = F.cross_entropy(logits, torch.arange(len(idx), device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0 or s == steps - 1:
            a, b = evalnn(St, Qt, KDt, CIDt); print(f"  step {s} infoNCE {loss.item():.3f} | HELD-OUT NN top-1 all-facts {a:.1f} %  same-kind {b:.1f} %")
    a, b = evalnn(S, Q, KD, CID); print(f"  (train-set NN top-1 all {a:.1f} % same-kind {b:.1f} %)")

if __name__ == "__main__" and os.environ.get("LEARNED", "0") == "1": learned_key()
