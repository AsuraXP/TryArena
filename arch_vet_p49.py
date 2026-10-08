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
Everything learned; loss = CE only; no position signal; slots are state (grown at eval).
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
DENSE = False   # C102: train on dense-reuse dialogues
WKB = 0   # C100: key = learned per-tap mix of the last WKB byte embeddings (no boundary, no gate)
MSLOTS = 8   # C99: key = prefix state c_t, value = next byte embedding, written every step   # C98: boundary also sees the learner's own surprise -log p(x_t) and entropy H_{t-1}   # soft | st (straight-through hard boundary) | force (b=1 every token: wiring control)


class CKB(nn.Module):
    def __init__(self, V, d, M=8):
        super().__init__(); self.V, self.d, self.M = V, d, M
        self.E = nn.Embedding(V, d); self.gh = nn.GRUCell(d, d); self.gc = nn.GRUCell(d, d)
        self.b = nn.Linear(2 * d + 2, 1); nn.init.constant_(self.b.bias, BINIT)
        self.theta = nn.Parameter(torch.tensor(1.0)); self.K0 = nn.Parameter(torch.randn(max(32, M), d) * 0.5)
        self.head = nn.Linear(2 * d + 1, V)
        self.taps = nn.Parameter(torch.full((max(WKB, 1), d), 0.5)); self.ws = nn.Linear(d, 1); nn.init.constant_(self.ws.bias, 2.2)

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
        K = self.K0[:M].unsqueeze(0).expand(B, M, d).clone(); Vm = torch.zeros(B, M, d); occ = torch.zeros(B, M)
        outs, bs = [], []; surp = torch.zeros(B, 2)
        for t in range(L):
            h = self.gh(e[:, t], h); c = self.gc(e[:, t], c)
            b = gmin + (1 - gmin) * torch.sigmoid(self.b(torch.cat([h, c, surp if SURPRISE else torch.zeros(B, 2)], -1)))
            if BMODE == "st": b = (b > 0.5).float() + b - b.detach()
            elif BMODE == "force": b = torch.ones_like(b)
            elif BMODE == "oracle":   # CONTROL ONLY: unit given = separator bytes (space, newline, , . : ? !)
                xt = x[:, t]; b = ((xt == 32) | (xt == 10) | (xt == 44) | (xt == 46) | (xt == 58) | (xt == 63) | (xt == 33)).float().unsqueeze(-1)
            bs.append(b)
            logocc = torch.log(occ + 1e-4)
            # read with the chunk so far (at a boundary: the whole chunk)
            if WKB:
                qv = self.wkey(e, t, B, d)
            else: qv = c
            sim = torch.einsum("bd,bmd->bm", qv, K) / math.sqrt(d); a = torch.softmax((sim + logocc) / tau, -1)
            r = torch.einsum("bm,bmd->bd", a, Vm); cf = (a * occ).sum(-1, keepdim=True)
            if PKB or WKB: rl, conf = r, cf
            else: rl = (1 - b) * rl + b * r; conf = (1 - b) * conf + b * cf
            lg = self.head(torch.cat([h, rl, conf], -1)); outs.append(lg)
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
                wsim = torch.einsum("bd,bmd->bm", pk, K) / math.sqrt(d); m = torch.softmax((wsim + logocc) / tau, -1)
                p = torch.sigmoid(wsim.max(-1, keepdim=True).values - self.theta); empty = torch.softmax(-occ / tau, -1)
                alpha = (p * m + (1 - p) * empty).unsqueeze(-1)
                K = (1 - alpha) * K + alpha * pk.unsqueeze(1); Vm = (1 - alpha) * Vm + alpha * e[:, t].unsqueeze(1); occ = occ + alpha.squeeze(-1) * (1 - occ)
                pk = c; c = c * (1 - b); continue
            # write (previous chunk -> this chunk) at the boundary
            wsim = torch.einsum("bd,bmd->bm", pk, K) / math.sqrt(d); m = torch.softmax((wsim + logocc) / tau, -1)
            p = torch.sigmoid(wsim.max(-1, keepdim=True).values - self.theta); empty = torch.softmax(-occ / tau, -1)
            alpha = (b * (p * m + (1 - p) * empty)).unsqueeze(-1)
            K = (1 - alpha) * K + alpha * pk.unsqueeze(1); Vm = (1 - alpha) * Vm + alpha * c.unsqueeze(1); occ = occ + alpha.squeeze(-1) * (1 - occ)
            pk = (1 - b) * pk + b * c; c = c * (1 - b)
        logits = torch.stack(outs, 1); bs = torch.cat(bs, 1)
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


def dialogue_dense(rng, names, nlines):
    """3 names recur on nearly every line; returns bytes and spans of every name occurrence after the first per name."""
    ns = rng.sample(names, 3); seen = set(); lines = []; spans = []; pos = 0
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
        x, _ = dense_batch(rng, B, NAMES_TR, 6, L) if DENSE else text_batch(rng, B, NAMES_TR, rng.randint(1, 3), L)
        if arm == "CKB":
            tau = 0.1 ** (s / max(steps - 1, 1)); gmin = GMIN0 * (1 - s / max(steps - 1, 1)); lg, bs, _ = m(x, tau=tau, gmin=gmin, states=True)
        else:
            lg = m(x); bs = None
        loss = ce_all(lg, x); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 250 == 0 or s == steps - 1:
            log(f"  [{arm} d{d} s{seed}] step {s} ce {loss.item():.3f}" + (f" b {bs[x != 0].mean().item():.3f}" if bs is not None else "") + f" ({time.time()-t0:.0f}s)")
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
    ap.add_argument("--B", type=int, default=8); ap.add_argument("--bmode", default="soft"); ap.add_argument("--surprise", type=int, default=0); ap.add_argument("--pkb", type=int, default=0); ap.add_argument("--wkb", type=int, default=0); ap.add_argument("--cgw", type=int, default=0); ap.add_argument("--dense", type=int, default=0); ap.add_argument("--M", type=int, default=8); ap.add_argument("--L", type=int, default=320)
    a = ap.parse_args(); log = lambda *x: print(*x, flush=True)
    global BMODE, SURPRISE, PKB, MSLOTS, WKB, CGW, DENSE; DENSE = bool(a.dense); MSLOTS = a.M; WKB = a.wkb; CGW = bool(a.cgw); BMODE = a.bmode; SURPRISE = bool(a.surprise); PKB = bool(a.pkb)
    res = {"tag": "ARCH-VET-LM-P49", "bmode": a.bmode, "surprise": a.surprise, "pkb": a.pkb, "wkb": a.wkb, "cgw": a.cgw, "dense": a.dense, "M": a.M, "task": a.task, "protocol": __doc__[:1500], "runs": []}
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
                    res["runs"].append(r); log(f"[p49 {arm} d{d} s{seed}] params {r['params']} name-CE in {r['in_name_ce']} ood {r['ood_names_name_ce']} far {r['far_ood_name_ce']} DENSE-unseen {r['dense_ood_name_ce']} | bg {r['in_bg_ce']} {r['ood_names_bg_ce']} {r['far_ood_bg_ce']}" + (f" | b sep/word {r['b_at_sep']}/{r['b_in_word']}" if arm == "CKB" else ""))
    with open("log.jsonl", "a") as fh: fh.write(json.dumps(res) + "\n")
    log("[P49] DONE")


if __name__ == "__main__":
    main()
