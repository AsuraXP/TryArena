"""kaggle_key_diag2.py — DIAGNOSTIC (instrument-first, ~5-8 min per model on a T4): which key INPUT gives phrasing-invariant addressing?

Runs 8/9 (log.md): a key built from an 8-token window of ONE mid-layer + token embeddings retrieves 0-3 % on held-out phrasings, and
training it on 7 varied forms per fact kind did not converge at all (infoNCE 0.4-1.0 vs 0.02 on fixed templates).  Before another
hour-long run we ask the cheap question directly: train only the key (InfoNCE, question-position <-> statement-position of the same
fact, in-batch negatives) on VARIED training phrasings (variant multi) and measure exact nearest-neighbour top-1 on the FROZEN held-out
phrasings (variant para) — for several candidate inputs built from the frozen base's hidden states:
  A  mid-layer window-8 + token-emb window-8            (the run 7-9 key; control)
  B  A + late-layer (3/4 depth) single token             ("meaning" vector added)
  C  late single token + causal EMA(.9) of late          (no surface window at all)
  D  A + late + EMA                                      (everything)
  E  late window-8 + token-emb window-8                  (same shape as A, deeper layer)
  F  last-layer single token + EMA(.9) of last
Scores: held-out para NN top-1 over all facts of a conversation, and over SAME-KIND facts only (the hard case); also held-out base-variant
for reference.  Pre-registered reading: if the best config is < 40 % same-kind on para, no key over these states generalises across
phrasing at this data size and the add-on must address differently (e.g. let the base answer, key on ITS answer token); if a config is
> 70 %, run 10 uses it.  usage: BASE=Qwen/Qwen2.5-1.5B python TryArena/kaggle_key_diag2.py
"""
import os, sys, random, time
BASE = os.environ.get("BASE", "HuggingFaceTB/SmolLM2-135M"); NTR = int(os.environ.get("NTR", 120)); NTE = int(os.environ.get("NTE", 30))
NF = int(os.environ.get("NF", 16)); T = int(os.environ.get("T", 2048)); STEPS = int(os.environ.get("STEPS", 1200)); WIN = 8
DROP = float(os.environ.get("DROP", 0.3)); WD = float(os.environ.get("WD", 0.01)); FRACS = [float(x) for x in os.environ.get("FRACS", "0.25,0.5,1.0").split(",")]
CONF = os.environ.get("CONF", "A,B,C,D,E,F").split(",")   # data curve (second diag run): train on fractions of the pool; 100 % train / 50 % held-out = memorisation
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work")
sys.path.insert(0, f"{WORK}/TryArena"); import kaggle_mem_test as K

def main():
    import torch, torch.nn as nn, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dt = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE); base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dt).to(dev).eval()
    E = base.get_input_embeddings().weight.float().detach(); d = E.shape[1]; sents = K.load_filler(); nl = base.config.num_hidden_layers
    LM, LL, LZ = nl // 2, (3 * nl) // 4, nl; t0 = time.time()
    def ln(h): return F.layer_norm(h, (h.shape[-1],))
    def ema(h, a=0.9):
        out = torch.empty_like(h); e = torch.zeros(h.shape[-1], device=h.device)
        for t in range(h.shape[0]): e = a * e + (1 - a) * h[t]; out[t] = e
        return out
    def win(v, p):   # (L,d) -> window ending at p, left-padded, flattened (WIN*d)
        a = max(0, p - WIN + 1); w = v[a:p + 1]; return F.pad(w, (0, 0, WIN - w.shape[0], 0)).reshape(-1)
    STREAMS = ["midw", "embw", "latew", "late1", "ema1", "last1", "lema1"]
    def extract(seed, n, variant):
        """per fact: statement pre-answer position and question pre-answer position -> dict stream -> (N, dim) fp16 cpu"""
        r = random.Random(seed); out = {s: [[], []] for s in STREAMS}; KD, CID = [], []
        for i in range(n):
            ids, spans, facts, ss = K.conversation(r, tok, NF, T, sents, variant=variant); x = torch.tensor([ids], device=dev)
            with torch.no_grad(): hs = base(input_ids=x, output_hidden_states=True).hidden_states
            hm, hl, hz = ln(hs[LM][0].float()), ln(hs[LL][0].float()), ln(hs[LZ][0].float()); ex = ln(E[x[0]]); el, ez = ln(ema(hl)), ln(ema(hz))
            sp = {fi: a for a, b, fi in ss}
            for a, b, fi in spans:
                if fi not in sp or a < 1 or sp[fi] < 1: continue
                for j, p in enumerate((sp[fi] - 1, a - 1)):   # 0 = statement, 1 = question
                    out["midw"][j].append(win(hm, p).half().cpu()); out["embw"][j].append(win(ex, p).half().cpu()); out["latew"][j].append(win(hl, p).half().cpu())
                    out["late1"][j].append(hl[p].half().cpu()); out["ema1"][j].append(el[p].half().cpu()); out["last1"][j].append(hz[p].half().cpu()); out["lema1"][j].append(ez[p].half().cpu())
                KD.append(facts[fi][0]); CID.append(i)
            del hs
        return {s: (torch.stack(v[0]), torch.stack(v[1])) for s, v in out.items()}, torch.tensor(KD), torch.tensor(CID)
    TR, KDtr, CIDtr = extract(100, NTR, "multi"); TEp, KDp, CIDp = extract(900, NTE, "para"); TEb, KDb, CIDb = extract(901, NTE, "base")
    print(f"[KEYDIAG2 {BASE}] layers mid {LM} late {LL} last {LZ} | train pairs {len(KDtr)} (multi) | test pairs para {len(KDp)} base {len(KDb)} | extract {time.time()-t0:.0f}s", flush=True)
    CONFIGS = {"A mid-w8+emb-w8": ["midw", "embw"], "B A+late1": ["midw", "embw", "late1"], "C late1+ema1": ["late1", "ema1"],
               "D A+late1+ema1": ["midw", "embw", "late1", "ema1"], "E late-w8+emb-w8": ["latew", "embw"], "F last1+lastema1": ["last1", "lema1"]}
    def feats(D, streams, side, idx=None):
        parts = [D[s][side] if idx is None else D[s][side][idx] for s in streams]; return torch.cat(parts, -1).to(dev).float()
    def evalnn(key, D, KD_, CID_, streams):
        with torch.no_grad():
            ks, kq = F.normalize(key(feats(D, streams, 0)), dim=-1), F.normalize(key(feats(D, streams, 1)), dim=-1); top = topk = n = 0
            for c in CID_.unique():
                m = (CID_ == c).nonzero().squeeze(-1); sim = kq[m] @ ks[m].t(); pred = sim.argmax(1); top += (pred == torch.arange(len(m), device=dev)).sum().item(); n += len(m)
                kd = KD_[m]
                for i in range(len(m)):
                    sub = sim[i].masked_fill(~(kd == kd[i]).to(dev), -2); topk += int(sub.argmax().item() == i)
            return 100 * top / n, 100 * topk / n
    for name, streams in CONFIGS.items():
      if name[0] not in CONF: continue
      for frac in FRACS:
        torch.manual_seed(0); din = sum(TR[s][0].shape[1] for s in streams); npool = int(frac * len(KDtr))
        key = nn.Sequential(nn.Linear(din, 512), nn.GELU(), nn.Dropout(DROP), nn.Linear(512, 256)).to(dev); opt = torch.optim.AdamW(key.parameters(), lr=1e-3, weight_decay=WD); t1 = time.time()
        for s in range(STEPS):
            key.train(); idx = torch.randint(0, npool, (256,)); ks, kq = F.normalize(key(feats(TR, streams, 0, idx)), dim=-1), F.normalize(key(feats(TR, streams, 1, idx)), dim=-1)
            loss = F.cross_entropy(kq @ ks.t() / 0.07, torch.arange(len(idx), device=dev)); opt.zero_grad(); loss.backward(); opt.step()
        key.eval(); pa, pk = evalnn(key, TEp, KDp, CIDp, streams); ba, bk = evalnn(key, TEb, KDb, CIDb, streams); ta, tk = evalnn(key, TR, KDtr, CIDtr, streams)
        print(f"[KEYDIAG2 {BASE}] {name:20s} train-pairs {npool:5d} dim {din:6d} | infoNCE {loss.item():.3f} | HELD-OUT para top-1 all {pa:5.1f} % same-kind {pk:5.1f} % | base-variant all {ba:5.1f} % same-kind {bk:5.1f} % | train all {ta:5.1f} % ({time.time()-t1:.0f}s)", flush=True)
    print(f"[KEYDIAG2 {BASE}] done {time.time()-t0:.0f}s")

if __name__ == "__main__": main()
