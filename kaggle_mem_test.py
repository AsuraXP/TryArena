"""kaggle_mem_test.py — integration test of TokenMemory (memory_module.py) on a FROZEN pretrained small model.

Kaggle cell (GPU T4 x2, Internet ON; for gated models add an HF_TOKEN secret and enable it for the notebook):
    !git clone -q -b arena/01a07767-tryarena --depth 1 https://github.com/AsuraXP/TryArena /kaggle/working/TryArena
    !cd /kaggle/working && pip install -q transformers && BASE=HuggingFaceTB/SmolLM2-135M python TryArena/kaggle_mem_test.py
Then (if the 135M run looks sane):  BASE=meta-llama/Llama-3.2-3B  (or BASE=Qwen/Qwen2.5-0.5B, ungated)

ARMS (same conversations, same seeds):
  full    base model, whole conversation in context (strong control; capped at FULLMAX tokens by GPU memory)
  window  base model sees only the last SEG tokens (segments processed independently)  -> what bounded memory costs a Transformer
  mem     window + TokenMemory carrying facts across segments (module state O(1) in length)
TESTS (pre-registered, see bottom of this header):
  recall       exact-match of the answer tokens for facts stated earlier (names / order numbers / file paths / dates), by length 2k/8k/16k
  confusion    answers that are a DIFFERENT stated fact of the same type (the model mixing up what it was told)
  no-harm      perplexity on ordinary prose, base vs base+mem (must not move)
  cost         module state size vs base KV-cache size at each length
DATA: synthetic conversations: NF random facts (random syllable names, random numbers/paths/dates: never memorisable), spread
  through real prose filler (public-domain books, build_corpus.py), queries at the end.  Random fillers are the S1 lesson.

PRE-REGISTRATION (written before the first run):
  P1 at 2k tokens the full-context base recalls >= 90 % (it has attention over everything) -> the module cannot beat it there,
     and must not hurt: mem >= window + 40 points, mem within 10 points of full.
  P2 at 8k and 16k (beyond SEG, and for 16k beyond FULLMAX): window recall ~0 for facts outside the last segment; mem recall >= 60 %
     and FLAT between 8k and 16k (within 10 points).  Falsifier: mem < 40 % at 8k, or a > 20-point drop 8k -> 16k.
  P3 confusion rate of mem < confusion rate of full at 8k (the store is addressed by content; attention over 30 similar facts dilutes).
  P4 no-harm: prose perplexity of base+mem within 1 % of base.
  P5 cost: module state (dk*d floats per resolution) is constant; base KV at 16k is > 10x the module state for the 3B.
"""
import os, sys, time, json, math, random, re, subprocess

BASE = os.environ.get("BASE", "HuggingFaceTB/SmolLM2-135M")
SEG = int(os.environ.get("SEG", 512)); FULLMAX = int(os.environ.get("FULLMAX", 8192))
STEPS = int(os.environ.get("STEPS", 600)); B = int(os.environ.get("B", 4)); TRAIN_LEN = int(os.environ.get("TRAIN_LEN", 2048))
NF_TRAIN = int(os.environ.get("NF", 16)); NEVAL = int(os.environ.get("NEVAL", 40)); LENS = [int(x) for x in os.environ.get("LENS", "2048,8192,16384").split(",")]
GFLOOR = float(os.environ.get("GFLOOR", 0.05)); AUXW = float(os.environ.get("AUXW", 1.0)); MODULE = os.environ.get("MODULE", "learned"); CONW = float(os.environ.get("CONW", 1.0)); WGW = float(os.environ.get("WGW", 1.0))
DK = int(os.environ.get("DK", 4096)); SPARSE = float(os.environ.get("SPARSE", 0.02)); MRES = int(os.environ.get("MRES", 2))
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("work"); os.makedirs(WORK, exist_ok=True)
TAG = re.sub(r"[^A-Za-z0-9]+", "_", BASE)
logf = open(f"{WORK}/mem_{TAG}.log", "w")   # fresh log per run (append mode mixed runs)
def log(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); logf.write(s + "\n"); logf.flush()

# ---------------------------------------------------------------- data
SYL = "ka ri to na ve lo mi su dar en ol ith ar un bel kor zan fel tor ash om quil ren vos".split()
def rname(r): return "".join(r.choice(SYL) for _ in range(r.randint(2, 3))).capitalize()
FACTS = [  # (statement template, question, answer prefix) ; {A} = the random answer
    ("my {rel}'s name is {A}", "what is my {rel}'s name?", "Your {rel}'s name is"),
    ("the order number for the {thing} is {A}", "what is the order number for the {thing}?", "The order number for the {thing} is"),
    ("the {thing} config lives at {A}", "where does the {thing} config live?", "The {thing} config lives at"),
    ("the {thing} meeting is on {A}", "when is the {thing} meeting?", "The {thing} meeting is on"),
]
RELS = "dog cat sister brother boss neighbour landlord dentist cousin plumber".split()
THINGS = "printer garden server budget kitchen roof database newsletter warehouse studio".split()
MONTHS = "January February March April May June July August September October November December".split()
def make_fact(r, i):
    kind = i % 4; t = FACTS[kind]; slot = {"rel": RELS[r.randrange(len(RELS))], "thing": THINGS[r.randrange(len(THINGS))]}
    A = [rname(r), str(r.randint(10000, 999999)), f"/srv/{rname(r).lower()}/{rname(r).lower()}.yml", f"{MONTHS[r.randrange(12)]} {r.randint(1,28)}"][kind]
    return kind, slot, A, t[0].format(A=A, **slot), t[1].format(**slot), t[2].format(**slot)

def load_filler():
    p = f"{WORK}/prose_fill.txt"
    if not os.path.exists(p):
        subprocess.check_call([sys.executable, f"{WORK}/TryArena/build_corpus.py", "--out", p, "--mb", "20", "--src", f"{WORK}/sfgram"])
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", open(p).read()) if 40 < len(s) < 220 and "\n" not in s]
    return sents

def conversation(r, tok, nfacts, target_len, sents, nquery=None):
    """returns token ids, answer spans [(start,end,fact_idx)], facts.  Facts spread through filler; queries at the end.
    Length is calibrated in tokens: a first draft measures the tokens-per-character ratio, the second draft hits target_len (+-5 %)."""
    seed = r.random(); ids, spans, facts, q0, ss = _conversation(random.Random(seed), tok, nfacts, target_len, sents, nquery, 1.0)
    qtok = len(ids) - q0; scale = max(0.05, (target_len - qtok) / max(1, q0))     # rescale the filler only; questions are measured exactly
    if abs(len(ids) - target_len) > 0.05 * target_len: ids, spans, facts, q0, ss = _conversation(random.Random(seed), tok, nfacts, target_len, sents, nquery, scale)
    return ids, spans, facts, ss   # spans: question answers (a,b,fact_idx); ss: statement answers (a,b,fact_idx)

def _conversation(r, tok, nfacts, target_len, sents, nquery, scale):
    facts = [make_fact(r, i) for i in range(nfacts)]; uniq = {}
    for f in facts: uniq.setdefault((f[0], f[1]["rel"] if f[0] == 0 else f[1]["thing"]), f)   # drop same-slot duplicates
    facts = list(uniq.values()); nq = nquery or len(facts); qs = r.sample(range(len(facts)), min(nq, len(facts)))
    # question block size estimate ~ 25 tokens each
    budget = target_len * scale; pieces = []; fi = 0; per = max(1, int(budget) // max(1, len(facts)))
    for fi, f in enumerate(facts):
        chunk = []; ln = 0
        while ln < per - 25:
            s = r.choice(sents); chunk.append(f"User: {s}\nAssistant: {r.choice(sents)}\n"); ln += len(s) // 4 + len(chunk[-1]) // 5
        pos = r.randrange(len(chunk) + 1); pieces += [(c, None) for c in chunk[:pos]]
        pieces += [(f"User: By the way, {f[3][:-len(f[2])-1]}", None), (" " + f[2], ("s", fi)), (".\nAssistant: Noted.\n", None)]   # statement answer tagged ('s', fi)
        pieces += [(c, None) for c in chunk[pos:]]
    qchar = sum(len(p) for p, _ in pieces)
    for qi in qs:
        f = facts[qi]; pieces.append((f"User: {f[4]}\nAssistant: {f[5]}", None)); pieces.append((" " + f[2], qi)); pieces.append((".\n", None))
    text = "".join(p for p, _ in pieces); enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc["input_ids"], enc["offset_mapping"]; spans = []; sspans = []; pos = 0
    for p, qi in pieces:
        if qi is not None:
            a, b = pos, pos + len(p); ti = [i for i, (s, e) in enumerate(offs) if e > a and s < b]
            if ti: (sspans if isinstance(qi, tuple) else spans).append((ti[0], ti[-1] + 1, qi[1] if isinstance(qi, tuple) else qi))
        pos += len(p)
    q0 = next((i for i, (a, e) in enumerate(offs) if e > qchar), len(ids))
    return ids, spans, facts, q0, sspans

# ---------------------------------------------------------------- model plumbing
def main():
    import torch, torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    sys.path.insert(0, f"{WORK}/TryArena"); from memory_module import TokenMemory, LearnedKeyMemory
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); dt = torch.float16 if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(BASE, token=os.environ.get("HF_TOKEN")); base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=dt, token=os.environ.get("HF_TOKEN")).to(dev).eval()
    for p in base.parameters(): p.requires_grad_(False)
    E = base.get_input_embeddings().weight; d = E.shape[1]; V = E.shape[0]
    nl = base.config.num_hidden_layers; HL = nl // 2 if MODULE == "learned" else nl   # learned key reads the MID layer (key diagnostic: L15 of 30 best)
    mem = (LearnedKeyMemory(d, E) if MODULE == "learned" else TokenMemory(d, E, dk=DK, sparse=SPARSE, mres=MRES)).to(dev); opt = torch.optim.Adam(mem.parameters(), lr=1e-3)
    nb = sum(p.numel() for p in base.parameters()); log(f"[SETUP] base {BASE} params {nb/1e6:.0f}M d {d} V {V} | module {MODULE} params {mem.n_params()} hidden layer {HL}/{nl} | SEG {SEG} FULLMAX {FULLMAX} | device {dev}")
    sents = load_filler(); log(f"[SETUP] filler sentences {len(sents)}")

    @torch.no_grad()
    def base_fwd(ids):   # ids (B,L) -> hidden (B,L,d) float32, logp (B,L,V) float32
        out = base(input_ids=ids, output_hidden_states=True); return out.hidden_states[HL].float(), F.log_softmax(out.logits.float(), -1)

    def run_stream(ids, arm, fn):
        """process (B,T) ids in segments; calls fn(logp_seg, seg_ids, s0) per segment and returns the list of its results.
        Full-vocab log-probs are never kept for more than one segment (memory)."""
        Bn, T = ids.shape; res = []
        if arm == "full":   # one base forward over the whole context; log-probs materialised 1024 positions at a time (152k vocab x 8k tokens would not fit)
            with torch.no_grad(): logits = base(input_ids=ids).logits
            for c0 in range(0, T, 1024):
                lp = F.log_softmax(logits[:, c0:c0 + 1024].float(), -1); res.append(fn(lp, ids[:, c0:c0 + 1024], c0)); del lp
            del logits; return res
        state = mem.init_state(Bn, dev)
        for s0 in range(0, T, SEG):
            seg = ids[:, s0:s0 + SEG]; h, lp = base_fwd(seg)
            if arm in ("mem", "copy"): lp, state = mem(h, lp, seg, state, targets=seg[:, 1:] if mem.training else None, gfloor=GFLOOR if mem.training else 0.0, force_g=1.0 if arm == "copy" else None)
            res.append(fn(lp, seg, s0)); del lp
        return res

    # ---------------- train the module (base frozen) on TRAIN_LEN conversations, segments of SEG carried
    r = random.Random(1); t0 = time.time(); mem.train()
    def pad_batch(convs):
        T = max(len(c[0]) for c in convs); ids = torch.full((len(convs), T), tok.pad_token_id or 0, dtype=torch.long); mask = torch.zeros(len(convs), T)
        wm = torch.zeros(len(convs), T)
        for i, (c, spans, _, ss) in enumerate(convs):
            ids[i, :len(c)] = torch.tensor(c); mask[i, :len(c)] = 1.0
            for a, b, _ in spans: mask[i, a:b] = 6.0            # answer tokens weighted up
            for a, b, _ in ss: wm[i, a:b] = 1.0                 # write targets: statement answer tokens
        return ids.to(dev), mask.to(dev), wm.to(dev)
    for s in range(STEPS):
        convs = [conversation(r, tok, NF_TRAIN, TRAIN_LEN, sents, nquery=8) for _ in range(B)]; ids, w, wm = pad_batch(convs); tot = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        # (question pre-answer position, statement pre-answer position) pairs per conversation, for the contrastive key loss
        pairs = [];  # (conv i, qpos, spos)
        for i, (c, spans, facts, ss) in enumerate(convs):
            sp = {fi: (a, b) for a, b, fi in ss}
            for a, b, fi in spans:   # every answer token j: question pos a-1+j <-> statement pos sa-1+j (same token offset)
                if fi in sp and a > 0:
                    sa, sb = sp[fi]; pairs += [(i, a - 1 + j, sa - 1 + j) for j in range(min(b - a, sb - sa))]
        qf, sf = {}, {}
        def step_fn(lp, seg, s0):   # per-segment loss + backward (state is detached between segments: truncated BPTT at SEG)
            ws = w[:, s0:s0 + seg.shape[1]]; nll = -lp[:, :-1].gather(-1, seg[:, 1:, None]).squeeze(-1)
            ans = (ws[:, 1:] > 1).float(); l = ((nll * ws[:, 1:]).sum() + AUXW * (mem.aux_nll * ans).sum()) / (ids.shape[0] * ids.shape[1])   # aux: copy path on answer tokens only
            if MODULE == "learned":   # learned write gate: BCE against statement-answer targets (positives rare -> pos_weight)
                wt = wm[:, s0:s0 + seg.shape[1]]; wl = F.binary_cross_entropy_with_logits(mem.wlogit, wt, pos_weight=torch.tensor(20.0, device=dev), reduction="sum") / (ids.shape[0] * ids.shape[1])
                l = l + WGW * wl; tot[5] += wl.item() * ids.shape[0] * ids.shape[1]
            tot[2] += (mem.aux_nll * ans).sum().item(); tot[3] += ans.sum().item(); tot[4] += (nll * ans).sum().item()   # diagnostics: copy-path / mixed NLL on answer tokens
            if MODULE == "learned":   # stash detached window features at pair positions that fall in this segment
                for j, (i, qp, spp) in enumerate(pairs):
                    if s0 <= qp < s0 + seg.shape[1]: qf[j] = mem.win_feats[i, qp - s0]
                    if s0 <= spp < s0 + seg.shape[1]: sf[j] = mem.win_feats[i, spp - s0]
            l.backward(); l = l * ids.shape[0] * ids.shape[1]; tot[0] += l.item(); tot[1] += ws[:, 1:].sum().item(); return None
        opt.zero_grad(); run_stream(ids, "mem", step_fn); con = torch.zeros((), device=dev)
        if MODULE == "learned":
            js = [j for j in range(len(pairs)) if j in qf and j in sf]
            if len(js) > 1:
                kq = F.normalize(mem.key(torch.stack([qf[j] for j in js])), dim=-1); ks = F.normalize(mem.key(torch.stack([sf[j] for j in js])), dim=-1)
                con = F.cross_entropy(kq @ ks.t() / 0.07, torch.arange(len(js), device=dev)); (CONW * con).backward()
        torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0); opt.step()
        if s % 25 == 0: log(f"  [train] step {s} loss {tot[0]/max(1,tot[1]):.3f} gate {mem.last['g']:.3f} (learned {mem.last['graw']:.4f}) write {mem.last['beta']:.3f} (frac>.5 {mem.last.get('wfrac', float('nan')):.3f}) theta {mem.theta.item():.2f} | answer-token NLL: copy-path {tot[2]/max(1,tot[3]):.3f} mixed {tot[4]/max(1,tot[3]):.3f} | key infoNCE {con.item():.3f} | write-gate BCE {tot[5]/max(1,tot[1]):.4f} ({time.time()-t0:.0f}s)")
    mem.eval(); torch.save(mem.state_dict(), f"{WORK}/mem_{TAG}.pt")

    # ---------------- evaluation
    @torch.no_grad()
    def evaluate(T, arm, n=NEVAL, seed=777):
        er = random.Random(seed); hit = tot = conf = 0; tk_ok = tk_n = first_ok = 0
        for i in range(n):
            nf = int(os.environ.get("NFEVAL", 0)) or max(8, min(40, T // 400)); c, spans, facts, _ss = conversation(er, tok, nf, int(T * 0.96), sents)   # 4 % slack under the cap
            ids = torch.tensor([c], device=dev)
            if arm == "full" and ids.shape[1] > FULLMAX: return None
            try: pred = torch.cat(run_stream(ids, arm, lambda lp, seg, s0: lp[0].argmax(-1)), 0)
            except torch.cuda.OutOfMemoryError: torch.cuda.empty_cache(); return None
            for a, b, qi in spans:
                tot += 1; eq = (pred[a - 1:b - 1] == ids[0, a:b]); ok = bool(eq.all()); hit += ok; tk_ok += int(eq.sum()); tk_n += eq.numel(); first_ok += int(eq[0])
                if not ok:   # confusion: predicted answer equals another stated fact of the same kind
                    ptxt = tok.decode(pred[a - 1:b - 1]).strip(); kind = facts[qi][0]
                    conf += any(f[2].strip() == ptxt for j, f in enumerate(facts) if j != qi and f[0] == kind)
        return {"recall": round(100 * hit / max(1, tot), 1), "first_tok": round(100 * first_ok / max(1, tot), 1), "tok_acc": round(100 * tk_ok / max(1, tk_n), 1), "confusion": round(100 * conf / max(1, tot), 1), "n_answers": tot}
    for T in LENS:
        for arm in ["full", "window", "mem", "copy"]:   # copy = diagnostic: gate forced to 1 (pure retrieval; shows whether the store holds the facts)
            t1 = time.time(); res = evaluate(T, arm); log(f"[RESULT {TAG}] len {T} arm {arm}: {res} ({time.time()-t1:.0f}s)")
    # ---------------- no-harm: prose perplexity base vs base+mem (streamed with SEG either way)
    @torch.no_grad()
    def prose_ppl(arm, n=20, T=2048, seed=5):
        pr = random.Random(seed); tot = cnt = 0.0; txt = " ".join(sents)
        for _ in range(n):
            o = pr.randrange(0, len(txt) - 12000); ids = torch.tensor([tok(txt[o:o + 12000], add_special_tokens=False)["input_ids"][:T]], device=dev)
            tot += sum(run_stream(ids, arm, lambda lp, seg, s0: -lp[0, :-1].gather(-1, seg[0, 1:, None]).sum().item())); cnt += ids.shape[1] - 1
        return round(math.exp(tot / cnt), 3)
    pw, pm = prose_ppl("window"), prose_ppl("mem"); log(f"[NOHARM {TAG}] prose ppl window {pw} mem {pm} (delta {100*(pm/pw-1):+.2f} %)")
    # ---------------- cost
    cfg = base.config; nl = getattr(cfg, "num_hidden_layers", 0); nkv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads", 1)); hd = d // getattr(cfg, "num_attention_heads", 1)
    for T in LENS:
        kv = 2 * nl * nkv * hd * T * 2 / 1e6; st = MRES * DK * d * 4 / 1e6
        log(f"[COST {TAG}] len {T}: base KV cache {kv:.0f} MB (fp16) | module state {st:.0f} MB (constant) | window KV {2*nl*nkv*hd*SEG*2/1e6:.0f} MB")
    log(f"[DONE {TAG}] total {time.time()-t0:.0f}s")

if __name__ == "__main__": main()
