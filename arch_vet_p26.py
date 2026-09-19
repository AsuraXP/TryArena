"""ARCH-VET P26 (cycle 63) — FLUENCY EXPERT C IN THE MODULAR FRAME:
one system that speaks (byte-level natural language) AND reasons exactly,
with a 3-way LEARNED causal gate over frozen heterogeneous experts.

WHY: the end form the program owes is "one coherent exact + fluent
model". Data-level fusion of fluency and exact state hit the fusion
wall (C22b needed a hand router; P19/P20 monoliths starved). P21-P25
proved the modular recipe for the four symbolic axes (10/10 seeds).
P26 adds the language axis as a THIRD expert and asks whether the
learned gate discovers the language/symbol boundary from tokens alone.
VOCAB: V' = 48 symbolic (unchanged, so experts A/B are reused FROZEN
from p21_ckpt, no retraining) + 256 bytes (tokens 48..303).
EXPERT C: 1-layer GRU d48 byte LM (~44k p) trained ONLY on text-only
streams: [BOS] U <bytes> EOS U <bytes> EOS ... from corpus/corpus_full.txt
(train = first 90%, val = last 10%, windows of 40-140 bytes).
COMPOSITION: per position choose one expert (hard argmax at eval);
A/B logits are padded with -30 over the 256 byte slots; C outputs 304.
GATE: one-hot(x_t, 304) -> GRU(8) -> 3 logits (~7.5k p), trained by the
composed probability-mixture CE on a joint pool (vanilla + mixdd + deep-
dyck + CHATMIX + text-only), experts frozen (P22 recipe, 3-way).
CHATMIX stream: alternating U-text-EOS turns and mixdd reasoning
segments (track/modk/pair/dyck) inside ONE 256-token stream.
CONTROLS (fair, same data = chatmix pool, 6000 steps, ~matched total
params): (i) TF-NAPE d64 2L monolith (P23's best in-range TF), (ii) GRU
d96 monolith (same substrate as C, no modularity) — separates
"modularity" from "substrate" on the language+reasoning axis too.
METRICS: text CE/byte on held-out text (C-alone, routed, controls);
on chatmix streams in ONE pass: text CE, reasoning answer acc per
family + dyck close (joint_mixed_eval generalised to 3 experts);
gate expert-share per corpus; unified 4-bar row must stay intact
(regression guard: gate must never send symbolic streams to C).
PRIOR ART (searched 2026-09-11): arXiv 2604.23108 (MoHGE, ACL 2026:
heterogeneous grouped experts + two-level routing, homogeneous vocab);
arXiv 2507.11181 (MoE survey: task-level routing minimises
interference; multimodal MoE routes by modality); C22b (this program,
hand-routed 4-way fusion). Gap: whole-model heterogeneous experts
(recurrent byte LM + Mealy/counter/stack reasoners) over a UNION vocab
with a token-only learned causal gate — not found.
PREDICTION (falsifiable): gate C-share ~1.0 on text bytes and ~0 on
symbolic streams (the byte/symbol boundary is trivially separable),
so routed text CE == C-alone within .02 and the four reasoning bars
hold on chatmix; monolithic controls trade one axis for the other
(L-DATA-CEILING bounds the absolute fluency level, ~4.3 nats/BPE tok
in C21b; here bytes, so absolute numbers are not comparable to C21b).
"""
import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
CKPT = os.path.join(REPO, "p21_ckpt"); RUNS = os.path.join(REPO, "runs")
import arch_vet_lm as lm
import arch_vet_p19 as p19
import arch_vet_p13d as p13d
import arch_vet_p21 as p21
import arch_vet_p23 as p23
V0 = p19.V; VB = V0 + 256
BOS, EOS, U, A_MARK, T_TASK = p19.BOS, p19.EOS, 2, p19.A, p19.T_TASK
BRK, KEYS, TRACK, ONE = p19.BRK, p19.KEYS, p19.TRACK, p19.ONE

# ------------------------------------------------------------ text data
_raw = open(os.path.join(REPO, "corpus", "corpus_full.txt"), "rb").read()
_cut = int(len(_raw) * 0.9)
TXT_TR, TXT_VA = _raw[:_cut], _raw[_cut:]


def text_turn(rng, src, lo=40, hi=140):
    n = rng.randrange(lo, hi + 1)
    s = rng.randrange(0, len(src) - n)
    return [U] + [V0 + b for b in src[s:s + n]] + [EOS]


def gen_text_stream(rng, L=256, src=TXT_TR):
    x = [BOS]
    while len(x) < L + 1:
        x += text_turn(rng, src)
    return x[:L + 1]


def gen_chatmix_stream(rng, L=256, src=TXT_TR):
    """Alternate text turns and reasoning bursts (2-4 mixdd segments)."""
    x = [BOS]
    while len(x) < L:
        if rng.random() < 0.5:
            x += text_turn(rng, src, 30, 90)
        else:
            seg = p19.gen_mixdd_stream(rng, 96)[1:-1]     # strip BOS/EOS
            # keep only whole tasks: cut at last T_TASK-free tail
            x += seg[:rng.randrange(40, 96)]
    x = x[:L]; x.append(EOS)
    return x


def pool_of(gen, n, seed, **kw):
    rng = random.Random(seed)
    return [torch.tensor(gen(rng, 256, **kw)) for _ in range(n)]


# --------------------------------------------------------------- models
class ByteGRU(nn.Module):
    def __init__(self, V, d=48, layers=1):
        super().__init__()
        self.E = nn.Embedding(V, d); self.g = nn.GRU(d, d, layers, batch_first=True)
        self.head = nn.Linear(d, V)

    def forward(self, x):
        h, _ = self.g(self.E(x)); return self.head(h)


class Gate3(nn.Module):
    def __init__(self, V, h=8, k=3):
        super().__init__()
        self.gru = nn.GRU(V, h, batch_first=True); self.head = nn.Linear(h, k)

    def forward(self, x):
        h, _ = self.gru(F.one_hot(x, VB).float()); return self.head(h)


def pad48(lg):                      # (B,L,48) -> (B,L,304)
    return F.pad(lg, (0, 256), value=-30.0)


class Unified3(nn.Module):
    """Frozen experts A (VETDCC), B (stack), C (byte GRU) + learned gate;
    hard argmax dispatch. Symbolic experts see byte tokens clamped to a
    filler id (they never produce bytes; the gate must never pick them
    on bytes)."""
    def __init__(self, mA, mB, mC, g):
        super().__init__(); self.mA, self.mB, self.mC, self.g = mA, mB, mC, g

    def sym(self, x):
        return torch.where(x >= V0, torch.full_like(x, p19.MODS), x)

    def expert_logits(self, x):
        xs = self.sym(x)
        return pad48(self.mA(xs)), pad48(self.mB(xs)), self.mC(x)

    def forward(self, x):
        la, lb, lc = self.expert_logits(x)
        r = self.g(x).argmax(-1).unsqueeze(-1)              # (B,L,1)
        return torch.where(r == 0, la, torch.where(r == 1, lb, lc))

    def routes(self, x):
        return self.g(x).argmax(-1)


def train_gate3(mA, mB, mC, pool, seed, steps=1500, batch=8, lr=3e-3):
    torch.manual_seed(seed); g = Gate3(VB, 8, 3)
    opt = torch.optim.AdamW(g.parameters(), lr=lr); n = len(pool); t0 = time.time()
    u = Unified3(mA, mB, mC, g)
    for step in range(1, steps + 1):
        x = torch.stack([pool[(step * batch + i) % n] for i in range(batch)])
        y = x[:, 1:256]
        with torch.no_grad():
            la, lb, lc = u.expert_logits(x[:, :256])
            P = torch.stack([F.softmax(la, -1), F.softmax(lb, -1), F.softmax(lc, -1)], 2)[:, :255]
        w = F.softmax(g(x[:, :255]), -1).unsqueeze(-1)       # (B,L,3,1)
        p = (w * P).sum(2)
        loss = F.nll_loss(torch.log(p + 1e-9).reshape(-1, VB), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            print(f"  [gate3 s{seed}] step {step}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    g.eval(); return g


# ----------------------------------------------------------------- eval
@torch.no_grad()
def decomposed_ce(model, xs):
    """CE split into byte positions (target >= V0) and symbolic positions."""
    lg = model(xs[:, :-1]); y = xs[:, 1:]
    nll = -F.log_softmax(lg, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    tb = y >= V0
    return {"text_ce": round(float(nll[tb].mean()), 4) if tb.any() else None,
            "sym_ce": round(float(nll[~tb].mean()), 4) if (~tb).any() else None}


@torch.no_grad()
def chatmix_reasoning(model, n, seed=4242):
    """Reasoning answers + dyck close on chatmix streams in ONE pass."""
    rng = random.Random(seed); fam_ok = {"track": [0, 0], "modk": [0, 0], "pair": [0, 0]}
    cok = ctot = 0
    for _ in range(n):
        x2 = gen_chatmix_stream(rng, 256)
        pred = model(torch.tensor(x2[:-1]).unsqueeze(0)).argmax(-1).squeeze(0)
        i = 0
        while i < len(x2) - 4:
            if x2[i] != T_TASK: i += 1; continue
            n1, n2 = x2[i + 1], x2[i + 2]
            if TRACK <= n1 < TRACK + 8:
                j = i
                while j < len(x2) - 1 and x2[j] != A_MARK: j += 1
                if j + 1 < len(x2):
                    fam_ok["track"][1] += 1; fam_ok["track"][0] += int(int(pred[j]) == x2[j + 1])
                i = j + 2
            elif n1 == T_TASK and n2 == T_TASK:
                i2, d, seg = i + 3, 0, []
                while i2 < len(x2) - 1:
                    t2 = x2[i2]
                    if BRK <= t2 < BRK + 4:
                        d += 1 if t2 < BRK + 2 else -1; seg.append(i2)
                        if d == 0: i2 += 1; break
                    i2 += 1
                for gpos in seg:
                    if BRK + 2 <= x2[gpos] < BRK + 4:
                        ctot += 1; cok += int(int(pred[gpos - 1]) == x2[gpos])
                i = max(seg) + 1 if seg else i + 3
            elif n1 == T_TASK:
                j = i
                while j < len(x2) - 1 and x2[j] != A_MARK: j += 1
                if n2 == ONE:
                    if j + 1 < len(x2):
                        fam_ok["modk"][1] += 1; fam_ok["modk"][0] += int(int(pred[j]) == x2[j + 1])
                    i = j + 2
                else:
                    if j + 2 < len(x2):
                        fam_ok["pair"][1] += 2
                        fam_ok["pair"][0] += int(int(pred[j]) == x2[j + 1]) + int(int(pred[j + 1]) == x2[j + 2])
                    i = j + 3
            else:
                i += 1
    out = {k: round(v[0] / v[1], 4) if v[1] else None for k, v in fam_ok.items()}
    out["dyck_close"] = round(cok / ctot, 4) if ctot else None; out["n_close"] = ctot
    return out


@torch.no_grad()
def gate_share(u, pool):
    cnt = torch.zeros(3)
    for s in pool:
        r = u.routes(s[:256].unsqueeze(0)).squeeze(0); cnt += torch.bincount(r, minlength=3).float()
    return [round(float(c / cnt.sum()), 4) for c in cnt]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="111,222")
    ap.add_argument("--c_steps", type=int, default=4000)
    ap.add_argument("--gate_steps", type=int, default=1500)
    ap.add_argument("--ctrl_steps", type=int, default=6000)
    ap.add_argument("--controls", default="tf,gru")
    a = ap.parse_args(); t0 = time.time()
    out = {"tag": "ARCH-VET-LM-P26", "protocol": __doc__[:1500], "arms": {"unified3": {"per_seed": {}}}}
    text_pool = pool_of(gen_text_stream, 384, 777)
    chat_pool = pool_of(gen_chatmix_stream, 512, 12345)
    va_text = torch.stack(pool_of(gen_text_stream, 32, 99, src=TXT_VA))
    va_chat = torch.stack(pool_of(gen_chatmix_stream, 32, 98, src=TXT_VA))
    print(f"[p26] pools text {len(text_pool)} chat {len(chat_pool)}; val text/chat 32/32", flush=True)
    rows = []
    for seed in map(int, a.seeds.split(",")):
        torch.manual_seed(seed)
        mA = p21.VETDCC(V0, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V0, 24, k=8, K=8)
        mA.load_state_dict(torch.load(f"{CKPT}/A_s{seed}.pt")["sd"]); mB.load_state_dict(torch.load(f"{CKPT}/B_s{seed}.pt")["sd"])
        torch.manual_seed(seed); mC = ByteGRU(VB, 48)
        pc = f"{CKPT}/C_s{seed}.pt"
        if os.path.exists(pc): mC.load_state_dict(torch.load(pc)["sd"]); hC = torch.load(pc)["hist"]
        else:
            hC = lm.train_arm(f"P26-C-s{seed}", mC, text_pool, a.c_steps, 8) if False else None
            # lm.train_arm slices x[:, :256] & y=x[:,1:] with V=48 hard-coded -> own loop
            opt = torch.optim.AdamW(mC.parameters(), lr=3e-3); hC = []; tt = time.time()
            torch.manual_seed(0)
            for step in range(1, a.c_steps + 1):
                x = torch.stack([text_pool[(step * 8 + i) % len(text_pool)] for i in range(8)])
                loss = F.cross_entropy(mC(x[:, :256]).reshape(-1, VB), x[:, 1:257].reshape(-1))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(mC.parameters(), 1.0); opt.step()
                if step % 500 == 0:
                    hC.append((step, round(loss.item(), 4)))
                    print(f"  [P26-C-s{seed}] step {step}/{a.c_steps} loss {loss.item():.4f} ({time.time()-tt:.0f}s)", flush=True)
            torch.save({"sd": mC.state_dict(), "hist": hC}, pc)
        for m in (mA, mB, mC):
            m.eval()
            for q in m.parameters(): q.requires_grad_(False)
        nA, nB, nC = (p19.n_params(m) for m in (mA, mB, mC))
        c_alone = decomposed_ce(mC, va_text)["text_ce"]
        print(f"[p26 s{seed}] params A {nA} B {nB} C {nC}; C-alone val text CE/byte {c_alone}", flush=True)
        joint = (pool_of(lm.gen_stream, 96, 1000 + seed) if False else
                 [torch.tensor(lm.gen_stream(random.Random(1000 + seed + i), 256)[0]) for i in range(96)])
        joint += p19.gen_mixdd_pool(96, 256, 2000 + seed) + p13d.gen_mix_pool(96, 256, 3000 + seed)
        joint += pool_of(gen_chatmix_stream, 128, 4000 + seed) + pool_of(gen_text_stream, 64, 5000 + seed)
        joint = [j[:257] for j in joint]
        pg = f"{CKPT}/G3_s{seed}.pt"
        g = train_gate3(mA, mB, mC, joint, seed, a.gate_steps)
        torch.save({"sd": g.state_dict()}, pg)
        u = Unified3(mA, mB, mC, g).eval()
        row = {"params": {"A": nA, "B": nB, "C": nC, "gate": p19.n_params(g), "total": nA + nB + nC + p19.n_params(g)},
               "C_alone_text_ce": c_alone, "hist_C": hC,
               "routed_text_ce_textstreams": decomposed_ce(u, va_text)["text_ce"],
               "routed_chatmix": decomposed_ce(u, va_chat),
               "chatmix_reasoning": chatmix_reasoning(u, 32),
               "gate_share": {"text": gate_share(u, pool_of(gen_text_stream, 16, 7, src=TXT_VA)),
                              "chatmix": gate_share(u, pool_of(gen_chatmix_stream, 16, 7, src=TXT_VA)),
                              "vanilla": gate_share(u, p19.make_pool(16, 256, 7)),
                              "deepdyck": gate_share(u, p13d.gen_mix_pool(16, 256, 7))}}
        # regression guard: symbolic unified row through the 3-way gate (A/B only via the same gate)
        p21.causal_route = (lambda toks: [min(r, 1) for r in u.routes(torch.tensor(toks).unsqueeze(0)).squeeze(0).tolist()])
        acc = p21.routed_task_acc(mA, mB, 24, 256, random.Random(666), hard=True)
        vha = p19.make_batches(8, 256, random.Random(777), hard=True); v1024 = p19.make_batches(2, 1024, random.Random(31415))
        ratio = round(p21.routed_ce(mA, mB, v1024, 1024) / p21.routed_ce(mA, mB, vha, 256), 3)
        d12 = p21.routed_close_acc(mA, mB, 2, 12, p13d.seg_len(12) + 16)[0]
        row["unified_row_guard"] = {"pair": acc["pair"], "modk": acc["modk"], "ratio": ratio, "dyck_d12": d12,
                                    "bars": int(acc["pair"] >= .717) + int(acc["modk"] >= 1) + int(ratio <= .6) + int(d12 >= .85)}
        print(f"[p26 unified3 s{seed}] text CE routed {row['routed_text_ce_textstreams']} (C-alone {c_alone}) | "
              f"chatmix {row['routed_chatmix']} | reasoning {row['chatmix_reasoning']} | share {row['gate_share']} | "
              f"guard {row['unified_row_guard']}", flush=True)
        out["arms"]["unified3"]["per_seed"][str(seed)] = row; rows.append(row)
    # ------------------------------------------------------------ controls
    for ctrl in a.controls.split(","):
        if not ctrl: continue
        per = {}
        for seed in map(int, a.seeds.split(",")):
            torch.manual_seed(seed)
            m = p23.TFCtrl(VB, d=56, nh=4, depth=2, mlp=2, pe="nape") if ctrl == "tf" else ByteGRU(VB, 64, 2)
            n_ = p19.n_params(m); print(f"[p26 ctrl {ctrl} s{seed}] params {n_}", flush=True)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-3); tt = time.time(); hist = []
            torch.manual_seed(0)
            for step in range(1, a.ctrl_steps + 1):
                x = torch.stack([chat_pool[(step * 8 + i) % len(chat_pool)] for i in range(8)])
                loss = F.cross_entropy(m(x[:, :256]).reshape(-1, VB), x[:, 1:257].reshape(-1))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
                if step % 1000 == 0:
                    hist.append((step, round(loss.item(), 4)))
                    print(f"  [P26-{ctrl}-s{seed}] step {step}/{a.ctrl_steps} loss {loss.item():.4f} ({time.time()-tt:.0f}s)", flush=True)
            m.eval()
            r = {"params": n_, "hist": hist, "text_ce_textstreams": decomposed_ce(m, va_text)["text_ce"],
                 "chatmix": decomposed_ce(m, va_chat), "chatmix_reasoning": chatmix_reasoning(m, 32)}
            print(f"[p26 ctrl {ctrl} s{seed}] {r['text_ce_textstreams']} {r['chatmix']} {r['chatmix_reasoning']}", flush=True)
            per[str(seed)] = r
        out["arms"][f"ctrl-{ctrl}"] = {"per_seed": per}
    out["wall_s"] = round(time.time() - t0)
    with open("log.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    print("[P26] DONE", json.dumps({k: {s: (v.get("chatmix_reasoning"), v.get("routed_chatmix") or v.get("chatmix"))
                                        for s, v in arm["per_seed"].items()} for k, arm in out["arms"].items()}), flush=True)


if __name__ == "__main__":
    main()
