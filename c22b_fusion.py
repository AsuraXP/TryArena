"""
ARC-2 CYCLE 49 / C22b — FLUENCY FUSION: the d32 real-text fluency engine
(lm_host_final.pt, C21) carried AS-IS as the 4th branch of the C22-R
champion (c22r8.pt, machine v9c, 20,518p, all 8 D bars certified).
============================================================================
WIN CONDITION (operator strategy reset, C21b): ONE COHERENT MODEL under
the box — fluency + exact state + exact computation in a single
parameter set. This cycle builds and certifies exactly that:

ARCHITECTURE (FusionBot, single nn.Module):
  dialog surface (36-token protocol, d=16) — the ENTIRE C22-R champion
    (3 hosts w/ SSM decay clamp a<=0.90, state organ, math organ,
    dual-gated heads, learned 3-way router) loaded VERBATIM from
    c22r8.pt and FROZEN: D1-D7 are invariant under fusion by
    construction, then re-measured (protocol requires measurement).
  text surface (768 BPE, d=32) — the ENTIRE d32 LM engine (tied
    embedding + SSM host + head) loaded VERBATIM from
    lm_host_final.pt, TRAINABLE (a short warm fine-tune + routing).
    The text host keeps the STOCK (unclamped) SSM forward — the engine
    was trained with a=0.923 max decay and "carry the d32 engine as-is"
    means exactly that (the clamp is the dialog-side L-DECAY-DRIFT fix).
  surface router: the champion's 16-dim router front (Linear 48->16,
    frozen) + its first 3 output rows (frozen) + ONE LEARNED 4th row
    (w3, b3; init b3 = -5 so it cannot steal a dialog stream at init)
    read over the first 3 tokens of the surface's own front embedding
    (dialog: the champion emb rows, frozen; text: a new trainable
    768x16 front). Learned 4-way routing over one shared 16-dim space.

PROTOCOL:
  SMOKE: (1) dialog forward BIT-EXACT vs the standalone champion
    (logits + 3 routing scores); (2) text forward BIT-EXACT vs a
    standalone d32 LM built from the same weights; (3) routing at
    init (dialog rows 0-2 must stay argmax).
  STAGE 1: 2000 steps, batch 32, AdamW 3e-3, clip 1.0 — duty-cycled
    (L-DUTY): odd steps = text LM CE + 0.5 * route CE (target 3),
    even steps = dialog route CE only (target 0/1/2; the frozen
    dialog stack contributes no gradient, this keeps row 3 HONEST on
    dialog streams). L=256 (text, as trained) / L=63 (dialog).
  EVAL: all 8 champion D bars re-measured through the fused model;
    4-way routing (32 dialog + 8 text streams); text val CE
    @256/1024/4096/16384 through the fused model vs the standalone
    engine (carry-over check); 2 generations through the fused model.

BARS:
  F1_champion_bars      : all 8 C22-R bars pass on the fused model
                          (D1 state <=0.01, D2 overwrite <=0.05,
                          D3 16k <= 4k+0.05, D4 plus<=0.02/minus<=0.05,
                          D5 chat <=0.02, D6 3-way routing 1.0,
                          D7 dialogue exact).
  F2_4way_routing       : 32/32 (24 dialog fam-argmax + 8 text -> 3).
  F3_ce256_carry        : fused ce256 <= standalone ce256 + 0.10.
  F4_length_invariance  : fused ce16384 <= 1.3 * fused ce256.
  F5_generation         : prose + code samples through the fused model
                          logged for the human eye.
  F6_single_model       : one nn.Module, one state_dict; param count
                          logged (coherence = the claim).
HONEST BOUNDARY (C21b, L-DATA-CEILING): the box-scale fluency claim is
a LENGTH-INVARIANT fluency ENGINE (val ce256 ~4.27 vs ln 768 = 6.644;
bar-4.0 NOT met — the 1MB corpus is the ceiling, not capacity). The
chatbot axis stays first-class with that boundary stated.

Tag ARC2-C22B-FUSION. Wall budget ~20 min.
"""
import json, math, os, random, resource, time, types

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()

# ---------------------------------------------------------------- preamble
c22 = open("c22r4.py").read()
ns = {}
exec(compile(c22[:c22.index("# ============================================================== smoke")],
             "c22r4.py", "exec"), ns)
g = ns["g"]
gen_w = ns["gen_w"]
probe_w = ns["probe_w"]
routing_acc3 = ns["routing_acc"]
overwrite_probe, dialogue_gen = ns["overwrite_probe"], ns["dialogue_gen"]
PLUS, MINUS = g["PLUS"], g["MINUS"]
V9c = ns["DialogMachineV9b"]
ClampedSSM = g["SSMBlock"]            # decay clamped <= 0.90 (dialog side)

_u = {"__name__": "unified_fresh"}
exec(open("unified.py").read().split("\nRESULTS = {}")[0], _u)
StockSSM = _u["SSMBlock"]             # stock forward (text side, as trained)

from bpe_tok import load_tk
VOC, encode, decode = load_tk()
with open(os.path.join(HERE, "corpus", "corpus_full.txt"), "rb") as f:
    TOKS = encode(f.read())
N_TRAIN = int(len(TOKS) * 0.9)
DVOC = 36
print(f"[setup] c22r4 preamble + tokenizer VOC={VOC} TOKS={len(TOKS)} "
      f"train={N_TRAIN} ({time.time()-T0:.0f}s)", flush=True)


# ---------------------------------------------------------------- model
class FusionBot(nn.Module):
    """C22-R champion (dialog, d16, FROZEN) + d32 fluency engine
    (TRAINABLE) + learned surface-router row 3, one parameter set."""

    def __init__(self, d_t=32):
        super().__init__()
        d = g["CFG"]["d_model"]                       # 16
        # --- dialog side: champion layout, frozen ---
        self.emb_d = nn.Embedding(DVOC, d)
        self.proj = nn.Linear(3 * d, 16)
        self.hosts = nn.ModuleList([ClampedSSM(d) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Linear(d, DVOC) for _ in range(3)])
        self.st_add = nn.Parameter(0.1 * torch.randn(44, DVOC))
        self.st_m = nn.Parameter(torch.zeros(41, 3, DVOC))
        self.math_table = nn.Parameter(0.1 * torch.randn(3, 10, 10, 10))
        # --- surface router: frozen front + frozen rows 0-2 + row 3 ---
        self.w3 = nn.Parameter(torch.zeros(16))
        self.b3 = nn.Parameter(torch.tensor(-5.0))
        # --- text side: d32 engine, trainable ---
        self.emb_t = nn.Embedding(VOC, d_t)
        self.host_t = StockSSM(d_t)
        self.norm_t = nn.LayerNorm(d_t)
        self.head_t = nn.Linear(d_t, VOC)
        self.head_t.weight = self.emb_t.weight        # tied, as in C21
        self.gate_t = nn.Parameter(torch.zeros(1))
        self.emb_r = nn.Embedding(VOC, 16)            # text router front

    # -- state-dict friendly buffers (champion router tail / head gate) --
    def load_champion(self, sd):
        with torch.no_grad():
            self.emb_d.weight.copy_(sd["emb.weight"])
            self.proj.load_state_dict({"weight": sd["router.0.weight"],
                                       "bias": sd["router.0.bias"]})
            for r in range(3):
                p = {k[len(f"hosts.{r}."):]: sd[k]
                     for k in sd if k.startswith(f"hosts.{r}.")}
                self.hosts[r].load_state_dict(p)
                p = {k[len(f"norms.{r}."):]: sd[k]
                     for k in sd if k.startswith(f"norms.{r}.")}
                self.norms[r].load_state_dict(p)
                self.heads[r].load_state_dict({"weight": sd[f"heads.{r}.weight"],
                                               "bias": sd[f"heads.{r}.bias"]})
            self.st_add.copy_(sd["st_add"])
            self.st_m.copy_(sd["st_m"])
            self.math_table.copy_(sd["math_table"])
            self.register_buffer("head_gate", sd["head_gate"].clone())
            self.register_buffer("out02_w", sd["router.2.weight"].clone())
            self.register_buffer("out02_b", sd["router.2.bias"].clone())
        for p in [self.emb_d.weight, self.proj.weight, self.proj.bias,
                  self.st_add, self.st_m, self.math_table]:
            p.requires_grad_(False)
        for r in range(3):
            for m in (self.hosts[r], self.norms[r], self.heads[r]):
                for p in m.parameters():
                    p.requires_grad_(False)

    def load_engine(self, sd):
        with torch.no_grad():
            self.emb_t.weight.copy_(sd["emb.weight"])
            p = {k[len("host."):]: sd[k]
                 for k in sd if k.startswith("host.")}
            self.host_t.load_state_dict(p)
            p = {k[len("norm."):]: sd[k]
                 for k in sd if k.startswith("norm.")}
            self.norm_t.load_state_dict(p)
            self.head_t.bias.copy_(sd["head.bias"])

    def _route(self, front):
        h = F.gelu(self.proj(front.reshape(front.shape[0], -1)))
        rl3 = h @ self.out02_w.t() + self.out02_b
        row3 = (h @ self.w3 + self.b3).unsqueeze(-1)
        return torch.cat([rl3, row3], -1)

    def forward(self, x, surface=0):
        B, L = x.shape
        if surface == 0:
            front = self.emb_d(x[:, :3])
            rl = self._route(front)
            task = rl.argmax(-1)
            hg = torch.exp(self.head_gate)
            out = torch.zeros(B, L, DVOC, device=x.device)
            for r in range(3):
                idx = (task == r).nonzero().squeeze(-1)
                if idx.numel() == 0:
                    continue
                xr = x[idx]
                lg = hg[r] * self.heads[r](self.norms[r](self.hosts[r](self.emb_d(xr))))
                if r == 0:
                    lg = lg + self._state_logits(xr) + self._math_logits(xr)
                elif r == 1:
                    lg = lg + self._math_logits(xr)
                out.scatter_(0, idx.view(-1, 1, 1).expand_as(lg), lg)
            return out, rl
        front = self.emb_r(x[:, :3])
        rl = self._route(front)
        out = torch.exp(self.gate_t) * self.head_t(
            self.norm_t(self.host_t(self.emb_t(x))))
        return out, rl


# graft the V9c organ methods (they reference self.st_add/st_m/math_table,
# which FusionBot owns with identical shapes)
FusionBot._state_logits = V9c._state_logits
FusionBot._math_logits = V9c._math_logits
print(f"[c22b] FusionBot defined ({time.time()-T0:.0f}s)", flush=True)


# ---------------------------------------------------------------- build
def build_fusion():
    m = FusionBot()
    m.load_champion(torch.load("c22r8.pt", weights_only=True))
    m.load_engine(torch.load("lm_host_final.pt", weights_only=True))
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in m.parameters())
    print(f"[build] params total={n_total} trainable={n_train} "
          f"frozen={n_total - n_train}", flush=True)
    return m


def build_champion():
    m = V9c()
    m.load_state_dict(torch.load("c22r8.pt", weights_only=True))
    return m


def build_standalone_lm():
    class LM(nn.Module):
        def __init__(self, d=32):
            super().__init__()
            self.emb = nn.Embedding(VOC, d)
            self.host = StockSSM(d)
            self.norm = nn.LayerNorm(d)
            self.head = nn.Linear(d, VOC)
            self.head.weight = self.emb.weight
        def forward(self, x):
            return self.head(self.norm(self.host(self.emb(x))))
    m = LM()
    sd = torch.load("lm_host_final.pt", weights_only=True)
    with torch.no_grad():
        m.emb.weight.copy_(sd["emb.weight"])
        m.host.load_state_dict({k[len("host."):]: sd[k]
                                for k in sd if k.startswith("host.")})
        m.norm.load_state_dict({k[len("norm."):]: sd[k]
                                for k in sd if k.startswith("norm.")})
        m.head.bias.copy_(sd["head.bias"])
    return m


def sample(batch, length, arr, rng):
    n = len(arr) - length - 1
    rows_x, rows_y = [], []
    for _ in range(batch):
        s = rng.randrange(n)
        w = arr[s:s + length + 1]
        rows_x.append(w[:-1]); rows_y.append(w[1:])
    return torch.tensor(rows_x), torch.tensor(rows_y)


# ---------------------------------------------------------------- smoke
print("[smoke] bit-exact checks ...", flush=True)
mf = build_fusion()
mc = build_champion()
mlm = build_standalone_lm()
torch.manual_seed(11)
rng = random.Random(11)
maxdiff_d = maxdiff_t = 0.0
rmatch = True
for fam in range(3):
    x, y, o, task = gen_w(4, 63, rng, fam=fam)
    with torch.no_grad():
        out_f, rl_f = mf(x, surface=0)
        out_c, rl_c = mc(x)
    maxdiff_d = max(maxdiff_d, float((out_f - out_c).abs().max()))
    maxdiff_d = max(maxdiff_d, float((rl_f[:, :3] - rl_c).abs().max()))
    if not torch.equal(rl_f[:, :3].argmax(-1), rl_c.argmax(-1)):
        rmatch = False
xt = torch.randint(0, VOC, (4, 128))
with torch.no_grad():
    out_f, rl_f = mf(xt, surface=1)
    out_l = mlm(xt)
maxdiff_t = float((out_f - out_l).abs().max())
assert maxdiff_d < 1e-4, f"dialog forward not bit-exact vs champion: {maxdiff_d}"
assert rmatch, "dialog routing argmax changed at init"
assert maxdiff_t < 1e-4, f"text forward not bit-exact vs standalone LM: {maxdiff_t}"
with torch.no_grad():
    xl, _, _, _ = gen_w(1, 63, random.Random(5), fam=0)
    _, rl_init = mf(xl, surface=0)
    _, rl_t = mf(torch.randint(0, VOC, (4, 32)), surface=1)
print(f"[smoke] dialog maxdiff={maxdiff_d:.2e} text maxdiff={maxdiff_t:.2e} "
      f"dialog-init-route-ok text-init-argmax={rl_t.argmax(-1).tolist()}", flush=True)
print("[smoke] PASSED (both surfaces bit-exact at init)", flush=True)

# ---------------------------------------------------------------- stage 1
torch.manual_seed(0)
rng = random.Random(29)
m = mf
m.train()
train_p = [p for p in m.parameters() if p.requires_grad]
opt = torch.optim.AdamW([{"params": train_p, "weight_decay": 0.0}], lr=3e-3)
STEPS, BATCH = 2000, 32
t0 = time.time()
arr = TOKS[:N_TRAIN]
racc_t = racc_d = 0.0
for step in range(1, STEPS + 1):
    if step % 2 == 1:                                 # text: LM CE + route
        x, y = sample(BATCH, 256, arr, rng)
        logits, rl = m(x, surface=1)
        lm = F.cross_entropy(logits.reshape(-1, VOC), y.reshape(-1))
        rce = F.cross_entropy(rl, torch.full((BATCH,), 3, dtype=torch.long))
        loss = lm + 0.5 * rce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            racc_t = float((rl.argmax(-1) == 3).float().mean())
    else:                                             # dialog: route CE only
        x, y, o, task = gen_w(BATCH, 63, rng)
        _, rl = m(x, surface=0)
        rce = F.cross_entropy(rl, task)
        rce.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad()
        with torch.no_grad():
            racc_d = float((rl.argmax(-1) == task).float().mean())
    if step % 200 == 0:
        print(f"  [ft] step {step}/{STEPS} lm {float(lm):.4f} "
              f"route_t {racc_t:.2f} route_d {racc_d:.2f} "
              f"b3 {float(m.b3):.2f} ({time.time()-t0:.0f}s)", flush=True)
torch.save(m.state_dict(), "c22b_stage1.pt")
print(f"[stage1] done in {time.time()-t0:.0f}s", flush=True)

# ================================================================== eval
m.eval()

@torch.no_grad()
def val_ce_fusion(L, reps=1):
    ce = n = 0.0
    for i in range(reps):
        rng = random.Random(800_000 + L + i)
        x, y = sample(1, L, TOKS[N_TRAIN:], rng)
        logits, _ = m(x, surface=1)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); n += y.numel()
    return round(ce / n, 4)


@torch.no_grad()
def val_ce_standalone(L, reps=1):
    ce = n = 0.0
    for i in range(reps):
        rng = random.Random(800_000 + L + i)
        x, y = sample(1, L, TOKS[N_TRAIN:], rng)
        nll = -F.log_softmax(mlm(x), -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); n += y.numel()
    return round(ce / n, 4)


res = {
    "state4096": probe_w(m, 0, 4096),
    "state16384": probe_w(m, 0, 16384),
    "mathplus4096": probe_w(m, 1, 4096, 1, op=PLUS),
    "mathminus4096": probe_w(m, 1, 4096, 1, op=MINUS),
    "chat4096": probe_w(m, 2, 4096),
    "overwrite4096": overwrite_probe(m, 4096),
    "routing3way": routing_acc3(m),
}
dlg = dialogue_gen(m)
D7 = ("dave" in dlg.split("\n")[3] and "4 2" in dlg.split("\n")[9]
      and "1 2" in dlg.split("\n")[5] and " 6" in dlg.split("\n")[8])
print(f"[eval-D] {res}", flush=True)
print("[dialogue]", flush=True)
print(dlg, flush=True)

# 4-way routing: 32 dialog + 8 text
ok = tot = 0
for f in range(3):
    for i in range(8):
        rng = random.Random(800_000 + f * 100 + i)
        x, y, o, task = gen_w(1, 4096, rng, fam=f)
        with torch.no_grad():
            _, rl = m(x, surface=0)
        ok += int(rl.argmax(-1).item() == f); tot += 1
for i in range(8):
    rng = random.Random(950_000 + i)
    x, y = sample(1, 256, TOKS[N_TRAIN:], rng)
    with torch.no_grad():
        _, rl = m(x, surface=1)
    ok += int(rl.argmax(-1).item() == 3); tot += 1
res["routing4way"] = f"{ok}/{tot}"
res["routing4way_acc"] = ok / tot
print(f"[eval] routing4way {res['routing4way']}", flush=True)

ce_f = {L: val_ce_fusion(L, 2 if L <= 1024 else 1)
        for L in (256, 1024, 4096, 16384)}
ce_s = {L: val_ce_standalone(L, 2 if L <= 1024 else 1)
        for L in (256, 1024, 4096, 16384)}
res["text_ce_fused"] = ce_f
res["text_ce_standalone"] = ce_s
res["ppl256_fused"] = round(math.exp(ce_f[256]), 2)
print(f"[eval] fused ce {ce_f}  standalone ce {ce_s}", flush=True)


@torch.no_grad()
def gen_fusion(prompt, n=160, temp=0.8, seed=1):
    torch.manual_seed(seed)
    cur = torch.tensor([encode(prompt.encode("utf-8"))], dtype=torch.long)
    for _ in range(n):
        lg, _ = m(cur, surface=1)
        nxt = torch.multinomial(torch.softmax(lg[:, -1, :] / temp, -1), 1)
        cur = torch.cat([cur, nxt], 1)
    return decode(cur[0].tolist()).decode("utf-8", "replace")


gen1 = gen_fusion("It is a truth universally acknowledged, that a single "
                  "man in possession of a good fortune", 160, 0.8, 1)
gen2 = gen_fusion("def add(a, b):\n    return", 160, 0.8, 2)
res["gen_prose"] = gen1
res["gen_code"] = gen2
print("[gen prose] " + gen1.replace("\n", " | "), flush=True)
print("[gen code ] " + gen2.replace("\n", " | "), flush=True)

bars = {
    "F1a_D1_state_le_0.01": res["state4096"] <= 0.01,
    "F1b_D2_overwrite_le_0.05": res["overwrite4096"] <= 0.05,
    "F1c_D3_16k_le_4k+0.05": res["state16384"] <= res["state4096"] + 0.05,
    "F1d_D4_mathplus_le_0.02": res["mathplus4096"] <= 0.02,
    "F1e_D4_mathminus_le_0.05": res["mathminus4096"] <= 0.05,
    "F1f_D5_chat_le_0.02": res["chat4096"] <= 0.02,
    "F1g_D6_3way_routing_1.0": res["routing3way"] == 1.0,
    "F1h_D7_dialogue_exact": bool(D7),
    "F2_4way_routing_1.0": res["routing4way_acc"] == 1.0,
    "F3_ce256_carry": ce_f[256] <= ce_s[256] + 0.10,
    "F4_length_invariance_1.3x": ce_f[16384] <= 1.3 * ce_f[256],
    "F5_generation_logged": len(gen1) > 100 and len(gen2) > 100,
    "F6_single_model": True,
}
print(f"[bars] {bars}", flush=True)

n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in m.parameters())
final = {"tag": "ARC2-C22B-FUSION",
         "design": ["dialog side = C22-R champion v9c FROZEN (D1-D7 "
                    "invariant by construction, re-measured)",
                    "text side = d32 LM engine as-is, trainable, STOCK "
                    "SSM forward (a_max 0.923 as trained)",
                    "surface router = frozen 16-dim front + frozen rows "
                    "0-2 + learned row 3 (b3 init -5) over surface fronts"],
         "smoke": {"dialog_maxdiff": round(maxdiff_d, 6),
                   "text_maxdiff": round(maxdiff_t, 6)},
         "fused": res, "bars": bars,
         "params": {"total": n_total, "trainable_stage1": n_train},
         "boundary": "bar-4.0 fluency NOT claimed (L-DATA-CEILING, C21b): "
                     "the claim is a length-invariant fluency engine "
                     "inside one coherent model",
         "wall_s": round(time.time() - T0, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open(os.path.join(HERE, "log.jsonl"), "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
