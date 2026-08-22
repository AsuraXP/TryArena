"""
ARC-2 CYCLE 22 / THE CHATBOT MACHINE (MACHINE v8)
=====================================================================
Operator directive: the machine must "nail and ace it as a chatbot" —
not limited to structured-reasoning tasks. C22 = the first fused
CONVERSATION machine: one artifact that (a) remembers facts across a
multi-turn dialogue (with overwrites), (b) does exact arithmetic in
conversation (plus / mod-10 minus), (c) echoes small talk — all on one
shared token stream, routed per example, certified against the exact
analytic oracle at 4096 AND 16384. Fluency axis = C21 (LM host,
separate run); branch fusion of the fluency engine is C22b.

Search (2026-08-22, phase-1): nearest prior lines = content-addressed
SRAM readout (this repo, ICL cert 0.0) and the v6/v7 query-keyed
readout organ (L-QUERY-READOUT: answer = f(exact STATE, QUERY)).
No prior work: mechanism-computed conversational SLOTS with
query-keyed exact readout as a routed organ in a heterogeneous
conversation machine, certified with the exact analytic oracle.

DIALOGUE SURFACE (own alphabet, VOCAB=36, disjoint from the 79-vocab
reasoning machine and the 768-vocab LM host — L-ORGAN-ALPHABET):
  0-9 digits | 10 U | 11 A | 12 my 13 name 14 is 15 now 16 code
  17 what 18 the 19 ok 20 fine 21 good 22 tell 23 me 24 it 25 and
  26 plus 27 minus | 28-35 N0..N7 (alice bob carol dave eve frank
  grace heidi)
TURNS:
  fill   : U ok A ok | U fine A fine | U good A good
           U tell me the and it A it
  f-name : U my name is Nj A ok
  f-now  : U my name now is Nj A ok            (OVERWRITE)
  f-code : U my code is d1 d2 A ok
  q-name : U what is my name  A Nj
  q-code : U what is my code  A d1 d2
  math   : U what is a plus b  A s1 s2   (s = a+b; s1=s//10, s2=s%10)
           U what is a minus b A (a-b) mod 10
FAMILIES (per-example router on first 3 tokens):
  0 state : fact, fact (either order), then a weighted mix
            (fill/q-name/q-code/overwrite/recode), final q-name + q-code
            guaranteed when they fit
  1 math  : plus / minus turns
  2 chat  : fill turns
Oracle = exact generation entropy (house convention): A-turn tokens 0
(exact answers); random U values get their choice entropy: name ln8,
digit ln10, op ln2, fill variant ln4 (chat family), first-fact-order
ln2, mid-turn-kind H8 on the first value token (state family).
dCE = (CE - oracle)/n; the A-turn (answer) tokens are the certification.

MACHINE v8 (3 branches, ~32k params):
  r0 state : SSM host d16 + STATE ORGAN. Exact state = (NAME 8-hot,
     CODE tens 10-hot, CODE ones 20 joint (value x answer-position),
     set flags). Slot updates computed by mechanism (pattern state
     machine over the token stream — same paradigm as the carry organ).
     Readout = additive table A (44 rows, 0.1-randn) + state x query
     BILINEAR M (41x3, zero-init = L-GATE-INIT on the interaction;
     inputs are exact features => full-rank gradient from step 1).
     Expressibility (verified by construction):
       q-name ans : M[name-j, qname] = C*onehot(Nj)
       q-code tens: M[d1-j, qcode] = +C*onehot(j);
                    M[d2j(k,0), qcode] = -C on all 10 digits
       q-code ones: M[d2j(k,1), qcode] = -C on all digits + 2C*onehot(k)
       => one-hot answers at every position for all name x code states.
  r1 math  : SSM host d16 + MATH ORGAN: exact table T (3x10x10x10,
     0.1-randn) keyed (case, a, b); case 0/1 = plus tens/ones,
     2 = minus (single-borrow = the C23 "borrow organ" pulled forward:
     conversation math needs it). (a, b) tracked by mechanism.
  r2 chat  : SSM host d16 only (finite-state 4-way echo; cf. the
     proven host-only mod7 branch).
  shared: emb(36,16), learned router MLP(first-3 -> 3) direct-CE
  (L-DIRECT-GRADIENT), per-branch zero-init heads + exp head-gates
  (L-DUAL-GATE).
Protocol: 12000 cycling steps (4000/family), batch 32, L=63, AdamW
3e-3, clip 1.0, seed 0, ckpts 3/6/9/12k; per ckpt: state/math-plus/
math-minus/chat @4096 stream dCE, overwrite probe @4096, state+math
@16384, head gates, organ mass; one logged greedy dialogue at final.
NO TF arm (operator directive C17).
SUCCESS = (D1) state @4096 <= 0.01 at 12k; (D2) overwrite final-name
  CE @4096 <= 0.05; (D3) state @16384 <= state @4096 + 0.05;
  (D4) math-plus @4096 <= 0.02, math-minus @4096 <= 0.05;
  (D5) chat @4096 <= 0.02; (D6) routing 1.0 all three families;
  (D7) logged dialogue reproduces all facts + math exactly.
USAGE: OMP_NUM_THREADS=1 python3 -u dialog_chat.py   (RESUME=1 to resume)
"""
import json, math, os, random, resource, time
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)
t_start = time.time()

g = {"__name__": "u4"}
exec(open("unified_add.py").read().split("\nRESULTS = {}")[0], g)
SSMBlock, n_params = g["SSMBlock"], g["n_params"]

# ----------------------------------------------------------------- vocab
DIG0, U, A = 0, 10, 11
MY, NAME, IS, NOW, CODE, WHAT, THE = 12, 13, 14, 15, 16, 17, 18
OK, FINE, GOOD, TELL, ME, IT, AND = 19, 20, 21, 22, 23, 24, 25
PLUS, MINUS = 26, 27
N0 = 28                      # names N0..N7
VOCAB = 36
NAMES = ["alice", "bob", "carol", "dave", "eve", "frank", "grace", "heidi"]
LN2, LN4, LN8, LN10 = (math.log(2.0), math.log(4.0), math.log(8.0),
                       math.log(10.0))
MIX_W = (45, 18, 18, 10, 9)                 # fill/qname/qcode/fnow/fcode
H8 = -sum((c / 100) * math.log(c / 100) for c in MIX_W)
CFG = dict(steps=12000, batch=32, train_len=63,
           ckpts=[3000, 6000, 9000, 12000], d_model=16)
print(f"[setup] dialog-chat (machine v8, chatbot) cfg={CFG} VOCAB={VOCAB}",
      flush=True)

# ----------------------------------------------------------------- data
def _fill_turn(rng):
    w = rng.choice([OK, FINE, GOOD, TELL])
    if w == TELL:
        u, a = [TELL, ME, THE, AND, IT], [IT]
    else:
        u, a = [w], [w]
    return u, a

def _fname_turn(rng, now=False):
    nj = N0 + rng.randrange(8)
    u = [MY, NAME, IS, nj] if not now else [MY, NOW, NAME, IS, nj]
    return u, [OK]

def _fcode_turn(rng):
    d1, d2 = rng.randrange(10), rng.randrange(10)
    return [MY, CODE, IS, d1, d2], [OK]

def _qname_turn(nj):
    return [WHAT, IS, MY, NAME], [nj]

def _qcode_turn(d1, d2):
    return [WHAT, IS, MY, CODE], [d1, d2]

def _math_turn(rng, op=None):
    a, b = rng.randrange(10), rng.randrange(10)
    op = op if op is not None else rng.choice([PLUS, MINUS])
    s = a + b if op == PLUS else (a - b) % 10
    ans = [s // 10, s % 10] if op == PLUS else [s]
    return [WHAT, IS, a, op, b], ans

def _emit(x, nll, u, a, ent_u):
    """append [U] + u + [A] + a with entropy ent_u aligned to u."""
    assert len(ent_u) == len(u), f"ent {len(ent_u)} != u {len(u)}"
    x += [U] + u + [A] + a
    nll += [0.0] + ent_u + [0.0] + [0.0] * len(a)

def _build_state(rng, length):
    """facts first (either order), weighted mix, final guaranteed queries."""
    name = N0 + rng.randrange(8)
    code = [rng.randrange(10), rng.randrange(10)]
    x, nll = [], []
    order = [(_fname_turn, [0.0, LN2, 0.0, LN8]),
             (_fcode_turn, [0.0, LN2, 0.0, LN10, LN10])][rng.randrange(2)]
    # first fact: ln2 order-entropy on the NAME/CODE token (3rd value pos)
    u, _ = order[0](rng)
    _emit(x, nll, u, [OK], order[1])
    if order[0] is _fname_turn:
        u, _ = _fcode_turn(rng)
        _emit(x, nll, u, [OK], [0, 0, 0, LN10, LN10])
    else:
        u, _ = _fname_turn(rng)
        _emit(x, nll, u, [OK], [0, 0, 0, LN8])
    kinds = (["fill"] * MIX_W[0] + ["qname"] * MIX_W[1] + ["qcode"] * MIX_W[2]
             + ["fnamenow"] * MIX_W[3] + ["fcode"] * MIX_W[4])
    while len(x) < length - 14:            # +15 (qname 7 + qcode 8) >= length+1
        kind = rng.choice(kinds)
        if kind == "fill":
            u, a = _fill_turn(rng)
            ent = [H8] + [0.0] * (len(u) - 1)
        elif kind == "qname":
            u, a = _qname_turn(name)
            ent = [H8] + [0.0] * (len(u) - 1)
        elif kind == "qcode":
            u, a = _qcode_turn(*code)
            ent = [H8] + [0.0] * (len(u) - 1)
        elif kind == "fnamenow":
            u, a = _fname_turn(rng, now=True)
            name = u[-1]
            ent = [H8] + [0.0, 0.0, 0.0, LN8]
        else:
            u, a = _fcode_turn(rng)
            code = [u[3], u[4]]
            ent = [H8] + [0.0, 0.0, LN10, LN10]
        _emit(x, nll, u, a, ent)
    u, a = _qname_turn(name)
    _emit(x, nll, u, a, [0.0] * len(u))
    u, a = _qcode_turn(*code)
    _emit(x, nll, u, a, [0.0] * len(u))
    assert len(x) >= length + 1
    return x, nll                    # full stream; gen_dialogue_t slices

def gen_dialogue_t(batch, length, rng, fam=None, op=None):
    xs, ys, os_, tasks = [], [], [], []
    for i in range(batch):
        f = fam if fam is not None else i % 3
        if f == 0:
            x, nll = _build_state(rng, length)
        elif f == 1:
            x, nll = [], []
            while len(x) < length + 1:
                u, a = _math_turn(rng, op)
                _emit(x, nll, u, a, [0.0, 0.0, LN10, LN2, LN10])
        else:
            x, nll = [], []
            while len(x) < length + 1:
                u, a = _fill_turn(rng)
                _emit(x, nll, u, a, [LN4] + [0.0] * (len(u) - 1))
        xs.append(x[:length]); ys.append(x[1:length + 1])
        os_.append(nll[1:length + 1])
        tasks.append(f)
    return (torch.tensor(xs), torch.tensor(ys), torch.tensor(os_),
            torch.tensor(tasks))

# ----------------------------------------------------------------- model
class DialogMachine(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG["d_model"]
        self.emb = nn.Embedding(VOCAB, d)
        nn.init.normal_(self.emb.weight, std=0.02)
        self.router = nn.Sequential(nn.Linear(3 * d, 16), nn.GELU(),
                                    nn.Linear(16, 3))
        self.hosts = nn.ModuleList([SSMBlock(d) for _ in range(3)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(3)])
        self.heads = nn.ModuleList([nn.Linear(d, VOCAB) for _ in range(3)])
        for h in self.heads:                              # L-GATE-INIT
            nn.init.zeros_(h.weight); nn.init.zeros_(h.bias)
        self.head_gate = nn.Parameter(torch.zeros(3))     # L-DUAL-GATE
        # state organ additive rows: 0-7 name | 8-17 d1 | 18-37 d2joint
        # 38 base | 39 setname | 40 setcode | 41 qnone | 42 qname | 43 qcode
        self.st_add = nn.Parameter(0.1 * torch.randn(44, VOCAB))
        # f (41): name 8 | d1 10 | d2joint 20 | setname 1 | setcode 1
        # qo (3): none | qname | qcode
        self.st_m = nn.Parameter(torch.zeros(41, 3, VOCAB))
        # math organ: (case, a, b) -> digit-10 logits
        self.math_table = nn.Parameter(0.1 * torch.randn(3, 10, 10, 10))

    def _state_logits(self, x, dbg=False):
        B, L = x.shape
        f = torch.zeros(B, L, 41)
        qo = torch.zeros(B, L, 3)
        idx = torch.full((B, L, 6), 38, dtype=torch.long)
        for b in range(B):
            name, d1v, d2v = -1, -1, -1
            hist = []
            qans = 0                     # 1=qname next | 2=tens | 3=ones
            for t in range(L):
                tok = int(x[b, t])
                # ---- emit (pre-update)
                if name >= 0:
                    f[b, t, name] = 1.0
                    idx[b, t, 0] = name
                    idx[b, t, 3] = 39
                if d1v >= 0:
                    f[b, t, 8 + d1v] = 1.0
                    idx[b, t, 1] = 8 + d1v
                    idx[b, t, 4] = 40
                if qans in (2, 3) and d2v >= 0:
                    pos = 0 if qans == 2 else 1
                    f[b, t, 18 + d2v * 2 + pos] = 1.0
                    idx[b, t, 2] = 18 + d2v * 2 + pos
                qi = 0 if qans == 0 else (1 if qans == 1 else 2)
                qo[b, t, qi] = 1.0
                idx[b, t, 5] = 41 + qi
                # ---- process
                h = hist
                if N0 <= tok <= N0 + 7 and len(h) >= 3 and h[-1] == IS \
                        and h[-2] == NAME and h[-3] in (MY, NOW):
                    name = tok - N0
                elif tok < 10 and len(h) >= 4 and h[-1] < 10 \
                        and h[-2] == IS and h[-3] == CODE and h[-4] == MY:
                    d1v, d2v = h[-1], tok
                elif tok == A:
                    if len(h) >= 4 and h[-1] == NAME and h[-2] == MY \
                            and h[-3] == IS and h[-4] == WHAT:
                        qans = 1
                    elif len(h) >= 4 and h[-1] == CODE and h[-2] == MY \
                            and h[-3] == IS and h[-4] == WHAT:
                        qans = 2
                # ---- consume answer positions (after the A token)
                if qans == 1 and tok != A:
                    qans = 0
                elif qans == 2 and tok != A:
                    qans = 3
                elif qans == 3 and tok != A:
                    qans = 0
                hist = (hist + [tok])[-4:]
        Aadd = (self.st_add[idx[:, :, 0]] + self.st_add[idx[:, :, 1]]
                + self.st_add[idx[:, :, 2]] + self.st_add[idx[:, :, 3]]
                + self.st_add[idx[:, :, 4]] + self.st_add[idx[:, :, 5]])
        if dbg:
            return Aadd, f, qo, idx
        return Aadd + torch.einsum("blm,bln,mnv->blv", f, qo, self.st_m)

    def _math_logits(self, x):
        B, L = x.shape
        out = torch.zeros(B, L, VOCAB, device=x.device)
        T = self.math_table
        for b in range(B):
            hist = []
            ma, mop = -1, -1
            mact, mp, mpend = False, 0, []
            ma_s = mb_s = 0
            mplus = False
            for t in range(L):
                tok = int(x[b, t])
                if mact and tok != A and mp < len(mpend):
                    case = mp if mplus else 2
                    out[b, t, :10] = T[case, ma_s, mb_s]
                    mp += 1
                    if mp >= len(mpend):
                        mact = False
                if tok < 10 and len(hist) >= 2 and hist[-1] == IS \
                        and hist[-2] == WHAT and mop < 0:
                    ma = tok
                elif tok in (PLUS, MINUS) and ma >= 0:
                    mop = tok
                    mplus = (tok == PLUS)
                elif tok < 10 and mop >= 0:
                    s = ma + tok if mop == PLUS else (ma - tok) % 10
                    ma_s, mb_s = ma, tok
                    mpend = [s // 10, s % 10] if mop == PLUS else [s]
                    mact, mp = True, 0
                    ma, mop = -1, -1
                elif tok == A and mact:
                    mp = 0
                hist = (hist + [tok])[-4:]
        return out

    def forward(self, x):
        B, L = x.shape
        rl = self.router(self.emb(x[:, :3]).reshape(B, -1))
        task = rl.argmax(-1)
        hg = torch.exp(self.head_gate)
        out = torch.zeros(B, L, VOCAB, device=x.device)
        for r in range(3):
            idx = (task == r).nonzero().squeeze(-1)
            if idx.numel() == 0:
                continue
            xr = x[idx]
            hr = self.norms[r](self.hosts[r](self.emb(xr)))
            lg = hg[r] * self.heads[r](hr)
            if r == 0:
                lg = lg + self._state_logits(xr)
            elif r == 1:
                lg = lg + self._math_logits(xr)
            out.scatter_(0, idx.view(-1, 1, 1).expand_as(lg), lg)
        return out, rl

# ----------------------------------------------------------------- training
def train_step(model, opt, x, y, task):
    logits, rl = model(x)
    l_lm = F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
    l_rt = F.cross_entropy(rl, task)
    loss = l_lm + 0.5 * l_rt
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad()
    return l_lm.item(), l_rt.item()

@torch.no_grad()
def stream_probe(model, fam, L, reps=1, op=None):
    """whole-stream dCE for one family at lens L."""
    model.eval()
    ce = orc = n = 0.0
    for i in range(reps):
        rng = random.Random(900_000 + L + fam * 100 + (7 if op else 0) + i)
        x, y, o, _ = gen_dialogue_t(1, L, rng, fam=fam, op=op)
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
    return round((ce - orc) / n, 4)

@torch.no_grad()
def overwrite_probe(model, L, reps=1):
    """name=N0, 3 fills, name-NOW=N7, fills, q-name -> final name CE."""
    model.eval()
    ce = n = 0.0
    for i in range(reps):
        rng = random.Random(920_000 + L + i)
        x = [U, MY, NAME, IS, N0, A, OK,
             U, OK, A, OK, U, FINE, A, FINE, U, GOOD, A, GOOD]
        x += [U, MY, NOW, NAME, IS, N0 + 7, A, OK]
        while len(x) < L - 7:
            w = rng.choice([OK, FINE, GOOD])
            x += [U, w, A, w]
        x = x[:L - 7]
        x += [U, WHAT, IS, MY, NAME, A, N0 + 7, U]   # len = L+1; name @ L-1
        xt = torch.tensor(x[:L])
        logits, rl = model(xt.unsqueeze(0))
        # logits[L-2] predicts x[L-1] = the (overwritten) name
        ce += float(-F.log_softmax(logits[0, L - 2], -1)[N0 + 7]); n += 1
    return round(ce / n, 4)

TMAP = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7",
        8: "8", 9: "9", U: "U", A: "A", MY: "my", NAME: "name", IS: "is",
        NOW: "now", CODE: "code", WHAT: "what", THE: "the", OK: "ok",
        FINE: "fine", GOOD: "good", TELL: "tell", ME: "me", IT: "it",
        AND: "and", PLUS: "plus", MINUS: "minus"}
TMAP.update({N0 + i: n for i, n in enumerate(NAMES)})

@torch.no_grad()
def dialogue_gen(model, L=1024):
    """Greedy conversation: feed user turns, model completes A turns."""
    model.eval()
    script = [(MY, NAME, IS, N0 + 3),        # my name is dave
              (MY, CODE, IS, 4, 2),          # my code is 4 2
              (OK,),                          # ok
              (WHAT, IS, MY, NAME),          # -> dave
              (TELL, ME, THE, AND, IT),      # -> it
              (WHAT, IS, 7, PLUS, 5),        # -> 1 2
              (MY, NOW, NAME, IS, N0 + 1),   # my name now is bob
              (FINE,),                        # fine
              (WHAT, IS, 3, MINUS, 7),       # -> 6
              (WHAT, IS, MY, CODE)]          # -> 4 2
    x, lines = [], []
    for turn in script:
        cur = [U] + list(turn)
        x += cur
        got = []
        for _ in range(3):
            xt = torch.tensor(x[:L])
            logits, _ = model(xt.unsqueeze(0))
            nxt = int(logits[0, -1].argmax())
            if nxt == U or nxt == A and got:
                break
            x.append(nxt); got.append(nxt)
        lines.append((cur[1:], got))
    return "\n".join(f"  U: {' '.join(TMAP[t] for t in cu):<22}"
                     f"| A: {' '.join(TMAP[t] for t in ga)}"
                     for cu, ga in lines)

def eval_all(model, tag, results):
    r = {"state4096": stream_probe(model, 0, 4096, 1),
         "mathplus4096": stream_probe(model, 1, 4096, 1, op=PLUS),
         "mathminus4096": stream_probe(model, 1, 4096, 1, op=MINUS),
         "chat4096": stream_probe(model, 2, 4096, 1),
         "overwrite4096": overwrite_probe(model, 4096, 1),
         "state16384": stream_probe(model, 0, 16384, 1),
         "math16384": stream_probe(model, 1, 16384, 1),
         "head_gates": [round(float(v), 3) for v in torch.exp(model.head_gate)],
         "st_m_abs": round(float(model.st_m.abs().sum()), 1)}
    results[tag] = r
    print(f"  {tag}: {r}", flush=True)

RESULTS = {}
torch.manual_seed(0)
m = DialogMachine()
print(f"[arm] machine v8 (chatbot) params={n_params(m)}", flush=True)

# ----------------------------------------------------------------- smoke
if os.environ.get("SMOKE") == "1":
    print("[smoke] wiring checks ...", flush=True)
    x = torch.tensor([[U, MY, NAME, IS, N0 + 3, A, OK, U, MY, CODE, IS, 4, 7,
                       A, OK, U, WHAT, IS, MY, NAME, A, N0 + 3,
                       U, WHAT, IS, MY, CODE, A, 4, 7,
                       U, OK, A, OK]])
    _, f, qo, idx = m._state_logits(x, dbg=True)
    # seq: 0U 1my 2name 3is 4N3 5A 6ok | 7U 8my 9code 10is 11=4 12=7 13A 14ok
    #      | 15U 16what 17is 18my 19name 20A 21N3 (q-name ANSWER)
    #      | 22U 23what 24is 25my 26code 27A 28=4 29=7 (tens, ones)
    assert float(f[0, 21, 3]) == 1.0, "name-j active at q-name answer pos"
    assert float(qo[0, 21, 1]) == 1.0, "qname query at answer pos"
    assert int(idx[0, 21, 0]) == 3, "idx c0 = name row"
    assert float(f[0, 28, 12]) == 1.0, "d1-j active at code tens pos"
    assert float(f[0, 28, 18 + 7 * 2 + 0]) == 1.0, "d2joint(k=7,pos0) tens"
    assert float(f[0, 29, 18 + 7 * 2 + 1]) == 1.0, "d2joint(k=7,pos1) ones"
    assert float(qo[0, 28, 2]) == 1.0 and float(qo[0, 29, 2]) == 1.0, \
        "qcode query at both answer positions"
    assert float(f[0, 1, 3]) == 0.0, "no name feat before name turn"
    print("[smoke] state-organ wiring OK", flush=True)
    xm = torch.tensor([[U, WHAT, IS, 7, PLUS, 5, A, 1, 2,
                        U, WHAT, IS, 3, MINUS, 7, A, 6]])
    # 0U 1what 2is 3=7 4plus 5=5 6A 7=s1 8=s2 | 9U 10what 11is 12=3 13minus
    # 14=7 15A 16=ans
    lm_ = m._math_logits(xm)
    assert float(lm_[0, 7].abs().sum()) > 0 and float(lm_[0, 8].abs().sum()) > 0 \
        and float(lm_[0, 16].abs().sum()) > 0, "answer rows active"
    for p in [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15]:
        assert float(lm_[0, p].abs().sum()) == 0.0, f"no logits at pos {p}"
    print("[smoke] math-organ wiring OK", flush=True)
    xs, ys, os_, tk = gen_dialogue_t(8, 63, random.Random(7))
    assert xs.shape == (8, 63) and ys.shape == (8, 63) and os_.shape == (8, 63) \
        and tk.shape == (8,)
    x2, n2 = _build_state(random.Random(3), 63)
    assert len(x2) >= 64 and len(n2) == len(x2)
    print(f"[smoke] data OK fams={tk.tolist()} sample="
          f"{' '.join(TMAP[int(t)] for t in xs[0][:26])}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    t0 = time.time()
    for s in range(30):
        x, y, o, task = gen_dialogue_t(4, 63, random.Random(90 + s))
        train_step(m, opt, x, y, task)
    dt = (time.time() - t0) / 30
    print(f"[smoke] ~{dt*1000:.0f} ms/step (b4) -> ~{dt*1000*8/1000/60:.0f} min "
          f"est for 12k steps (b32)", flush=True)
    print("[smoke] PASSED", flush=True)
    raise SystemExit(0)

# ----------------------------------------------------------------- run
m.train()
opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
rng = random.Random(17)
t0 = time.time()
last = 0
if os.environ.get("RESUME") == "1":
    cands = sorted(int(c[:-3]) for c in os.listdir(".")
                   if c.startswith("dialog_chat_") and c.endswith(".pt"))
    if cands:
        last = max(cands)
        m.load_state_dict(torch.load(f"dialog_chat_{last}.pt"))
        for s in range(1, last + 1):
            gen_dialogue_t(CFG["batch"], CFG["train_len"], rng)
        print(f"[v8] RESUME from step {last} (data rng fast-forwarded)",
              flush=True)
ckpts_done = set()
for step in range(last + 1, CFG["steps"] + 1):
    x, y, o, task = gen_dialogue_t(CFG["batch"], CFG["train_len"], rng)
    lm, rt = train_step(m, opt, x, y, task)
    if step in CFG["ckpts"] and step not in ckpts_done:
        torch.save(m.state_dict(), f"dialog_chat_{step}.pt")
        ckpts_done.add(step)
        print(f"    [v8] checkpoint at step {step}", flush=True)
    if step % 2000 == 0:
        print(f"  [v8 s0] step {step}/{CFG['steps']} lm {lm:.4f} rt {rt:.4f} "
              f"head_gates {torch.exp(m.head_gate).tolist()} "
              f"st_m_abs {float(m.st_m.abs().sum()):.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"[v8] trained in {time.time()-t0:.0f}s", flush=True)
torch.save(m.state_dict(), "dialog_chat_final.pt")

print("[eval] checkpoints @4096 (+ @16384 probes):", flush=True)
for c in CFG["ckpts"]:
    m.load_state_dict(torch.load(f"dialog_chat_{c}.pt"))
    eval_all(m, f"v8_{c // 1000}k", RESULTS)
m.load_state_dict(torch.load("dialog_chat_final.pt"))
print("[eval] generated dialogue (greedy, 10 turns):", flush=True)
print(dialogue_gen(m), flush=True)

print("\n" + "=" * 100)
print("MACHINE v8 CHATBOT — SUCCESS: (D1) state4096 <= 0.01 @12k;")
print("(D2) overwrite <= 0.05; (D3) state16k <= state4k + 0.05;")
print("(D4) math-plus <= 0.02 / math-minus <= 0.05; (D5) chat <= 0.02;")
print("(D6) routing 1.0; (D7) dialogue exact.")
print("=" * 100)
for k in ["v8_3k", "v8_6k", "v8_9k", "v8_12k"]:
    v = RESULTS[k]
    print(f"{k:<7} state {v['state4096']}  s16k {v['state16384']}  "
          f"m+ {v['mathplus4096']}  m- {v['mathminus4096']}  "
          f"m16k {v['math16384']}  chat {v['chat4096']}  "
          f"ow {v['overwrite4096']}  hg {v['head_gates']}  "
          f"m_abs {v['st_m_abs']}", flush=True)
final = {"tag": "ARC2-C22-CHATBOT-MACHINE-V8", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
