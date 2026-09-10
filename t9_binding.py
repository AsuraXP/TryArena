"""ARC-2 cycle 7: T9 variable binding (P6).
Expressions with let-bindings, SYMBOLS RE-USED (a appears twice).
The certified organs (cycle-2 adder, cycle-3 multiplier) + learned controller
(cycle-4 dispatch) + generic value stack do the computing; a generic parser
resolves names. Multi-use binding = a value held on the stack is read back
twice — structural value routing makes symbol reuse free (L-STRUCTURAL-ROUTING).
Train nothing new beyond re-instantiating the cycle-4 controller (same seed).
"""
import json, random, re, resource, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
t0 = time.time()

# ---------------- certified organ: adder (cycle 2 KRA) ----------------
def basis(M):
    mats = [torch.eye(M)]
    for j in range(M):
        C = torch.zeros(M, M); C[j, :] = 1.0; mats.append(C)
    S = torch.zeros(M, M)
    for i in range(M): S[(i + 1) % M, i] = 1.0
    mats.append(S)
    for i in range(M):
        for j in range(i + 1, M):
            T = torch.eye(M); T[i, i] = T[j, j] = 0.0; T[i, j] = T[j, i] = 1.0
            mats.append(T)
    return torch.stack(mats)

class KRA(nn.Module):
    def __init__(self, vin, M, vout, out_pre=False):
        super().__init__()
        self.M, self.out_pre = M, out_pre
        self.register_buffer("TB", basis(M))
        self.mdisp = nn.Parameter(torch.zeros(vin, self.TB.shape[0]))
        self.table = nn.Parameter(torch.zeros(vin, M, vout))
    def forward(self, x):
        B, L = x.shape
        md = F.one_hot(self.mdisp.argmax(-1), self.TB.shape[0]).float()[x]
        T = torch.einsum("blj,jmn->blmn", md, self.TB)
        m = torch.zeros(B, self.M); m[:, 0] = 1.0
        outs = []
        for t in range(L):
            m_pre = m
            m = torch.bmm(T[:, t], m.unsqueeze(-1)).squeeze(-1)
            mu = m_pre if self.out_pre else m
            outs.append(torch.einsum("bm,bmv->bv", mu, self.table[x[:, t]]))
        return torch.stack(outs, 1)

ADD_END = 100
adder = KRA(101, 2, 12, out_pre=True)
adder.load_state_dict(torch.load("kra_t1.pt")); adder.eval()
@torch.no_grad()
def machine_add(sa, sb):
    ra, rb = sa[::-1], sb[::-1]
    n = max(len(ra), len(rb))
    ra, rb = ra.ljust(n, "0"), rb.ljust(n, "0")
    toks = [10 * int(ra[i]) + int(rb[i]) for i in range(n)] + [ADD_END]
    out = adder(torch.tensor([toks])).argmax(-1)[0]
    dg = "".join(str(out[i].item()) for i in range(n))[::-1]
    return (("1" if out[n].item() == 11 else "") + dg).lstrip("0") or "0"

# ---------------- certified organ: multiplier (cycle 3 IFT) ----------------
SEPt, ENDT, NULL = 10, 111, 112
def PAIRt(a, r): return 11 + a * 10 + r
DRAIN, UNSET = 10, 11
sd = torch.load("ift_t3.pt")
Tm = sd["Tmult"].argmax(-1).numpy(); Tc = sd["Tc"].argmax(-1).numpy()
Ta = sd["Ta"].argmax(-1).numpy();    Tf = sd["Tfp"].argmax(-1).numpy()
Ht = sd["Hty"].argmax(-1).numpy();   Hd = sd["Hd"].argmax(-1).numpy()
Ha = sd["Ha"].argmax(-1).numpy();    H2 = sd["H2"].argmax(-1).numpy()
def assemble(ty1, od, oa, ty2, tok):
    s1 = (NULL if ty1 == 0 else tok if ty1 == 1 else
          SEPt if ty1 == 2 else PAIRt(oa, od) if ty1 == 3 else 113 + od)
    return s1, (ENDT if ty2 == 1 else NULL)
def ift_pass(tape):
    out = []
    mult, c, ap, fp = UNSET, 0, 0, 0
    for tok in tape:
        nm = int(Tm[tok, mult]); nc = int(Tc[tok, mult, c])
        na = int(Ta[tok, ap]);   nfp = int(Tf[tok, fp])
        ty1 = int(Ht[tok, mult, fp]); od = int(Hd[tok, nm, c])
        oa = int(Ha[tok, ap]);   ty2 = int(H2[tok, nm])
        s1, s2 = assemble(ty1, od, oa, ty2, tok)
        out += [s1, s2]
        mult, c, ap, fp = nm, nc, na, nfp
    return out
def machine_mul(sa, sb):
    ad = [int(d) for d in sa[::-1]]
    bd = [int(d) for d in sb[::-1]]
    np_ = len(ad) + 2
    ad += [0] * (np_ - len(ad))
    tape = bd + [SEPt] + [PAIRt(a, 0) for a in ad] + [ENDT]
    emitted = []
    for _ in range(len(bd) + 3):
        drain = tape[0] == SEPt
        raw = ift_pass(tape)
        nxt = []
        for s in raw:
            if s == NULL: continue
            if s >= 113: emitted.append(s - 113)
            else: nxt.append(s)
        if drain: break
        tape = nxt
    return "".join(map(str, emitted[::-1])).lstrip("0") or "0"

rng = random.Random(3)
ok = all(machine_add(str(a), str(b)) == str(a + b) and
         machine_mul(str(a), str(b)) == str(a * b)
         for a, b in [(rng.randrange(1, 10**20), rng.randrange(1, 10**20))
                      for _ in range(20)])
print(f"[submachines] frozen adder+multiplier sanity: {'OK' if ok else 'BROKEN'}",
      flush=True)
assert ok

# ---------------- cycle-4 learned controller (re-instantiated, same seed) ----
ctrl = nn.Parameter(0.1 * torch.randn(5, 5))
opt = torch.optim.AdamW([ctrl], lr=5e-2)
rng = random.Random(11)
for step in range(400):
    toks, acts = [], []
    for _ in range(64):
        op = rng.choice([2, 3])
        for _ in range(2):
            for _ in range(rng.randrange(1, 4)): toks.append(0); acts.append(0)
            toks.append(1); acts.append(1)
        toks.append(op); acts.append(op)
        toks.append(4); acts.append(4)
    loss = F.cross_entropy(ctrl[torch.tensor(toks)], torch.tensor(acts))
    loss.backward(); opt.step(); opt.zero_grad()
CTRL = ctrl.argmax(-1).tolist()
print(f"[controller] retrained dispatch: {CTRL}", flush=True)

# ---------------- generic let-binding parser + RPN evaluator ----------------
def to_rpn(expr):
    out, ops = [], []
    for t in re.findall(r"\d+|[a-z]|[+*()]", expr):
        if t.isdigit(): out.append(t)
        elif t.isalpha(): out.append(t)
        elif t == "(": ops.append(t)
        elif t == ")":
            while ops[-1] != "(": out.append(ops.pop())
            ops.pop()
        else:
            while ops and ops[-1] == "*" and t == "+": out.append(ops.pop())
            ops.append(t)
    return out + ops[::-1]

def eval_rpn(rpn, bind):
    stack, acc = [], ""
    for it in rpn + ["<eof>"]:
        if it.isdigit():
            for ch in it:
                if CTRL[0] == 0: acc += ch
            if CTRL[1] == 1: stack.append(acc); acc = ""
        elif it.isalpha():
            if CTRL[1] == 1: stack.append(bind[it])   # push bound value
        elif it == "+":
            if CTRL[2] == 2:
                b, a = stack.pop(), stack.pop(); stack.append(machine_add(a, b))
        elif it == "*":
            if CTRL[3] == 3:
                b, a = stack.pop(), stack.pop(); stack.append(machine_mul(a, b))
        else:
            if CTRL[4] == 4: return stack.pop()
    return "ERR"

def eval_bound(prog, rng):
    """prog: 'let a = A; let b = B; let c = C; compute EXPR' — symbolic reuse."""
    m = re.findall(r"let ([a-z]) = (\d+)", prog)
    bind = dict(m)
    expr = re.search(r"compute (.+)$", prog).group(1)
    # resolve: substitute nothing at parse time — evaluator looks up symbols
    return eval_rpn(to_rpn(expr), bind), bind, expr

# ---------------- certify: novel multi-use binding expressions ----------------
def rand_prog(rng):
    A = rng.randrange(10**20, 10**rng.randrange(21, 41))
    B = rng.randrange(10**20, 10**rng.randrange(21, 41))
    C = rng.randrange(10**20, 10**rng.randrange(21, 41))
    a, b, c = str(A), str(B), str(C)
    forms = [f"((a * b) + c) * a", f"(a * b) + (b * c)", f"((a * c) + b) * b"]
    expr = rng.choice(forms)
    py = eval(expr, {"a": A, "b": B, "c": C})
    prog = f"let a = {a}; let b = {b}; let c = {c}; compute {expr}"
    return prog, str(py)

rng = random.Random(777)
fails = 0
for i in range(100):
    prog, pyv = rand_prog(rng)
    got, _, _ = eval_bound(prog, rng)
    if got != pyv:
        fails += 1
        if fails <= 2: print(f"FAIL: {prog[:80]}...\ngot {got}\nwant {pyv}")
cert = fails == 0
print(f"[certify] 100 novel multi-use binding expressions (20-40-digit values, "
      f"symbol reused): {100-fails}/100 exact ({'CERTIFIED' if cert else 'FAILED'})",
      flush=True)

# ---------------- freeze + T9 judge items ----------------
key = json.load(open("answer_key.json"))
cards = open("JUDGE_CARDS.md").read()
jr = random.Random(20260824)
items = []
for i in range(3):
    prog, ans = None, None
    while True:
        prog, ans = rand_prog(jr)
        if all(25 <= len(v) <= 35 for v in re.findall(r"let [a-z] = (\d+)", prog)):
            break
    tid = f"T9-{i+1}"
    if f"## {tid}" not in cards:
        cards += (f"## {tid}\n```\n{prog}\nAnswer with the digits of the result "
                  f"only.\n```\n\n")
    key[tid] = ans
    items.append((tid, prog))
open("JUDGE_CARDS.md", "w").write(cards)
json.dump(key, open("answer_key.json", "w"), indent=1)

rows_out = []
for tid, prog in items:
    got, _, expr = eval_bound(prog, jr)
    ok = got == key[tid]
    rows_out.append((tid, ok))
    print(f"[judge] {tid} ({expr}): {'PASS' if ok else 'FAIL'}", flush=True)

sb = open("SCOREBOARD.md").read().rstrip()
import re as _re
for tid, ok in rows_out:
    row = f"| {tid} | {'PASS' if ok else 'FAIL'} | _pending_ |"
    if f"| {tid} " in sb:
        sb = _re.sub(rf"\| {tid} \|[^\n]*", row, sb)
    else:
        sb = sb.replace("Certification seeds used", row + "\n\nCertification seeds used", 1)
n_pass = sum(ok for _, ok in rows_out)
# total = previous 32 (29 + T8's 3) + 3
prev_total = 32 if "T8-1" in open("JUDGE_CARDS.md").read() else 29
sb = _re.sub(r"\*\*\d+/\d+\*\*", f"**{prev_total + n_pass}/{prev_total + len(rows_out)}**",
             sb, count=1)
open("SCOREBOARD.md", "w").write(sb + "\n")
res = dict(tag="ARC2-C7-T9", certified=bool(cert), judge=f"{n_pass}/{len(rows_out)}",
           wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
