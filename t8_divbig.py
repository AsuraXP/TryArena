"""ARC-2 cycle 7: T8 big/big division N / D (D up to 50 digits) via IFT.
Nested-iteration frontier (P4): outer loop over N's digits (MSB-first), inner
loop = up to 9 compare-subtract attempts per digit.

Tape: [N-digits MSB-first][SEP][pairs (r_i,d_i) LSD-first][DEC?][QTOK][END]
  pair(r,d) = 11 + 10r + d ; QTOK[q] = 111+q ; DEC(f) = 122+f ; EMIT(q) = 130+q
Each pass = Mealy transducer with TWO output slots per input token (T3 pattern),
NONE = VTOK. Three passes with LEARNED factored tables (next-state + outputs;
L-DIRECT-GRADIENT — every discrete decision gets direct per-cell supervision):
  SHIFT : consume next N-digit a, r <- 10r+a, cmp(new r vs D), (DEC, QTOK).
  SUB   : (harness runs when DEC(1)) borrow-subtract D, Q+1, re-cmp, (DEC, END).
  EMIT  : (harness runs when DEC(0)) drop DEC, (EMIT(Q), END).
Harness (generic token rule, T3-class):
  DEC(1) -> SUB | DEC(0) -> EMIT | else N-digit present -> SHIFT | else done.
Oracle-first protocol: encoding verified 500/500 BEFORE learning."""
import json, random, resource, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.manual_seed(0)
t0 = time.time()

# ---------------- tokens ----------------
SEP = 10
def PAIR(r, d): return 11 + 10 * r + d
def UNPAIR(t): return (t - 11) // 10, (t - 11) % 10
def QTOK(q): return 111 + q
def QVAL(t): return t - 111
def DEC(f): return 122 + f
def DFLAG(t): return t - 122
END = 121
def EMIT(q): return 130 + q
VTOK = 140          # first unused token id
VNOC = VTOK + 1     # output classes: tokens 0..139 + NONE
NEUT = VTOK         # neutral marker (never emitted); distinct from END

EQ, GT, LT = 0, 1, 2
def cmp_of(r, d):
    return GT if r > d else (LT if r < d else EQ)
def is_pair(t): return 11 <= t <= 110
def is_q(t): return 111 <= t <= 120
def is_dec(t): return 122 <= t <= 123

NSH = 3 * 11 * 10 * 2 * 3      # SHIFT state: phase*11*10*2*3, ahold, rhold, first, cmp
NSU = 3 * 2 * 3                # SUB state: phase, borrow, cmp
NE  = 3 * 10                   # EMIT state: phase, qreg

# ---------------- reference passes (the oracle) ----------------
def ref_shift(tape):
    out = []
    phase, ahold, rhold, first, cmp_ = 0, 10, 0, 0, EQ
    for tok in tape:
        if phase == 0:
            if tok == SEP:
                out += [SEP, NEUT]; phase, first = 1, 1
            elif ahold == 10:
                ahold = tok; out += [NEUT, NEUT]
            else:
                out += [tok, NEUT]
        elif phase == 1:
            if is_pair(tok):
                r_i, d_i = UNPAIR(tok)
                nr = ahold if first else rhold
                if nr != d_i: cmp_ = cmp_of(nr, d_i)
                first, rhold = 0, r_i
                out += [PAIR(nr, d_i), NEUT]
            elif is_q(tok):
                out += [DEC(1 if cmp_ in (GT, EQ) else 0), QTOK(0)]; phase = 2   # Q reset per outer digit
        else:
            out += [tok, NEUT]
    return [x for x in out if x != NEUT]

def ref_sub(tape):
    out = []
    phase, borrow, cmp_ = 0, 0, EQ
    for tok in tape:
        if is_dec(tok):
            out += [NEUT, NEUT]                    # consume DEC at any phase
        elif phase == 0:
            if tok == SEP: phase = 1
            out += [tok, NEUT]
        elif phase == 1:
            if is_pair(tok):
                r_i, d_i = UNPAIR(tok)
                t = r_i - d_i - borrow
                borrow, nr = (1 if t < 0 else 0), t % 10
                if nr != d_i: cmp_ = cmp_of(nr, d_i)
                out += [PAIR(nr, d_i), NEUT]
            elif is_q(tok):
                out += [QTOK(min(QVAL(tok) + 1, 9)), NEUT]; phase = 2
            else:
                out += [tok, NEUT]
        else:
            if tok == END: out += [DEC(1 if cmp_ in (GT, EQ) else 0), END]
            else: out += [tok, NEUT]
    return [x for x in out if x != NEUT]

def ref_emit(tape):
    out = []
    phase, qreg = 0, 0
    for tok in tape:
        if is_dec(tok):
            out += [NEUT, NEUT]
        elif is_q(tok):
            qreg = QVAL(tok)
            if phase == 1: phase = 2
            out += [tok, NEUT]
        else:
            if tok == SEP and phase == 0: phase = 1
            if tok == END: out += [EMIT(qreg), END]
            else: out += [tok, NEUT]
    return [x for x in out if x != NEUT]

REF = {"shift": ref_shift, "sub": ref_sub, "emit": ref_emit}

def make_tape(N, D):
    n, d = str(N), str(D)
    W = max(len(d), 1) + 1
    d = d.zfill(W)
    return ([int(c) for c in n] + [SEP]
            + [PAIR(0, int(d[-1 - i])) for i in range(W)] + [QTOK(0), END])

def run_pipeline(N, D, passes, max_passes=20000):
    tape = make_tape(N, D)
    emitted = []
    for _ in range(max_passes):
        dts = [t for t in tape if is_dec(t)]
        if dts:
            tape = (passes["sub"] if DFLAG(dts[0]) == 1 else passes["emit"])(tape)
        elif tape[0] != SEP:
            tape = passes["shift"](tape)
        else:
            break
        e = [t for t in tape if t >= 130]
        if e:
            emitted.extend(t - 130 for t in e)
            tape = [t for t in tape if t < 130]
    q = "".join(map(str, emitted)).lstrip("0") or "0"     # q-digits emitted MSB-first
    r = "".join(str(UNPAIR(t)[0]) for t in tape if is_pair(t))[::-1].lstrip("0") or "0"
    return q, r

# ---------------- oracle: verify the ENCODING before any learning ----------------
rng = random.Random(7)
bad = 0
for _ in range(500):
    D = rng.randrange(2, 10 ** rng.randrange(1, 13))
    N = rng.randrange(1, 10 ** rng.randrange(1, 40))
    if run_pipeline(N, D, REF) != (str(N // D), str(N % D)): bad += 1
print(f"[oracle] encoding check: {500-bad}/500 exact "
      f"({'OK' if bad == 0 else 'BROKEN'})", flush=True)
assert bad == 0

# ---------------- row collection: (tok, si, next-state..., output types) -------
# State (phase redundant: token ranges disambiguate N/SEP/pair/Q/END):
#   SHIFT: (ahold[11], rhold[10], first[2], cmp[3])  = 660
#   SUB  : (borrow[2], cmp[3])                       =   6
#   EMIT : (qreg[10])                                =  10
# Factored outputs (T3 pattern): type class + learned r-digit value table.
#   o1ty: 0=NONE 1=TOK 2=SEP 3=PAIR' 4=DEC 5=END 6=QTOK0 7=QTOK' 8=EMIT'
#   o2ty: 0=NONE 1=TOK 2=QTOK0 3=END
NSH = 11 * 10 * 2 * 3
NSU = 2 * 3
NE  = 10
O1N, O2N = 9, 4

def sh_idx(ah, rh, fs, cm): return ((ah * 10 + rh) * 2 + fs) * 3 + cm
def su_idx(bw, cm): return bw * 3 + cm

def rows_shift(tape, rows):
    phase, ahold, rhold, first, cmp_ = 0, 10, 0, 0, EQ
    for tok in tape:
        si = sh_idx(ahold, rhold, first, cmp_)
        nah, nrh, nfs, ncm, o1ty, o2ty, nrr = ahold, rhold, first, cmp_, 0, 0, 0
        if phase == 0:
            if tok == SEP:
                o1ty, nfs = 2, 1
            elif ahold == 10:
                nah = tok
            else:
                o1ty = 1
        elif phase == 1:
            if is_pair(tok):
                r_i, d_i = UNPAIR(tok)
                nr = ahold if first else rhold
                ncm = cmp_of(nr, d_i) if nr != d_i else cmp_
                nfs, nrh, nrr = 0, r_i, nr
                o1ty = 3
            elif is_q(tok):
                o1ty, o2ty = 4, 2
        else:
            o1ty = 1                                   # phase 2: pass through
        rows.append((tok, si, nah, nrh, nfs, ncm, o1ty, o2ty, nrr))
        phase = 1 if (phase == 0 and tok == SEP) else \
                (2 if (phase == 1 and is_q(tok)) else phase)
        ahold, rhold, first, cmp_ = nah, nrh, nfs, ncm

def rows_sub(tape, rows):
    phase, borrow, cmp_ = 0, 0, EQ
    for tok in tape:
        si = su_idx(borrow, cmp_)
        nbw, ncm, o1ty, o2ty, nrr = borrow, cmp_, 0, 0, 0
        if is_dec(tok):
            pass                                    # consume DEC at any phase
        elif phase == 0:
            o1ty = 1
            if tok == SEP: phase = 1
        elif phase == 1:
            if is_pair(tok):
                r_i, d_i = UNPAIR(tok)
                t = r_i - d_i - borrow
                nbw, nr = (1 if t < 0 else 0), t % 10
                ncm = cmp_of(nr, d_i) if nr != d_i else cmp_
                o1ty, nrr = 3, nr
            elif is_q(tok):
                o1ty = 7
                phase = 2
            else:
                o1ty = 1
        else:
            if tok == END:
                o1ty, o2ty = 4, 3
        rows.append((tok, si, nbw, ncm, o1ty, o2ty, nrr))
        phase, borrow, cmp_ = phase, nbw, ncm

def rows_emit(tape, rows):
    phase, qreg = 0, 0
    for tok in tape:
        si = qreg
        nqr, o1ty, o2ty = qreg, 0, 0
        if is_dec(tok):
            pass
        elif is_q(tok):
            nqr = QVAL(tok)
            o1ty = 1
            if phase == 1: phase = 2
        elif tok == END:
            o1ty, o2ty = 8, 3
        else:
            o1ty = 1
            if tok == SEP and phase == 0: phase = 1
        rows.append((tok, si, nqr, o1ty, o2ty))
        phase, qreg = phase, nqr

def gen_item(nmax, dmax, rng):
    # 30% of items: N has a digit-prefix that is an exact multiple of D
    # (guarantees the rare R == 2D exact-equality subtract state, cmp=EQ)
    if rng.random() < 0.3:
        D = rng.randrange(2, 10 ** dmax)
        nd = len(str(D))
        Q = rng.randrange(2, 10 ** (nmax - nd))
        P = Q * D
        s = rng.randrange(0, nmax - len(str(P)) + 1)
        S = rng.randrange(0, 10 ** s) if s else 0
        N = P * (10 ** s) + S
    else:
        D = rng.randrange(2, 10 ** dmax)
        N = rng.randrange(1, 10 ** nmax)
    return N, D

def collect_rows(n_items, nmax, dmin, dmax, rng):
    RS, RU, RE = [], [], []
    for _ in range(n_items):
        N, D = gen_item(nmax, dmax, rng)
        tape = make_tape(N, D)
        for _ in range(20000):
            dts = [t for t in tape if is_dec(t)]
            if dts:
                if DFLAG(dts[0]) == 1:
                    rows_sub(tape, RU); tape = REF["sub"](tape)
                else:
                    rows_emit(tape, RE); tape = REF["emit"](tape)
            elif tape[0] != SEP:
                rows_shift(tape, RS); tape = REF["shift"](tape)
            else:
                break
            e = [t for t in tape if t >= 130]
            if e: tape = [t for t in tape if t < 130]
    return torch.tensor(RS), torch.tensor(RU), torch.tensor(RE)

# ---------------- learned factored tables ----------------
class BigDiv(nn.Module):
    def __init__(self):
        super().__init__()
        z = lambda *s: nn.Parameter(0.1 * torch.randn(*s))
        self.S_ah = z(VTOK, NSH, 11)
        self.S_rh = z(VTOK, NSH, 10)
        self.S_fs = z(VTOK, NSH, 2)
        self.S_cm = z(VTOK, NSH, 3)
        self.S_o1 = z(VTOK, NSH, O1N)
        self.S_o2 = z(VTOK, NSH, O2N)
        self.S_nr = z(VTOK, NSH, 10)
        self.U_bw = z(VTOK, NSU, 2)
        self.U_cm = z(VTOK, NSU, 3)
        self.U_o1 = z(VTOK, NSU, O1N)
        self.U_o2 = z(VTOK, NSU, O2N)
        self.U_nr = z(VTOK, NSU, 10)
        self.E_qr = z(VTOK, NE, 10)
        self.E_o1 = z(VTOK, NE, O1N)
        self.E_o2 = z(VTOK, NE, O2N)

model = BigDiv()
opt = torch.optim.AdamW(model.parameters(), lr=2e-2)
rng = random.Random(1)
for step in range(1, 2501):
    rs, ru, re_ = collect_rows(32, 12, 0, 9, rng)
    loss = (
        F.cross_entropy(model.S_ah[rs[:, 0], rs[:, 1]], rs[:, 2])
        + F.cross_entropy(model.S_rh[rs[:, 0], rs[:, 1]], rs[:, 3])
        + F.cross_entropy(model.S_fs[rs[:, 0], rs[:, 1]], rs[:, 4])
        + F.cross_entropy(model.S_cm[rs[:, 0], rs[:, 1]], rs[:, 5])
        + F.cross_entropy(model.S_o1[rs[:, 0], rs[:, 1]], rs[:, 6])
        + F.cross_entropy(model.S_o2[rs[:, 0], rs[:, 1]], rs[:, 7])
        + F.cross_entropy(model.S_nr[rs[:, 0], rs[:, 1]], rs[:, 8])
        + F.cross_entropy(model.U_bw[ru[:, 0], ru[:, 1]], ru[:, 2])
        + F.cross_entropy(model.U_cm[ru[:, 0], ru[:, 1]], ru[:, 3])
        + F.cross_entropy(model.U_o1[ru[:, 0], ru[:, 1]], ru[:, 4])
        + F.cross_entropy(model.U_o2[ru[:, 0], ru[:, 1]], ru[:, 5])
        + F.cross_entropy(model.U_nr[ru[:, 0], ru[:, 1]], ru[:, 6])
        + F.cross_entropy(model.E_qr[re_[:, 0], re_[:, 1]], re_[:, 2])
        + F.cross_entropy(model.E_o1[re_[:, 0], re_[:, 1]], re_[:, 3])
        + F.cross_entropy(model.E_o2[re_[:, 0], re_[:, 1]], re_[:, 4]))
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 500 == 0:
        print(f"[train] {step}/2500 CE {loss.item():.5f}", flush=True)

# ---------------- snap to integer machine (LEARNED next-state, T7 pattern) -------
with torch.no_grad():
    Sah, Srh, Sfs, Scm = model.S_ah.argmax(-1).numpy(), model.S_rh.argmax(-1).numpy(), \
        model.S_fs.argmax(-1).numpy(), model.S_cm.argmax(-1).numpy()
    So1, So2, Snr = model.S_o1.argmax(-1).numpy(), model.S_o2.argmax(-1).numpy(), \
        model.S_nr.argmax(-1).numpy()
    Ubw, Ucm = model.U_bw.argmax(-1).numpy(), model.U_cm.argmax(-1).numpy()
    Uo1, Uo2, Unr = model.U_o1.argmax(-1).numpy(), model.U_o2.argmax(-1).numpy(), \
        model.U_nr.argmax(-1).numpy()
    Eqr, Eo1, Eo2 = model.E_qr.argmax(-1).numpy(), model.E_o1.argmax(-1).numpy(), \
        model.E_o2.argmax(-1).numpy()

def lshift(tape):
    out = []
    ahold, rhold, first, cmp_ = 10, 0, 0, EQ
    for tok in tape:
        si = sh_idx(ahold, rhold, first, cmp_)
        t1, t2 = int(So1[tok, si]), int(So2[tok, si])
        if t1 == 3: out.append(PAIR(int(Snr[tok, si]), UNPAIR(tok)[1]))
        elif t1 == 4: out.append(DEC(1 if cmp_ in (GT, EQ) else 0))
        elif t1 in (1, 2, 5): out.append({1: tok, 2: SEP, 5: END}[t1])
        if t2 == 2: out.append(QTOK(0))
        elif t2 == 1: out.append(tok)
        ahold, rhold, first, cmp_ = int(Sah[tok, si]), int(Srh[tok, si]), \
            int(Sfs[tok, si]), int(Scm[tok, si])
    return out

def lsub(tape):
    out = []
    borrow, cmp_ = 0, EQ
    for tok in tape:
        si = su_idx(borrow, cmp_)
        t1, t2 = int(Uo1[tok, si]), int(Uo2[tok, si])
        if t1 == 3: out.append(PAIR(int(Unr[tok, si]), UNPAIR(tok)[1]))
        elif t1 == 4: out.append(DEC(1 if cmp_ in (GT, EQ) else 0))
        elif t1 == 7: out.append(QTOK(min(QVAL(tok) + 1, 9)))
        elif t1 == 1: out.append(tok)
        if t2 == 3: out.append(END)
        elif t2 == 1: out.append(tok)
        borrow, cmp_ = int(Ubw[tok, si]), int(Ucm[tok, si])
    return out

def lemit(tape):
    out = []
    qreg = 0
    for tok in tape:
        si = qreg
        t1, t2 = int(Eo1[tok, si]), int(Eo2[tok, si])
        if t1 == 8: out.append(EMIT(qreg))
        elif t1 == 1: out.append(tok)
        if t2 == 3: out.append(END)
        qreg = int(Eqr[tok, si])
    return out

LP = {"shift": lshift, "sub": lsub, "emit": lemit}
torch.save({"model": model.state_dict()}, "t8_divbig.pt")
print("[ckpt] t8_divbig.pt saved", flush=True)

# sanity: learned vs oracle on fresh small cases
rng = random.Random(555)
bad = 0
for ci in range(50):
    D = rng.randrange(2, 10 ** rng.randrange(1, 11))
    N = rng.randrange(1, 10 ** rng.randrange(1, 25))
    a, b = run_pipeline(N, D, LP), run_pipeline(N, D, REF)
    if a != b:
        bad += 1
        print(f"[sanity] MISMATCH case {ci}: {N} // {D}: LP {a} vs REF {b}", flush=True)
    if ci % 10 == 0: print(f"[sanity] {ci}/50 done", flush=True)
print(f"[sanity] learned vs oracle, 50 fresh small cases: {50-bad}/50 agree", flush=True)

# coverage audit: does the table agree with REF's true (tok,si)->output mapping
# over a large independent long-divisor set (stresses rare states)?
rng = random.Random(777)
AUD_RS, AUD_RU, AUD_RE = [], [], []
for _ in range(400):
    D = rng.randrange(100, 10 ** rng.randrange(3, 13))
    N = rng.randrange(10 ** rng.randrange(10, 31), 10 ** 31)
    t = make_tape(N, D)
    for _ in range(20000):
        dts = [x for x in t if is_dec(x)]
        if dts:
            f = DFLAG(dts[0])
            (rows_sub if f == 1 else rows_emit)(t, AUD_RU if f == 1 else AUD_RE)
            t = (ref_sub if f == 1 else ref_emit)(t)
        elif t[0] != SEP:
            rows_shift(t, AUD_RS); t = ref_shift(t)
        else:
            break
        e = [x for x in t if x >= 130]
        if e: t = [x for x in t if x < 130]
def audit(rows, name, *tabs):
    m = {}
    for r in rows: m[(r[0], r[1])] = tuple(r[2:])
    mism = 0
    for k, tgt in m.items():
        got = tuple(int(v) for v in (tab[k] for tab in tabs))
        if got != tgt: mism += 1
    print(f"[audit] {name}: {len(m)} distinct (tok,si) combos, table mismatches: {mism}", flush=True)
    return mism
am = audit(AUD_RS, "shift", Sah, Srh, Sfs, Scm, So1, So2, Snr)
am += audit(AUD_RU, "sub", Ubw, Ucm, Uo1, Uo2, Unr)
am += audit(AUD_RE, "emit", Eqr, Eo1, Eo2)
print(f"[audit] total table mismatches vs REF mapping: {am}", flush=True)

# ---------------- certify: big operands ----------------
rng = random.Random(999)
fails, total = 0, 0
for i in range(200):
    nd1, nd2 = rng.randrange(40, 151), rng.randrange(5, 51)
    N = rng.randrange(10 ** (nd1 - 1), 10 ** nd1)
    D = rng.randrange(10 ** (nd2 - 1), 10 ** nd2)
    total += 1
    got = run_pipeline(N, D, LP)
    want = (str(N // D), str(N % D))
    if got != want:
        fails += 1
        print(f"[certify] FAIL i={i}: {N} // {D}: got {got} want {want}", flush=True)
    if i % 20 == 0: print(f"[certify] {i}/200 done", flush=True)
# directed certify: exact-multiple prefixes (force the R == 2D EQ state at big scale)
for j in range(40):
    D = rng.randrange(10 ** 4, 10 ** 50)
    nd = len(str(D))
    Q = rng.randrange(2, 10 ** min(nd, 4))
    P = Q * D
    s = rng.randrange(0, 151 - len(str(P)))
    S = rng.randrange(0, 10 ** s) if s else 0
    N = P * (10 ** s) + S
    total += 1
    got = run_pipeline(N, D, LP)
    want = (str(N // D), str(N % D))
    if got != want:
        fails += 1
        print(f"[certify] FAIL mult j={j}: {N} // {D}: got {got} want {want}", flush=True)
# quotient-zero family: N < D
for k in range(20):
    D = rng.randrange(10 ** rng.randrange(5, 51))
    N2 = rng.randrange(1, D)
    total += 1
    got = run_pipeline(N2, D, LP)
    want = (str(N2 // D), str(N2 % D))
    if got != want:
        fails += 1
        print(f"[certify] FAIL zero k={k}: {N2} // {D}: got {got} want {want}", flush=True)
cert = fails == 0
print(f"[certify] N 40-150 digits / D 5-50 digits (incl N<D + 40 exact-multiple): "
      f"{total-fails}/{total} exact ({'CERTIFIED' if cert else 'FAILED'})", flush=True)

# ---------------- freeze + T8 judge items ----------------
key = json.load(open("answer_key.json"))
cards = open("JUDGE_CARDS.md").read()
jr = random.Random(20260823)
items = []
for i in range(3):
    nd1, nd2 = jr.choice([100, 120, 150]), jr.choice([15, 25, 40])
    N = jr.randrange(10 ** (nd1 - 1), 10 ** nd1)
    D = jr.randrange(10 ** (nd2 - 1), 10 ** nd2)
    tid = f"T8-{i+1}"
    if f"## {tid}" not in cards:
        cards += (f"## {tid}\n```\nCompute exactly: {N} divided by {D}. "
                  f"Answer as: quotient digits, then the word 'remainder', "
                  f"then the remainder.\n```\n\n")
    key[tid] = f"{N // D} remainder {N % D}"
    items.append((tid, N, D))
open("JUDGE_CARDS.md", "w").write(cards)
json.dump(key, open("answer_key.json", "w"), indent=1)

rows_out = []
for tid, N, D in items:
    qs, rem = run_pipeline(N, D, LP)
    ok = f"{qs} remainder {rem}" == key[tid]
    rows_out.append((tid, ok))
    print(f"[judge] {tid} ({len(str(N))}/{len(str(D))} digits): "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

sb = open("SCOREBOARD.md").read().rstrip()
import re as _re
for tid, ok in rows_out:
    row = f"| {tid} | {'PASS' if ok else 'FAIL'} | _pending_ |"
    if f"| {tid} " in sb:
        sb = _re.sub(rf"\| {tid} \|[^\n]*", row, sb)
    else:
        sb = sb.replace("Certification seeds used", row + "\n\nCertification seeds used", 1)
n_pass = sum(ok for _, ok in rows_out)
sb = _re.sub(r"\*\*\d+/\d+\*\*", f"**{29 + n_pass}/{29 + len(rows_out)}**", sb, count=1)
open("SCOREBOARD.md", "w").write(sb + "\n")
params = sum(p.numel() for p in model.parameters())
certified = bool(cert) and bad == 0 and am == 0
res = dict(tag="ARC2-C7-T8", certified=certified, judge=f"{n_pass}/{len(rows_out)}",
           sanity=f"{50-bad}/50", audit_mism=am, certify=f"{total-fails}/{total}",
           params=params, wall_s=round(time.time() - t0, 1),
           peak_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1))
print("RESULT " + json.dumps(res), flush=True)
open("log.jsonl", "a").write(json.dumps(res) + "\n")
torch.save(model.state_dict(), "divbig_t8.pt")
