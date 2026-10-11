"""
ARCH-VET P51 — MICRO BINDING HARNESS (minutes per hypothesis, two lanes).
Purpose: every mechanism/learning-rule hypothesis is tested here FIRST; only
survivors touch the text harness (arch_vet_p49). Same core (CKB from p49,
lowmem regime), same fair controls (GRU, TFMicro), same batches per arm.

Stream (unmemorizable by construction, L-COPY-NEEDS-UNMEMORIZABLE-TRAINING):
  vocabulary 70 symbols: 0 pad, 1 '.', 2 'Q', 3..66 content symbols, 67 '=' .
  A sequence = K facts  "<name2> = <val2> ."  then queries "Q <name2> =" -> val2
  names/values are fresh random 2-symbol strings per sequence (never memorizable).
  in     : K facts, symbols drawn from the TRAIN subset (3..50)
  ood    : names built from symbols never seen in the name role (values unchanged)
  len    : 2K facts (2x length), train symbols
Metric: CE and exact accuracy on the value symbols at query time only.
"""
import sys, json, random, time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "/home/user/TryArena")
import arch_vet_lm as A

_argv = sys.argv; sys.argv = ["p49"]
_src = open("/home/user/TryArena/arch_vet_p49.py").read().replace('if __name__ == "__main__":\n    main()', '')
P = type(sys)("p49"); exec(compile(_src, "arch_vet_p49.py", "exec"), P.__dict__); sys.argv = _argv

V = 70; PAD, DOT, QQ, EQ = 0, 1, 2, 67; HOP = 1
TR = list(range(3, 51)); NM, VL = TR[:24], TR[24:]   # train: names from NM, values from VL; ood = roles swapped (embeddings seen, roles novel)


def make_seq2(rng, K, syms, nq):
    """2-hop: K name->name links (a = b) and K name->value facts (b = v); query 'Q Q a =' -> v (never stated)."""
    ns, vs = (NM, VL) if syms == 'tr' else (VL, VL)
    A = [(rng.choice(ns), rng.choice(ns)) for _ in range(K)]; Bn = [(rng.choice(ns), rng.choice(ns)) for _ in range(K)]
    vals = [(rng.choice(vs), rng.choice(vs)) for _ in range(K)]
    facts = [(a, b) for a, b in zip(A, Bn)] + [(b, v) for b, v in zip(Bn, vals)]; rng.shuffle(facts)
    x = []
    for n, v in facts: x += [n[0], n[1], EQ, v[0], v[1], DOT]
    tgt = [0] * len(x)
    for _ in range(nq):
        i = rng.randrange(K); a, v = A[i], vals[i]
        x += [QQ, QQ, a[0], a[1], EQ, v[0], v[1], DOT]; tgt += [0, 0, 0, 0, 0, 1, 2, 0]
    return x, tgt


def make_seq(rng, K, syms, nq):
    if HOP == 2: return make_seq2(rng, K, syms, nq)
    ns, vs = (NM, VL) if syms == 'tr' else (VL, VL)   # ood: names made of symbols never seen as names (values unchanged)
    names = [(rng.choice(ns), rng.choice(ns)) for _ in range(K)]
    vals = [(rng.choice(vs), rng.choice(vs)) for _ in range(K)]
    x = []; tgt = []
    for n, v in zip(names, vals): x += [n[0], n[1], EQ, v[0], v[1], DOT]
    tgt = [0] * len(x)
    for _ in range(nq):
        i = rng.randrange(K); n, v = names[i], vals[i]
        x += [QQ, n[0], n[1], EQ, v[0], v[1], DOT]; tgt += [0, 0, 0, 0, 1, 2, 0]
    return x, tgt


def batch(rng, B, K, syms, nq):
    seqs = [make_seq(rng, K, syms, nq) for _ in range(B)]; L = max(len(s[0]) for s in seqs)
    x = torch.zeros(B, L, dtype=torch.long); m = torch.zeros(B, L)
    for i, (s, t) in enumerate(seqs): x[i, :len(s)] = torch.tensor(s); m[i, :len(t)] = torch.tensor(t, dtype=torch.float)
    return x, m


def q_metrics(lg, x, m):
    """CE and exact acc on positions whose NEXT token is a queried value symbol (mask m marks the value positions)."""
    pm = m[:, 1:] > 0; lgp = lg[:, :-1][pm]; y = x[:, 1:][pm]
    p1 = m[:, 1:] == 1; a1 = (lg[:, :-1][p1].argmax(-1) == x[:, 1:][p1]).float().mean().item()
    return F.cross_entropy(lgp, y).item(), (lgp.argmax(-1) == y).float().mean().item(), a1


def fwd(m, arm, x, tau):
    if arm == "CKB": return m(x, tau=tau, gmin=0.0, states=True)[0]
    return m(x)


def run(arm, d, seed, steps, B, K, nq, log):
    torch.manual_seed(seed); rng = random.Random(seed)
    m = A.TFMicro(V, d) if arm == "TF" else P.GRUCore(V, d) if arm == "GRU" else P.CKB(V, d, P.MSLOTS)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time()
    for s in range(steps):
        x, mk = batch(rng, B, K, 'tr', nq); tau = 0.1 ** (s / max(steps - 1, 1)) if arm == "CKB" else 0.1
        lg = fwd(m, arm, x, tau); loss = P.ce_all(lg, x) + 2.0 * F.cross_entropy(lg[:, :-1][mk[:, 1:] > 0], x[:, 1:][mk[:, 1:] > 0])
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 100 == 0 or s == steps - 1: log(f"  [{arm} d{d} s{seed}] step {s} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")
    er = random.Random(999); res = {"arm": arm, "d": d, "seed": seed, "params": P.n_params(m)}
    with torch.no_grad():
        for nm, (kk, syms) in {"in": (K, "tr"), "ood": (K, "swap"), "len": (2 * K, "tr")}.items():
            ces, accs, a1s = [], [], []
            for _ in range(6):
                x, mk = batch(er, 16, kk, syms, nq); lg = fwd(m, arm, x, 0.1); ce, ac, a1 = q_metrics(lg, x, mk); ces.append(ce); accs.append(ac); a1s.append(a1)
            res[nm + "_ce"] = round(sum(ces) / 6, 3); res[nm + "_acc"] = round(sum(accs) / 6, 3); res[nm + "_a1"] = round(sum(a1s) / 6, 3)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="GRU,TF,CKB"); ap.add_argument("--ds", default="16,32,64"); ap.add_argument("--seeds", default="1")
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--B", type=int, default=16); ap.add_argument("--K", type=int, default=6); ap.add_argument("--nq", type=int, default=4)
    ap.add_argument("--bmode", default="rand"); ap.add_argument("--randp", type=float, default=0.1); ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--lowmem", type=int, default=1); ap.add_argument("--knorm", type=int, default=0); ap.add_argument("--dscale", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="p51"); ap.add_argument("--hop", type=int, default=1); ap.add_argument("--out", default="")
    a = ap.parse_args(); torch.set_num_threads(a.threads); global HOP; HOP = a.hop
    P.BMODE, P.RANDP, P.MSLOTS, P.LOWMEM, P.KNORM, P.DSCALE = a.bmode, a.randp, a.M, a.lowmem, a.knorm, a.dscale
    P.PKB, P.PUREADDR, P.ROLE, P.SURPRISE, P.DENSE, P.RANDNAMES = 1, 3, 4, 1, True, True
    out = open(a.out, "a") if a.out else None
    def log(s):
        print(s, flush=True)
        if out: out.write(s + "\n"); out.flush()
    rows = []
    for arm in a.arms.split(","):
        for d in [int(v) for v in a.ds.split(",")]:
            for seed in [int(v) for v in a.seeds.split(",")]:
                r = run(arm, d, seed, a.steps, a.B, a.K, a.nq, log); r["tag"] = a.tag; rows.append(r)
                log(f"[{a.tag} {arm} d{d} s{seed}] params {r['params']} in {r['in_ce']}/{r['in_acc']} ood {r['ood_ce']}/{r['ood_acc']} len {r['len_ce']}/{r['len_acc']} | first-symbol acc in/ood/len {r['in_a1']}/{r['ood_a1']}/{r['len_a1']}")
    log("[P51] " + json.dumps(rows))


if __name__ == "__main__":
    main()
