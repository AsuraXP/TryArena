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
import os as _os
_src = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "arch_vet_p49.py")).read().replace('if __name__ == "__main__":\n    main()', '')
P = type(sys)("p49"); exec(compile(_src, "arch_vet_p49.py", "exec"), P.__dict__); sys.argv = _argv

V = 70; PAD, DOT, QQ, EQ = 0, 1, 2, 67; HOP = 1; MIXEVAL = False
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


def make_seq4(rng, K, syms, nq):
    """C122 hops in time: same facts as make_seq2; each query is either the TWO-STEP form 'Q a = b . Q b = v .' (two 1-hop reads in
    real time, both scored) or the ONE-STEP form 'Q Q a = v .' on a DIFFERENT link (never the same i, so the one-step answer is never a
    recency copy). Eval (MIXEVAL) is one-step only."""
    ns, vs = (NM, VL) if syms == 'tr' else (VL, VL)
    A = [(rng.choice(ns), rng.choice(ns)) for _ in range(K)]; Bn = [(rng.choice(ns), rng.choice(ns)) for _ in range(K)]
    vals = [(rng.choice(vs), rng.choice(vs)) for _ in range(K)]
    facts = [(a, b) for a, b in zip(A, Bn)] + [(b, v) for b, v in zip(Bn, vals)]; rng.shuffle(facts)
    x = []
    for n, v in facts: x += [n[0], n[1], EQ, v[0], v[1], DOT]
    tgt = [0] * len(x); idx = list(range(K)); rng.shuffle(idx); two = set(idx[:K // 2]); one = [i for i in idx if i not in two]
    for _ in range(nq):
        if rng.random() < 0.5 or not one:
            i = rng.choice(sorted(two)); a, b, v = A[i], Bn[i], vals[i]
            x += [QQ, a[0], a[1], EQ, b[0], b[1], DOT, QQ, b[0], b[1], EQ, v[0], v[1], DOT]; tgt += [0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 0, 1, 2, 0]
        else:
            i = rng.choice(one); a, v = A[i], vals[i]
            x += [QQ, QQ, a[0], a[1], EQ, v[0], v[1], DOT]; tgt += [0, 0, 0, 0, 0, 1, 2, 0]
    return x, tgt


def make_seq(rng, K, syms, nq):
    if HOP == 4: return make_seq2(rng, K, syms, nq) if MIXEVAL else make_seq4(rng, K, syms, nq)
    if HOP == 2 or (HOP == 3 and (MIXEVAL or rng.random() < 0.5)): return make_seq2(rng, K, syms, nq)
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
        loss = loss + (P.IMCONS * m.im_loss if arm == 'CKB' and P.IMCONS else 0) + (m.hx_loss if arm == 'CKB' and P.HOPX else 0); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 100 == 0 or s == steps - 1: log(f"  [{arm} d{d} s{seed}] step {s} loss {loss.item():.3f} ({time.time()-t0:.0f}s)")
    global MIXEVAL; MIXEVAL = True
    er = random.Random(999); res = {"arm": arm, "d": d, "seed": seed, "params": P.n_params(m)}
    with torch.no_grad():
        for nm, (kk, syms) in {"in": (K, "tr"), "ood": (K, "swap"), "len": (2 * K, "tr")}.items():
            ces, accs, a1s = [], [], []
            for _ in range(6):
                x, mk = batch(er, 16, kk, syms, nq); lg = fwd(m, arm, x, 0.1); ce, ac, a1 = q_metrics(lg, x, mk); ces.append(ce); accs.append(ac); a1s.append(a1)
            if arm == "CKB" and P.LOOK and nm == "in":
                W = torch.stack(m._w2, 1); qpos = mk[:, 1:] == 1; res["hop2w_at_q"] = [round(v, 3) for v in W[:, :-1][qpos].mean(0).tolist()]; res["hop2w_all"] = [round(v, 3) for v in W.mean((0, 1)).tolist()]
            if arm == "CKB" and nm == "in":   # C124b diagnostic: does the learner's boundary fire at DOT? (b-rate at DOT vs elsewhere)
                with torch.no_grad():
                    _, bsx, _ = m(x, tau=tau, gmin=0.0, states=True); bsx = bsx.squeeze(-1)
                    res["b_dot"] = round(bsx[x == DOT].mean().item(), 3); res["b_other"] = round(bsx[(x != DOT) & (x != PAD)].mean().item(), 3); res["b_hard_dot"] = round((bsx[x == DOT] > 0.5).float().mean().item(), 3)
            if arm == "CKB" and P.INNER and nm == "in" and HOP in (2, 3, 4) and m._jw:   # C123 diagnostic: where does each hop land?
                AH = torch.stack(m._ah, 1); JW = torch.stack(m._jw, 1); xs = x.tolist(); hits = {}
                for bb in range(x.shape[0]):
                    s = xs[bb]; nf = next((i for i, v in enumerate(s) if v == QQ), len(s)) // 6
                    facts = {(s[6 * f], s[6 * f + 1]): f for f in range(nf)}
                    t = nf * 6
                    while t + 7 < len(s):
                        if s[t] == QQ and s[t + 1] == QQ:
                            a = (s[t + 2], s[t + 3]); tq = t + 4
                            if a in facts:
                                fa = facts[a]; b = (s[6 * fa + 3], s[6 * fa + 4])
                                if b in facts:
                                    fb = facts[b]; want = {"a_b0": 6 * fa + 3, "a_b1": 6 * fa + 4, "a_dot": 6 * fa + 5, "b_v0": 6 * fb + 3}
                                    for hk in range(AH.shape[2]):
                                        for nmk, tw in want.items():
                                            key = (hk, nmk); hits.setdefault(key, [0, 0]); hits[key][1] += 1; hits[key][0] += int(AH[bb, tq, hk] == JW[bb, tw])
                            t += 8
                        else: t += 7
                res["hopland"] = {f"h{k}_{n}": round(v[0] / max(v[1], 1), 3) for (k, n), v in sorted(hits.items())}
            if arm == "CKB" and P.INNER and nm == "in":
                W = torch.stack(m._w, 1); qpos = mk[:, 1:] == 1; res["hopw_at_q"] = [round(v, 3) for v in W[:, :-1][qpos].mean(0).tolist()]; res["hopw_all"] = [round(v, 3) for v in W.mean((0, 1)).tolist()]; res["im_loss"] = round(float(m.im_loss), 4)
            res[nm + "_ce"] = round(sum(ces) / 6, 3); res[nm + "_acc"] = round(sum(accs) / 6, 3); res[nm + "_a1"] = round(sum(a1s) / 6, 3)
    MIXEVAL = False
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="GRU,TF,CKB"); ap.add_argument("--ds", default="16,32,64"); ap.add_argument("--seeds", default="1")
    ap.add_argument("--steps", type=int, default=400); ap.add_argument("--B", type=int, default=16); ap.add_argument("--K", type=int, default=6); ap.add_argument("--nq", type=int, default=4)
    ap.add_argument("--bmode", default="rand"); ap.add_argument("--randp", type=float, default=0.1); ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--lowmem", type=int, default=1); ap.add_argument("--knorm", type=int, default=0); ap.add_argument("--dscale", type=int, default=0); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--tag", default="p51"); ap.add_argument("--hop", type=int, default=1); ap.add_argument("--inner", type=int, default=0); ap.add_argument("--repose", type=int, default=0); ap.add_argument("--hopx", type=float, default=0.0); ap.add_argument("--look", type=int, default=0); ap.add_argument("--lookh", type=int, default=3); ap.add_argument("--imcons", type=float, default=0.0); ap.add_argument("--out", default="")
    a = ap.parse_args(); torch.set_num_threads(a.threads); global HOP; HOP = a.hop
    P.BMODE, P.RANDP, P.MSLOTS, P.LOWMEM, P.KNORM, P.DSCALE, P.INNER, P.IMCONS, P.LOOK, P.LOOKH, P.HOPX, P.REPOSE = a.bmode, a.randp, a.M, a.lowmem, a.knorm, a.dscale, a.inner, a.imcons, a.look, a.lookh, a.hopx, a.repose
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
                log(f"[{a.tag} {arm} d{d} s{seed}] params {r['params']} in {r['in_ce']}/{r['in_acc']} ood {r['ood_ce']}/{r['ood_acc']} len {r['len_ce']}/{r['len_acc']} | first-symbol acc in/ood/len {r['in_a1']}/{r['ood_a1']}/{r['len_a1']}" + (f" | hopw@q {r.get('hopw_at_q')} hopw_all {r.get('hopw_all')}" if 'hopw_at_q' in r else "") + (f" | hop2w@q {r.get('hop2w_at_q')} all {r.get('hop2w_all')}" if 'hop2w_at_q' in r else ""))
    log("[P51] " + json.dumps(rows))


if __name__ == "__main__":
    main()
