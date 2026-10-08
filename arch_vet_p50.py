#!/usr/bin/env python3
"""C104 / P50 — COUNTERFACTUAL TRANSPORT (CT): learn the rule as the
EXACT effect of an intervention, not as two cases and not as "be different".

For paired worlds differing only by the determining fact k:v -> k:v',
CT requires the answer distribution at queries of k to transform by the
known value-label permutation v<->v'; distributions at queries of every
other key must stay invariant. CE anchors both worlds. The hypothesis is
that a rule is the transport shared by many cases, while a lookup is not.

Novelty search (2026-10-08):
- CSR / Causal Consistency Regularization, arXiv:2509.01544, maximizes
  answer distance after corrupting a reasoning operator. Limitation: any
  changed answer satisfies it; CT specifies the correct change + locality.
- Double Counterfactual Consistency, arXiv:2602.16787, is an inference /
  reward-time restore-cycle criterion, not dependency-local transport.
- Names Don't Matter, arXiv:2601.23169, and Symbol-Equivariant RRMs,
  arXiv:2603.02193, hard-wire symbol permutation equivariance for known
  symbol classes. CT instead identifies fact->prediction causal dependence.
- C94 in this repo killed latent invariance/distance: state had no binding
  place and arbitrary distance did not specify the consumer effect.

Arms use identical paired batches: fair TFMicro and the learned SBC state,
each with CE alone or CE+CT. Three sizes; slope is the decision metric.
"""
import sys, json, random, time, math, argparse
import torch, torch.nn as nn, torch.nn.functional as F
sys.argv_p50 = list(sys.argv); sys.argv = ["x"]
import arch_vet_lm as A
import arch_vet_p46 as T
import arch_vet_p47 as P
sys.argv = sys.argv_p50
torch.set_num_threads(2)


def recalc(ev):
    cur, out = {}, []
    for e in ev:
        if e[0] == "F":
            cur[e[1]] = e[2]; out.append(e)
        else:
            out.append(("Q", e[1], cur[e[1]]))
    return out


def render_with_layout(ev, fills):
    toks, ans = [], []
    for e, fs in zip(ev, fills):
        toks.extend(fs)
        if e[0] == "F": toks += [T.KEY0 + e[1], T.VAL0 + e[2]]
        else:
            toks += [T.Q, T.KEY0 + e[1]]; ans.append(len(toks)); toks.append(T.VAL0 + e[2])
    return toks, ans


def make_cf_batch(rng, B, nfacts_rng=(2, 4), nq=2, run_max=4):
    xa, xb, anss, meta = [], [], [], []
    for _ in range(B):
        ev = T.structure(rng, rng.randint(*nfacts_rng), nq)
        qk = next(e[1] for e in ev if e[0] == "Q")
        fi = max(i for i, e in enumerate(ev) if e[0] == "F" and e[1] == qk)
        old = ev[fi][2]; new = rng.choice([v for v in range(T.NV) if v != old])
        ev2 = list(ev); ev2[fi] = ("F", qk, new); ev2 = recalc(ev2)
        fills = [[T.FIL0 + rng.randrange(T.NF) for _ in range(rng.randint(0, run_max))] for _ in ev]
        a, apos = render_with_layout(ev, fills); b, bpos = render_with_layout(ev2, fills)
        assert apos == bpos and len(a) == len(b)
        mm = []
        qi = 0
        for e1, e2 in zip(ev, ev2):
            if e1[0] == "Q":
                mm.append((apos[qi], T.VAL0 + e1[2], T.VAL0 + e2[2], e1[2] != e2[2])); qi += 1
        xa.append(a); xb.append(b); anss.append(apos); meta.append(mm)
    L = max(max(map(len, xa)), max(map(len, xb)))
    return {"x": T.pad(xa, L), "xc": T.pad(xb, L), "ans": anss, "meta": meta}


def fwd(m, x, is_sbc, Meval=None):
    return m(x, M=Meval) if is_sbc else m(x)


def ct_objective(m, bt, is_sbc, lam):
    la, lb = fwd(m, bt["x"], is_sbc), fwd(m, bt["xc"], is_sbc)
    ce = .5 * (T.ce_loss(la, bt["x"], bt["ans"]) + T.ce_loss(lb, bt["xc"], bt["ans"]))
    if not lam: return ce, ce.detach(), torch.tensor(0.)
    tr = la.new_tensor(0.); n = 0
    for i, rows in enumerate(bt["meta"]):
        for pos, old, new, changed in rows:
            lpa = F.log_softmax(la[i, pos - 1], -1); lpb = F.log_softmax(lb[i, pos - 1], -1)
            ta = lpa.detach().exp().clone(); tb = lpb.detach().exp().clone()
            if changed:
                # base -> counterfactual swaps precisely old/new; inverse is same permutation.
                ta[old], ta[new] = ta[new].clone(), ta[old].clone()
                tb[old], tb[new] = tb[new].clone(), tb[old].clone()
            tr = tr + .5 * (F.kl_div(lpb, ta, reduction="sum") + F.kl_div(lpa, tb, reduction="sum")); n += 1
    tr = tr / max(n, 1)
    return ce + lam * tr, ce.detach(), tr.detach()


def train(arm, d, seed, steps, B, log, M=4, lam=1.0):
    torch.manual_seed(seed); rng = random.Random(seed)
    is_sbc = arm.startswith("SBC"); m = P.SBC(d, M) if is_sbc else A.TFMicro(T.V, d)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3); t0 = time.time()
    for s in range(steps):
        bt = make_cf_batch(rng, B)
        if is_sbc:
            tau = 0.1 ** (s / max(steps - 1, 1)); P.GMIN0 = .2 * (1 - s / max(steps - 1, 1))
            # SBC.forward gets gmin explicitly; use a tiny wrapper here instead of mutating architecture.
            la = m(bt["x"], tau=tau, gmin=P.GMIN0); lb = m(bt["xc"], tau=tau, gmin=P.GMIN0)
            ce = .5 * (T.ce_loss(la, bt["x"], bt["ans"]) + T.ce_loss(lb, bt["xc"], bt["ans"]))
            if arm.endswith("CT"):
                tr = la.new_tensor(0.); n = 0
                for i, rows in enumerate(bt["meta"]):
                    for pos, old, new, changed in rows:
                        lpa = F.log_softmax(la[i, pos-1], -1); lpb = F.log_softmax(lb[i, pos-1], -1)
                        ta, tb = lpa.detach().exp().clone(), lpb.detach().exp().clone()
                        if changed:
                            ta[old],ta[new]=ta[new].clone(),ta[old].clone(); tb[old],tb[new]=tb[new].clone(),tb[old].clone()
                        tr += .5*(F.kl_div(lpb,ta,reduction="sum")+F.kl_div(lpa,tb,reduction="sum")); n += 1
                tr = tr/max(n,1); loss = ce + lam*tr
            else: tr=ce.new_tensor(0.); loss=ce
        else:
            loss, ce, tr = ct_objective(m, bt, False, lam if arm.endswith("CT") else 0.)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if s % 400 == 0 or s == steps-1:
            log(f"  [{arm} d{d} s{seed}] step {s} ce {float(ce):.3f} ct {float(tr):.3f} ({time.time()-t0:.0f}s)")
    return m


@torch.no_grad()
def acc(m, batches, is_sbc, Meval=8):
    m.eval(); c=t=0
    for bt in batches:
        p=fwd(m,bt["x"],is_sbc,Meval if is_sbc else None).argmax(-1)
        for i, poss in enumerate(bt["ans"]):
            for s in poss: c += int(p[i,s-1].item()==bt["x"][i,s].item()); t += 1
    m.train(); return c/max(t,1)


@torch.no_grad()
def intervention_score(m, batches, is_sbc, Meval=8):
    m.eval(); ex=nex=stable=nst=0
    for bt in batches:
        pa=fwd(m,bt["x"],is_sbc,Meval if is_sbc else None).argmax(-1)
        pb=fwd(m,bt["xc"],is_sbc,Meval if is_sbc else None).argmax(-1)
        for i, rows in enumerate(bt["meta"]):
            for pos, old, new, changed in rows:
                if changed:
                    ex += int(pa[i,pos-1].item()==old and pb[i,pos-1].item()==new); nex += 1
                else:
                    stable += int(pa[i,pos-1].item()==pb[i,pos-1].item()); nst += 1
    m.train(); return ex/max(nex,1), stable/max(nst,1)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--arms",default="TF,TFCT,SBC,SBCCT"); ap.add_argument("--ds",default="16,32,64")
    ap.add_argument("--seeds",default="1"); ap.add_argument("--steps",type=int,default=2000); ap.add_argument("--B",type=int,default=16); ap.add_argument("--lam",type=float,default=1.0)
    a=ap.parse_args(); log=lambda *x: print(*x,flush=True); P.KQ_FROM="win"; P.ST=False
    def ev(seed,nfr,nq,run,n=12):
        r=random.Random(seed); return [T.make_batch(r,16,nfr,nq,run,False) for _ in range(n)]
    evs={"in":ev(999,(2,4),2,4),"len":ev(998,(2,4),2,30),"cnt":ev(997,(8,8),2,4),"both":ev(996,(8,8),2,30),"far":ev(995,(8,8),2,100,6)}
    rr=random.Random(994); cf_ev=[make_cf_batch(rr,16) for _ in range(12)]
    res={"tag":"ARCH-VET-LM-P50","protocol":__doc__[:1800],"runs":[]}
    for arm in a.arms.split(","):
        for d in map(int,a.ds.split(",")):
            for seed in map(int,a.seeds.split(",")):
                m=train(arm,d,seed,a.steps,a.B,log,lam=a.lam); sbc=arm.startswith("SBC")
                r={"arm":arm,"d":d,"seed":seed,"params":T.n_params(m)}
                r.update({k:acc(m,v,sbc) for k,v in evs.items()}); r["cf_exact"],r["cf_stable"]=intervention_score(m,cf_ev,sbc)
                res["runs"].append(r); log(f"[p50 {arm} d{d} s{seed}] params {r['params']} in {r['in']:.3f} len {r['len']:.3f} cnt {r['cnt']:.3f} both {r['both']:.3f} far {r['far']:.3f} | CF exact/stable {r['cf_exact']:.3f}/{r['cf_stable']:.3f}")
    res["slopes"]={}
    for arm in a.arms.split(","):
        rs=[r for r in res["runs"] if r["arm"]==arm]
        if len(rs)>=2:
            xs=[math.log2(r["params"]) for r in rs]; mx=sum(xs)/len(xs)
            for met in ("in","len","cnt","both","far","cf_exact"):
                ys=[r[met] for r in rs]; my=sum(ys)/len(ys)
                res["slopes"][f"{arm}_{met}"]=round(sum((x-mx)*(y-my) for x,y in zip(xs,ys))/max(sum((x-mx)**2 for x in xs),1e-9),4)
    log("[P50] slopes "+json.dumps(res["slopes"])); open("log.jsonl","a").write(json.dumps(res)+"\n"); log("[P50] DONE")
if __name__=="__main__": main()
