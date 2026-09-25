"""ARCH-VET P42 (cycle 83) — k-HOP POINTER CHASING with ITERATED EXACT DEREFERENCE (regime R6).
Task: bindings k_i -> v_i where every value is itself one of the sampled keys (a random functional graph on n keys);
query = A k k ... k ans, where the key is repeated h times and ans = f^h(k). Train h in {1,2}, n in [2,4];
eval h in {1,2,3,4,6,8} at n = 6 (hop count 4x OOD, binding count 1.5x OOD), hard gaps.
Mechanism (grammar-free, zero new params): in the keyed register bank a REPEATED token dereferences the previous
read result instead of itself (structural rule 'repeat = continue the chase'); the value id is recovered exactly
by nearest-embedding lookup of the stored cell (stored cell == E(v) bit-for-bit). Hop count is therefore spelled
by run length, unbounded at test time, O(1) per token.
Why: Sanford-Hsu-Telgarsky 2024 (ICML, arXiv 2402.09268): transformers solve k-hop in O(log k) depth (pointer
doubling) and recurrent/SSM models need Omega(k) sequential steps; a fixed-depth TF trained on h<=2 must
extrapolate the doubling circuit to h=8 — the falsifiable prediction is that it cannot, while the sequential exact
dereference generalises in h by construction once the 1-hop read is learned.
Write predicate: learned (hindsight). HONESTY: the P37 bigram-recurrence label does not cover chain intermediates
(the query never repeats the bigram (k_i,v_i) for h>=2), so R6 uses a PATH-HINDSIGHT label: a binding is positive
iff it lies on the dereference path of a later query within W. This label is computed from the stream's own
future (self-supervised) but is task-aware; arm HOP-BIGRAM keeps the P37 label as the ablation.
Controls: TF-ALiBi 2L d48 (42.7k p) and 4L d96 (308k p) on the identical pool/steps/eval."""
import argparse, inspect, json, os, random, time, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19, arch_vet_p9 as p9, arch_vet_p32 as p32, arch_vet_p37 as p37, arch_vet_p38 as p38, arch_vet_p23 as p23
V = p32.V; A = p32.A; T = p32.T; BOS = p32.BOS; EOS = p32.EOS; fill = p32.fill
K16 = p32.K16

def regime6(): return dict(keys=K16, vals=K16, n_lo=2, n_hi=4, h_lo=1, h_hi=2, comp=False, slots=8, n_eval=6, h_eval=(1, 2, 3, 4, 6, 8))

def gen_stream(rng, R, L=256, hard=False, n_fixed=None, h_fixed=None):
    glo, ghi = (24, 48) if hard else (4, 12); x = [BOS]
    while len(x) < L:
        n = n_fixed or rng.randrange(R["n_lo"], R["n_hi"] + 1); ks = rng.sample(R["keys"], n); f = {k: rng.choice(ks) for k in ks}
        seg = [T, T] + [t for k in ks for t in (k, f[k])] + [fill(rng) for _ in range(rng.randrange(glo, ghi + 1))]
        qs = ks[:]; rng.shuffle(qs)
        for k in qs:
            h = h_fixed or rng.randrange(R["h_lo"], R["h_hi"] + 1); a = k
            for _ in range(h): a = f[a]
            seg += [A] + [k] * h + [a]
        if len(x) + len(seg) > L: x += [fill(rng)] * (L - len(x)); break
        x += seg
    x = x[:L]; x.append(EOS); return x

def path_hindsight_labels(x, W=160):
    """lab[b,t]=1 iff the bigram (x_t, x_{t+1}) is a binding that lies on the dereference path of a later query (A k^h a) within W."""
    B, L = x.shape; lab = torch.zeros(B, L); xs = x.tolist(); keys = set(K16)
    for bi in range(B):
        seq = xs[bi]; t = 1
        while t < L - 1:
            if seq[t] == T and seq[t + 1] == T:
                t += 2; f = {}; pos = {}
                while t + 1 < L and seq[t] in keys and seq[t + 1] in keys and seq[t] not in f: f[seq[t]] = seq[t + 1]; pos[seq[t]] = t; t += 2
                start = t
                while t < L - 1 and seq[t] != T:
                    if seq[t] == A and t + 1 < L:
                        k = seq[t + 1]; h = 0; u = t + 1
                        while u < L and seq[u] == k: h += 1; u += 1
                        a = k
                        for _ in range(h):
                            if a in pos and pos[a] >= start - W: lab[bi, pos[a]] = 1.0
                            a = f.get(a, a)
                        t = u
                    else: t += 1
            else: t += 1
    return lab

@torch.no_grad()
def acc_hop(model, R, n_streams, L, rng, h):
    ok = tot = 0
    for _ in range(n_streams):
        x = gen_stream(rng, R, L, True, R["n_eval"], h); pred = model(torch.tensor(x[:L]).unsqueeze(0)).argmax(-1).squeeze(0)
        t = 1
        while t < L - 2:
            if x[t] == A:
                k = x[t + 1]; u = t + 1
                while u < L and x[u] == k: u += 1
                if u < L: tot += 1; ok += int(int(pred[u - 1]) == x[u])
                t = u
            else: t += 1
    return round(ok / max(1, tot), 4)

# ---- KRBHop: KRBBlocked.forward with the 'repeat = continue chase' rule patched in (source-level, so the certified write path is byte-identical)
_src = inspect.getsource(p38.KRBBlocked.forward)
_src = _src.replace("        kgl_list = []; kg_prev = torch.zeros(B); lg = torch.empty(B, L, V)",
                    "        kgl_list = []; kg_prev = torch.zeros(B); lg = torch.empty(B, L, V); rid = torch.full((B,), -1, dtype=torch.long)")
_src = _src.replace("            i1 = H1[:, t]; i2 = H2[:, t]; m1 = tags[ar, i1] == xid.unsqueeze(-1); m2 = tags[ar, i2] == xid.unsqueeze(-1)   # B,b",
                    "            rep = (t > 0) & (rid >= 0) & (xid == x[:, max(t - 1, 0)]); qid = torch.where(rep, rid.clamp(min=0), xid)\n"
                    "            i1 = self.HF1[qid].argmax(-1); i2 = self.HF2[qid].argmax(-1); m1 = tags[ar, i1] == qid.unsqueeze(-1); m2 = tags[ar, i2] == qid.unsqueeze(-1)   # B,b")
_src = _src.replace("            R = (1 - read_on) * R + read_on * cand\n",
                    "            R = (1 - read_on) * R + read_on * cand\n"
                    "            nid = ((cand.unsqueeze(1) - self.E.weight.unsqueeze(0)) ** 2).sum(-1).argmin(-1); rid = torch.where(hit.squeeze(-1) > 0.5, nid, torch.full_like(nid, -1))\n")
assert _src.count("qid") >= 3 and "nid" in _src, "patch failed"
_ns = dict(p38.__dict__); exec("def _hop_forward" + _src[_src.index("(self"):], _ns)
class KRBHop(p38.KRBBlocked):
    forward = _ns["_hop_forward"]

def train(name, model, pool, steps, labfn, batch=8, lr=3e-3, lam=0.5, ck=None):
    torch.manual_seed(0); opt = torch.optim.AdamW(model.parameters(), lr=lr); model.train(); t0 = time.time(); n_pool = len(pool); start = 1
    if ck and os.path.exists(ck):
        st = torch.load(ck); model.load_state_dict(st["sd"]); opt.load_state_dict(st["opt"]); torch.set_rng_state(st["rng"]); start = st["step"] + 1; print(f"  [{name}] RESUMED at {start-1}", flush=True)
    labs = {}
    for step in range(start, steps + 1):
        sel = [(step * batch + i) % n_pool for i in range(batch)]; x = torch.stack([pool[i] for i in sel]); y = x[:, 1:]; xin = x[:, :256]
        lg = model(xin); ce = F.cross_entropy(lg.reshape(-1, V), y.reshape(-1))
        key = tuple(sel)
        if key not in labs: labs[key] = labfn(xin)
        bce = F.binary_cross_entropy_with_logits(model.kg_logits, labs[key]); loss = ce + lam * bce
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 250 == 0:
            print(f"  [{name}] step {step}/{steps} ce {float(ce):.4f} bce {float(bce):.3f} ({time.time()-t0:.0f}s)", flush=True)
            if ck: torch.save({"sd": model.state_dict(), "opt": opt.state_dict(), "rng": torch.get_rng_state(), "step": step}, ck)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="HOP"); ap.add_argument("--seed", type=int, default=111); ap.add_argument("--steps", type=int, default=4000); a = ap.parse_args()
    R = regime6(); rng = random.Random(12345); pool = [torch.tensor(gen_stream(rng, R, 256)) for _ in range(512)]
    out = {"tag": "ARCH-VET-LM-P42", "protocol": __doc__[:1800], "seed": a.seed, "steps": a.steps, "arm": a.arm}
    torch.manual_seed(a.seed)
    if a.arm.startswith("HOP"):
        m = KRBHop(V, 24, R["slots"], b=4, smart=True); labfn = p37.hindsight_labels if a.arm == "HOP-BIGRAM" else path_hindsight_labels
        print(f"[p42] R6:{a.arm} params {p19.n_params(m)}", flush=True)
        train(f"P42-{a.arm}", m, pool, a.steps, labfn, ck=f"p21_ckpt/P42_{a.arm}_s{a.seed}.resume.pt" if a.steps >= 1000 else None)
    else:
        m = p23.TFCtrl(V, pe="alibi") if a.arm == "TF-alibi" else p23.TFCtrl(V, d=96, nh=4, depth=4, mlp=2, pe="alibi")
        print(f"[p42] R6:{a.arm} params {p19.n_params(m)}", flush=True); p19.train_arm(f"P42-{a.arm}", m, pool, a.steps, 8)
    m.eval(); r = {f"h{h}": acc_hop(m, R, 10, 320, random.Random(600 + h), h) for h in R["h_eval"]}; r["params"] = p19.n_params(m)
    print(f"[p42 R6:{a.arm} s{a.seed}] {r}", flush=True); out["result"] = r
    torch.save({"sd": m.state_dict()}, f"p21_ckpt/P42_{a.arm}_s{a.seed}.pt"); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P42] DONE", flush=True)

if __name__ == "__main__":
    main()
