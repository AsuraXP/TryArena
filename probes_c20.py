"""
ARC-2 CYCLE 20 / GENERALIZATION PROBES — certified at the certified point;
what TRANSFERS zero-shot? (eval-only; loads the C19 machine v7 final ckpt)
============================================================================
The C18/C19 verdicts certify each family AT the trained distribution.
C20 closes the logged debt ("transfer unproven") with five zero-shot
probes on the untouched v7 artifact (no fine-tuning):
  P1  mod5 walk (values 0..4, wrap 4->0, tokens 38..42): the trained
      branch is the mod7 ring; wrap-4 is a NOVEL transition. Tests ring
      generalization of the finite-state host.
  P2  mod6 walk (wrap 5->0): one transition removed from mod7 — the
      intermediate case (control ladder: mod7 in-machine = ~0.0034).
  P3  ICL multi-query: 3 queries per row on ONE mapping (trained data
      has exactly 1 terminal query). Tests register persistence across
      reads.
  P4  ICL redefinition: a key re-presented with a NEW value; query
      answer = LATEST value (trained repeats keep the same value).
      Tests overwrite semantics of the SRAM organ.
  P5  kstack bottom/deep: k = depth when depth <= 8 (BOTTOM readout,
      the organ's exposure limit) and k = 8 under load (depth 9..16+).
      Per-depth answer CE characterizes the exact capability boundary
      (depth > 8 bottom = structurally out of exposure: no Q token).
  P6  subtraction zero-shot: same add vocabulary, c = (a-b-borrow) mod 10
      with borrow transition (trained transition is carry). Expected
      FAIL — certifies the organ is transition-specific and defines the
      C21 borrow organ.
@16384: P1, P3, P5 re-run (length invariance of whatever transfers).
Controls: in-machine mod7 + ICL single-query must reproduce C19 numbers.
Tag: ARC2-C20-GEN-PROBES. USAGE: OMP_NUM_THREADS=1 python3 -u probes_c20.py
"""
import json, math, random, resource, time
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
t_start = time.time()
LN3, LN16 = math.log(3.0), math.log(16.0)

g = {"__name__": "p20"}
exec(open("unified_kstack8.py").read().split("\nRESULTS = {}")[0], g)
MachineV7, n_params, eval_task = g["MachineV7"], g["n_params"], g["eval_task"]
gen_echo_t, gen_icl_t, gen_mod7_t, gen_add_t, gen_kstack_t = (
    g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"], g["gen_add_t"], g["gen_kstack_t"])
VOCAB, Q0, KD = g["VOCAB"], g["Q0"], g["KD"]
CFG = g["CFG"]

CKPT = "unified_kstack8_final.pt"

# ---------------------------------------------------------------- probe data
def gen_modM_t(M, batch, length, rng):
    """deterministic mod-M walk, r in {1,2,3}; values 0..M-1 on tokens 38..44."""
    base = 38
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x = [base + rng.randrange(M)]
        for _ in range(length):
            r = rng.randrange(1, 4)
            x.append(base + (x[-1] - base + r) % M)
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append([LN3] * length)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

def gen_icl_multiquery_t(batch, length, rng, nq=3):
    """n definition pairs (same mapping as gen_icl) + nq terminal queries."""
    NK = 16; OFF = 6
    assert length % 2 == 0
    n = (length - 2 * nq) // 2
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        mapping = list(range(NK)); rng.shuffle(mapping)
        x, nll, seen = [], [], set()
        for i in range(n):
            k = rng.randrange(NK); v = NK + mapping[k]
            if k not in seen:
                nll.append(LN16 + math.log(NK - len(seen))); seen.add(k)
            else:
                nll.append(LN16)
            x.append(k); x.append(v)
        for q in range(nq):
            qq = rng.randrange(NK)
            x.append(qq); nll.append(LN16)
            x.append(NK + mapping[qq]); nll.append(0.0)
        xs.append(x[:-1]); ys.append(x[1:]); os_.append(nll[1:])
    return torch.tensor(xs) + OFF, torch.tensor(ys) + OFF, torch.tensor(os_)

def gen_icl_redef_t(batch, length, rng):
    """every repeat presentation of a key carries a NEW value (cur = cur+1 mod 16);
    the single terminal query must answer the LATEST value."""
    NK = 16; OFF = 6
    assert length % 2 == 0
    n = (length - 2) // 2
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        mapping = list(range(NK)); rng.shuffle(mapping)
        cur = {}
        x, nll, first_seen = [], [], set()
        for i in range(n):
            k = rng.randrange(NK)
            if k in cur:
                cur[k] = (cur[k] + 1) % NK          # redefinition: new value
                nll.append(LN16)
            else:
                cur[k] = mapping[k]
                nll.append(LN16 + math.log(NK - len(first_seen))); first_seen.add(k)
            x.append(k); x.append(NK + cur[k])
        q = rng.randrange(NK)
        x.append(q); nll.append(LN16)
        x.append(NK + cur[q]); nll.append(0.0)
        xs.append(x[:-1]); ys.append(x[1:]); os_.append(nll[1:])
    return torch.tensor(xs) + OFF, torch.tensor(ys) + OFF, torch.tensor(os_)

def gen_kstack_deep_t(batch, length, rng):
    """triplet stream; k = depth if depth <= KD else KD (bottom readout up to
    the exposure limit, deep-KD readout beyond). Returns meta (B, ngroups)
    arrays (depth_after, k) for the per-depth diagnostic."""
    ngen = (length + 3) // 3
    H_POP = -(2 * 0.3 * math.log(0.3) + 0.4 * math.log(0.4))
    xs, ys, os_, metas = [], [], [], []
    for _ in range(batch):
        x, nll = [], []
        stack = []
        meta = []
        for t in range(ngen):
            d = len(stack)
            op = 2 if (d >= 2 and rng.random() < 0.4) else rng.randrange(2)
            if op in (0, 1):
                if len(stack) < CFG["KSTACK"]:
                    stack.append(op)
            else:
                stack.pop()
            k = min(KD, len(stack))
            ans = stack[-k]
            x.append(op);              nll.append(H_POP if d >= 2 else math.log(2.0))
            x.append(Q0 + k - 1);      nll.append(math.log(float(k)))
            x.append(ans);             nll.append(0.0)
            meta.append((len(stack), k))
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append(nll[1:length + 1])
        metas.append(meta)
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_), metas

def gen_sub_t(batch, length, rng):
    """subtraction on the ADD vocabulary: c = (a - b - borrow) mod 10,
    borrow' = 1 if a - b - borrow < 0. Trained transition is carry (P6)."""
    AD = 45
    ngen = (length + 3) // 3
    xs, ys, os_ = [], [], []
    for _ in range(batch):
        x, nll = [], []
        borrow = 0
        for t in range(ngen):
            a = rng.randrange(10); b = rng.randrange(10)
            c = (a - b - borrow) % 10
            borrow = 1 if (a - b - borrow) < 0 else 0
            x.append(AD + a);      nll.append(math.log(10.0))
            x.append(AD + 10 + b); nll.append(math.log(10.0))
            x.append(AD + 20 + c); nll.append(0.0)
        xs.append(x[:length]); ys.append(x[1:length + 1]); os_.append(nll[1:length + 1])
    return torch.tensor(xs), torch.tensor(ys), torch.tensor(os_)

# ---------------------------------------------------------------- probe run
@torch.no_grad()
def probe(model, gen, L, reps=1, task_id=0, nq=None, nans=None):
    """dCE (repo convention) + answer-position CE (oracle-exact positions).
    ICL-family gens use the house PAIR oracle (o not token-aligned);
    for those, answers = trailing nans tokens (positions -1, -3, ...)."""
    model.eval()
    bs = max(1, min(4, 4096 // L))
    ce = orc = n = 0.0
    ace = an = 0.0
    meta = None
    for i in range(reps):
        rng = random.Random(900_000 + L + i)
        out = gen(bs, L, rng)
        if nq is not None:
            out = gen(bs, L, rng, nq=nq)
        x, y, o = out[0], out[1], out[2]
        if len(out) == 4:
            meta = out[3]
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        if o.numel() == y.numel():
            am = (o == 0)
        else:
            am = torch.zeros_like(y, dtype=torch.bool)
            for i in range(nans if nans else 1):
                am[:, -(2 * i + 1)] = True
        ace += nll[am].sum().item(); an += int(am.sum())
    d = round((ce - orc) / n, 4)
    a = round(ace / max(1, an), 4)
    return d, a, meta

def kstack_perdepth(model, L=4096, reps=1):
    """answer CE per (depth, k) — the boundary diagnostic for P5."""
    model.eval()
    x, y, o, metas = gen_kstack_deep_t(1, L, random.Random(930_000 + L))
    logits, rl = model(x)
    nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    agg = {}
    for gi, (depth, k) in enumerate(metas[0]):
        t = 3 * gi + 1                      # y = x[1:]: answer x[3gi+2] -> y[3gi+1]
        if t >= L:
            break
        key = f"d{min(depth, 16)}{'+' if depth > 16 else ''}/k{k}"
        v = nll[0, t].item()
        c, s = agg.get(key, (0.0, 0))
        agg[key] = (c + v, s + 1)
    out = {}
    for key in sorted(agg):
        c, s = agg[key]
        out[key] = round(c / s, 4)
    return out

# ---------------------------------------------------------------- run
torch.manual_seed(0)
m = MachineV7()
m.load_state_dict(torch.load(CKPT))
m.eval()
print(f"[arm] machine v7 loaded from {CKPT}, params={n_params(m)}", flush=True)
RESULTS = {}

L = 4096
d, a, _ = probe(m, gen_mod7_t, L, 2, task_id=2)
RESULTS["control_mod7"] = {"dCE": d, "answer_CE": a}
print(f"  control mod7 (in-machine): dCE {d} answer {a}", flush=True)
d, a, _ = probe(m, gen_icl_t, L, 2, task_id=1, nans=1)
RESULTS["control_icl_single"] = {"dCE": d, "answer_CE": a}
print(f"  control icl single-query: dCE {d} answer {a}", flush=True)

d, a, _ = probe(m, lambda b, l, r: gen_modM_t(5, b, l, r), L, 2, task_id=2)
RESULTS["P1_mod5_zeroshot"] = {"dCE": d, "answer_CE": a}
print(f"  P1 mod5 zero-shot:  dCE {d} answer {a}   (trained ring = mod7)", flush=True)
d, a, _ = probe(m, lambda b, l, r: gen_modM_t(6, b, l, r), L, 2, task_id=2)
RESULTS["P2_mod6_zeroshot"] = {"dCE": d, "answer_CE": a}
print(f"  P2 mod6 zero-shot:  dCE {d} answer {a}   (one wrap novel)", flush=True)
d, a, _ = probe(m, gen_icl_multiquery_t, L, 1, task_id=1, nq=3, nans=3)
RESULTS["P3_icl_multiquery3"] = {"dCE": d, "answer_CE": a}
print(f"  P3 icl 3 queries:   dCE {d} answer {a}   (trained: 1 query/row)", flush=True)
d, a, _ = probe(m, gen_icl_redef_t, L, 1, task_id=1, nans=1)
RESULTS["P4_icl_redefinition"] = {"dCE": d, "answer_CE": a}
print(f"  P4 icl redefinition: dCE {d} answer {a}   (latest value wins?)", flush=True)
d, a, _ = probe(m, gen_kstack_deep_t, L, 1, task_id=0)
RESULTS["P5_kstack_bottom_deep"] = {"dCE": d, "answer_CE": a}
print(f"  P5 kstack bottom/deep: dCE {d} answer {a}", flush=True)
pd = kstack_perdepth(m, L, 1)
RESULTS["P5_perdepth"] = pd
print(f"  P5 per-depth answer CE: {pd}", flush=True)
d, a, _ = probe(m, gen_sub_t, L, 1, task_id=3)
RESULTS["P6_subtraction_zeroshot"] = {"dCE": d, "answer_CE": a}
print(f"  P6 subtraction zero-shot: dCE {d} answer {a}   (trained transition = carry)", flush=True)

L = 16384
d, a, _ = probe(m, lambda b, l, r: gen_modM_t(5, b, l, r), L, 1, task_id=2)
RESULTS["P1_mod5_16384"] = {"dCE": d, "answer_CE": a}
print(f"  P1 mod5 @16384: dCE {d} answer {a}", flush=True)
d, a, _ = probe(m, gen_icl_multiquery_t, L, 1, task_id=1, nq=3, nans=3)
RESULTS["P3_icl_multiquery3_16384"] = {"dCE": d, "answer_CE": a}
print(f"  P3 icl 3q @16384: dCE {d} answer {a}", flush=True)
d, a, _ = probe(m, gen_kstack_deep_t, L, 1, task_id=0)
RESULTS["P5_kstack_bottom_deep_16384"] = {"dCE": d, "answer_CE": a}
print(f"  P5 kstack deep @16384: dCE {d} answer {a}", flush=True)

print("\n" + "=" * 90)
print("C20 GENERALIZATION PROBES on machine v7 (zero-shot, no fine-tuning)")
print("controls must reproduce C19: mod7 ~0.0034, icl single ~0.0")
print("transfers cleanly / partially / fails => defines C21+ builds")
print("=" * 90)
final = {"tag": "ARC2-C20-GEN-PROBES", "ckpt": CKPT, "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
