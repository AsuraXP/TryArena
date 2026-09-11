"""ARCH-VET P28 (cycle 65) — IN-RANGE PARITY DIAGNOSIS: decompose the
256-hard CE gap (unified VET 2.5-2.9 vs TF ~2.0) into irreducible
generator entropy vs model error, PER TOKEN CLASS, and locate the gap.
Why: it is the single axis where the Transformer still leads. Before
building anything, measure whether the gap is (a) at deterministic
(answer/close) positions = a reasoning failure, or (b) at stochastic
positions (fillers, task choice, key/value/count draws) = a density-
modelling gap that carries no reasoning content.
Method: for each eval stream (256-hard / 256-train / 1024), label each
target position by class {det: answers/closes/structural continuations,
stoch: filler bytes, task-header draw, key/val/count draw}; compute the
exact per-class entropy of the generator (fills: log 8; key/val: log 4;
task choice: -sum p log p over the 4 types; etc.) and the model NLL per
class for A-alone, the unified learned-gate system, and the P23
controls. Report gap_det and gap_stoch. Tag ARCH-VET-LM-P28.
Prior art (2026-09-11): arXiv 2204.09636 (Residual MoE) and arXiv
2609.00575 (residual sparsification) — residual/base decompositions of
expert PARAMETERS; here the decomposition is of the LOSS by token
determinism, to decide whether a residual density expert is needed.
"""
import json, math, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
CKPT = os.path.join(REPO, "p21_ckpt")
import arch_vet_lm as lm, arch_vet_p19 as p19, arch_vet_p21 as p21, arch_vet_p22 as p22, arch_vet_p23 as p23
V = p19.V; BOS, EOS, A, T = p19.BOS, p19.EOS, p19.A, p19.T_TASK
MODS, TRACK, ONE, MANS, BRK, KEYS, VALS = p19.MODS, p19.TRACK, p19.ONE, p19.MANS, p19.BRK, p19.KEYS, p19.VALS


def classify(x):
    """Per target position t (predict x[t+1] from x[:t+1]) -> class label.
    det = fully determined by the grammar; stoch = a generator draw."""
    L = len(x) - 1; cls = ["stoch_other"] * L
    i = 1
    while i < len(x) - 1:
        tok = x[i]
        if tok != T:
            i += 1; continue
        # count T run
        r = 0
        while i + r < len(x) and x[i + r] == T: r += 1
        if r == 1 and i + 1 < len(x) and TRACK <= x[i + 1] < TRACK + 8:    # TRACK
            cls[i] = "stoch_sym"                       # predicting x[i+1] = symbol draw
            j = i + 2
            while j < len(x) - 1 and x[j] != A: cls[j - 1] = "stoch_fill_or_A"; j += 1
            if j < len(x) - 1:
                cls[j - 1] = "stoch_fill_or_A"           # position before A: fill or A (gap draw)
                cls[j] = "det_answer"                    # A -> symbol
            i = j + 2
        elif r == 2 and i + 2 < len(x) and x[i + 2] == ONE:                  # MODK
            cls[i + 1] = "stoch_task"                    # T T -> ONE/bracket/key = task type draw
            j = i + 2
            while j < len(x) - 1 and x[j] == ONE: cls[j] = "stoch_count"; j += 1   # ONE -> ONE or A
            if j < len(x) - 1 and x[j] == A: cls[j] = "det_answer"
            i = j + 2
        elif r == 2 and i + 2 < len(x) and KEYS <= x[i + 2] < KEYS + 4:      # PAIR
            cls[i + 1] = "stoch_task"; cls[i + 2] = "stoch_kv"; cls[i + 3] = "stoch_fill_or_A"
            j = i + 4
            while j < len(x) - 1 and x[j] != A: cls[j - 1] = "stoch_fill_or_A"; j += 1
            if j < len(x) - 1:
                cls[j - 1] = "stoch_fill_or_A"; cls[j] = "det_answer"; 
                if j + 1 < L: cls[j + 1] = "det_answer"
            i = j + 3
        elif r >= 3:                                                          # DYCK
            cls[i + 1] = "stoch_task"; cls[i + 2] = "stoch_dyck_open"
            j = i + 3; d = 0
            while j < len(x) - 1:
                t2 = x[j]
                if BRK <= t2 < BRK + 4:
                    d += 1 if t2 < BRK + 2 else -1
                nxt = x[j + 1] if j + 1 < len(x) else None
                if nxt is not None and BRK + 2 <= nxt < BRK + 4: cls[j] = "det_close"
                elif nxt is not None and BRK <= nxt < BRK + 2: cls[j] = "stoch_dyck_open"
                else: cls[j] = "stoch_other"
                if d == 0: break
                j += 1
            i = j + 1
        else:
            i += r
    # T_TASK-header positions (predicting first T of a task after previous task) = stoch_task
    for t in range(L):
        if x[t + 1] == T and cls[t] in ("stoch_other", "stoch_fill_or_A") and x[t] != T: cls[t] = "stoch_taskstart"
    return cls


@torch.no_grad()
def nll_by_class(logits_fn, xs, L):
    agg = {}
    for b in range(xs.shape[0]):
        x = xs[b, :L + 1].tolist()
        lg = logits_fn(xs[b:b + 1, :L + 1])[0, :L]
        nll = -F.log_softmax(lg, -1).gather(-1, xs[b, 1:L + 1].unsqueeze(-1)).squeeze(-1).tolist()
        for t, c in enumerate(classify(x)):
            a = agg.setdefault(c, [0.0, 0]); a[0] += nll[t]; a[1] += 1
    tot = sum(v[0] for v in agg.values()) / sum(v[1] for v in agg.values())
    return {"total": round(tot, 4), **{k: {"nll": round(v[0] / v[1], 4), "n": v[1]} for k, v in sorted(agg.items())}}


def det_stoch(d):
    det = sum(d[k]["nll"] * d[k]["n"] for k in d if k.startswith("det")); nd = sum(d[k]["n"] for k in d if k.startswith("det"))
    st = sum(d[k]["nll"] * d[k]["n"] for k in d if k.startswith("stoch")); ns = sum(d[k]["n"] for k in d if k.startswith("stoch"))
    return {"det_nll": round(det / max(1, nd), 4), "det_n": nd, "stoch_nll": round(st / max(1, ns), 4), "stoch_n": ns,
            "det_share_of_total_loss": round(det / max(1e-9, det + st), 4)}


def main():
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P28", "protocol": __doc__[:900], "arms": {}}
    vha = p19.make_batches(8, 256, random.Random(777), hard=True)
    vtr = p19.make_batches(8, 256, random.Random(999))
    v1024 = p19.make_batches(2, 1024, random.Random(31415))
    sets = {"256_hard": (vha, 256), "256_train": (vtr, 256), "1024": (v1024, 1024)}
    # oracle floor per stochastic class (exact generator entropy where closed-form)
    out["generator_entropy_nats"] = {"fill": round(math.log(8), 4), "kv": round(math.log(4), 4), "sym": round(math.log(8), 4),
                                     "task": round(math.log(4), 4), "note": "count/gap-length draws have geometric-like entropy ~1-2 nats"}
    for seed in (111, 222):
        torch.manual_seed(seed)
        mA = p21.VETDCC(V, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V, 24, k=8, K=8); g = p22.Gate(V, 8)
        mA.load_state_dict(torch.load(f"{CKPT}/A_s{seed}.pt")["sd"]); mB.load_state_dict(torch.load(f"{CKPT}/B_s{seed}.pt")["sd"]); g.load_state_dict(torch.load(f"{CKPT}/G_s{seed}.pt")["sd"])
        mA.eval(); mB.eval(); g.eval()
        p21.causal_route = (lambda toks: p22.gate_routes(g, toks))
        arms = {"A-alone": (lambda x: mA(x)), "unified-gate": (lambda x: p21.routed_logits(mA, mB, x))}
        for name, fn in arms.items():
            r = {}
            for sn, (xs, L) in sets.items():
                d = nll_by_class(fn, xs, L); r[sn] = {**d, **det_stoch(d)}
            out["arms"].setdefault(name, {})[str(seed)] = r
            print(f"[p28 {name} s{seed}] " + " | ".join(f"{sn}: tot {r[sn]['total']} det {r[sn]['det_nll']}(n{r[sn]['det_n']}) stoch {r[sn]['stoch_nll']}" for sn in sets), flush=True)
    # TF controls: retrain P23 NoPE/ALiBi quickly? They were not saved -> train TF-ALiBi d48 (best ratio) + TF-NoPE 6000 steps on the union pool, 1 seed each
    pool = p23.joint_pool()
    for pe in ("alibi", "nope"):
        torch.manual_seed(111); m = p23.TFCtrl(V, pe=pe); pth = f"{CKPT}/P23_{pe}_s111.pt"
        if os.path.exists(pth): m.load_state_dict(torch.load(pth)["sd"])
        else: p19.train_arm(f"P28-TF-{pe}", m, pool, 6000, 8); torch.save({"sd": m.state_dict()}, pth)
        m.eval(); r = {}
        for sn, (xs, L) in sets.items():
            d = nll_by_class(lambda x: m(x), xs, L); r[sn] = {**d, **det_stoch(d)}
        out["arms"][f"TF-{pe}"] = {"111": r}
        print(f"[p28 TF-{pe} s111] " + " | ".join(f"{sn}: tot {r[sn]['total']} det {r[sn]['det_nll']}(n{r[sn]['det_n']}) stoch {r[sn]['stoch_nll']}" for sn in sets), flush=True)
    # per-class table for 256_hard
    for name, per in out["arms"].items():
        r = per[list(per)[0]]["256_hard"]
        print(f"[p28 classes 256_hard {name}] " + " ".join(f"{k}={v['nll']}/{v['n']}" for k, v in r.items() if isinstance(v, dict)), flush=True)
    out["wall_s"] = round(time.time() - t0)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P28] DONE", flush=True)


if __name__ == "__main__":
    main()
