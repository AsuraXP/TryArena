"""ARCH-VET P30 (cycle 66) — DISTRIBUTION-SHIFT ROUTING: does the
hierarchical learned gate (M over G) keep dispatching correctly when
the STREAM MIXTURE shifts away from anything it was trained on?
Prior art (2026-09-11): arXiv 2510.16448 (IDA-MoE: routers trained
with task+balance losses give high-entropy, shift-fragile routing);
arXiv 2510.14853 (test-time rerouting because routers fail under
deployment shift). Our gate has NO balance loss and is token-causal;
the hypothesis is that its policy is grammar-local (depends on the
last few tokens' structure, not on mixture statistics), hence shift-
robust. Falsifiable: dispatch identity vs the certified 2-way gate on
symbolic positions, and text->C share, should stay ~1.0 under shift.
SHIFT FAMILIES (all unseen at gate training time):
  S1 text-heavy chatmix (text turn prob .9, turns 100-240 bytes)
  S2 reasoning-heavy chatmix (text prob .1, 5-8 reasoning segs)
  S3 long-turn text (single turns 200-250 bytes, no reasoning)
  S4 L=1024 chatmix (4x the training length)
  S5 rapid alternation (text turns of 8-20 bytes between every task)
  S6 dyck-only deep bursts inside text (depth 5-8 segments + text)
  S7 byte-noise text (10% random bytes) — corrupt-input robustness
Metrics per shift: (a) modality accuracy of M (position is byte-text
vs symbolic, gold from the generator), (b) symbolic dispatch identity
vs G, (c) routed text CE vs C-alone, (d) in-stream reasoning answers +
dyck close via chatmix_reasoning generalised, (e) gate entropy.
Seeds 111/222 (C/M/G from P27). Tag ARCH-VET-LM-P30.
"""
import json, math, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); CKPT = os.path.join(REPO, "p21_ckpt")
import arch_vet_p26 as p26, arch_vet_p26b as p26b, arch_vet_p22 as p22, arch_vet_p21 as p21, arch_vet_p19 as p19, arch_vet_p13d as p13d
V0, VB = p26.V0, p26.VB; U, EOS, T = 2, p19.EOS, p19.T_TASK; TXT = p26.TXT_VA


def text_turn(rng, lo, hi, noise=0.0):
    n = rng.randrange(lo, hi + 1); s = rng.randrange(0, len(TXT) - n)
    b = [V0 + (rng.randrange(256) if rng.random() < noise else c) for c in TXT[s:s + n]]
    return [U] + b + [EOS]


def reasoning_burst(rng, nseg, L=96):
    seg = p19.gen_mixdd_stream(rng, L)[1:-1]
    return seg[:rng.randrange(20, L)] if nseg == 1 else seg


def deep_dyck_burst(rng):
    d = rng.randrange(5, 9); return [T, T, T] + p13d.rt_segment(rng, d)


def shift_stream(rng, kind, L=256):
    x = [p19.BOS]
    while len(x) < L:
        if kind == "S1": x += text_turn(rng, 100, 240) if rng.random() < .9 else reasoning_burst(rng, 1)
        elif kind == "S2": x += text_turn(rng, 20, 60) if rng.random() < .1 else p19.gen_mixdd_stream(rng, 160)[1:-1]
        elif kind == "S3": x += text_turn(rng, 200, 250)
        elif kind == "S4": x += text_turn(rng, 30, 90) if rng.random() < .5 else reasoning_burst(rng, 1)
        elif kind == "S5": x += text_turn(rng, 8, 20) + p19.gen_mixdd_stream(rng, 48)[1:-1][:rng.randrange(8, 40)]
        elif kind == "S6": x += text_turn(rng, 30, 80) if rng.random() < .5 else deep_dyck_burst(rng)
        elif kind == "S7": x += text_turn(rng, 30, 90, noise=.10) if rng.random() < .5 else reasoning_burst(rng, 1)
    x = x[:L]; x.append(EOS); return x


def gold_modality(x):
    """1 where the TARGET token (x[t+1]) is a byte, else 0 — the gate at
    position t must choose C iff the next token is text. Byte targets
    are unambiguous; the U/EOS delimiters are counted as symbolic."""
    return [1 if x[t + 1] >= V0 else 0 for t in range(len(x) - 1)]


@torch.no_grad()
def evaluate(u, mC, g2, kind, n, L, seed):
    rng = random.Random(seed); mod_ok = mod_n = 0; id_ok = id_n = 0; tce = tn = 0.0; ent = 0.0; en = 0
    c_ce = 0.0
    for _ in range(n):
        x = shift_stream(rng, kind, L); xt = torch.tensor(x).unsqueeze(0)
        lg3 = u.g(xt[:, :L]); r3 = lg3.argmax(-1).squeeze(0); pr = F.softmax(lg3, -1).squeeze(0)
        ent += float(-(pr * torch.log(pr + 1e-9)).sum(-1).mean()); en += 1
        gm = gold_modality(x)
        for t in range(L):
            isC = int(r3[t]) == 2
            mod_ok += int(isC == bool(gm[t])); mod_n += 1
        xs = u.sym(xt[:, :L]); r2 = g2(xs).argmax(-1).squeeze(0)
        for t in range(L):
            if gm[t] == 0 and x[t] < V0:
                id_n += 1; id_ok += int(int(r3[t]) == int(r2[t])) if int(r3[t]) != 2 else 0
        lg = u(xt[:, :L]); lc = mC(xt[:, :L]); y = xt[:, 1:L + 1]
        nll = -F.log_softmax(lg, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
        nllc = -F.log_softmax(lc, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1).squeeze(0)
        tb = torch.tensor(gm, dtype=torch.bool)
        if tb.any(): tce += float(nll[tb].sum()); c_ce += float(nllc[tb].sum()); tn += int(tb.sum())
    out = {"modality_acc": round(mod_ok / mod_n, 4), "sym_dispatch_identity": round(id_ok / max(1, id_n), 4), "n_sym": id_n,
           "routed_text_ce": round(tce / max(1, tn), 4), "C_alone_text_ce": round(c_ce / max(1, tn), 4), "gate_entropy": round(ent / en, 4)}
    # in-stream reasoning (families present)
    if kind != "S3":
        old = p26.gen_chatmix_stream
        p26.gen_chatmix_stream = (lambda r, LL=256: shift_stream(r, kind, L))
        try: out["reasoning"] = p26.chatmix_reasoning(u, n, seed=seed + 1)
        finally: p26.gen_chatmix_stream = old
    return out


def main():
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P30", "protocol": __doc__[:1500], "arms": {}}
    for seed in (111, 222):
        mA = p21.VETDCC(V0, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V0, 24, k=8, K=8); mC = p26.ByteGRU(VB, 48); g2 = p22.Gate(V0, 8); gm = p26b.GateM(4)
        for m, f in ((mA, "A"), (mB, "B"), (mC, "C"), (g2, "G"), (gm, "GM")): m.load_state_dict(torch.load(f"{CKPT}/{f}_s{seed}.pt")["sd"]); m.eval()
        u = p26.Unified3(mA, mB, mC, p26b.HierGate(g2, gm)).eval()
        res = {"S0_in_dist": None}
        old = p26.gen_chatmix_stream
        # in-distribution reference (chatmix as trained, val text)
        r0 = evaluate(u, mC, g2, "S4", 16, 256, 5000)   # S4 at L=256 == training chatmix distribution
        res["S0_in_dist"] = r0
        for kind, L, n in (("S1", 256, 16), ("S2", 256, 16), ("S3", 256, 16), ("S4", 1024, 6), ("S5", 256, 16), ("S6", 320, 16), ("S7", 256, 16)):
            res[kind] = evaluate(u, mC, g2, kind, n, L, 6000 + hash(kind) % 100)
            print(f"[p30 s{seed} {kind} L{L}] {res[kind]}", flush=True)
        print(f"[p30 s{seed} S0] {r0}", flush=True)
        out["arms"][str(seed)] = res
    summ = {k: {"modality_acc": [out["arms"][s][k]["modality_acc"] for s in out["arms"]],
                "identity": [out["arms"][s][k]["sym_dispatch_identity"] for s in out["arms"]],
                "text_ce_routed_minus_C": [round(out["arms"][s][k]["routed_text_ce"] - out["arms"][s][k]["C_alone_text_ce"], 4) for s in out["arms"]],
                "modk": [out["arms"][s][k].get("reasoning", {}).get("modk") for s in out["arms"]],
                "pair": [out["arms"][s][k].get("reasoning", {}).get("pair") for s in out["arms"]],
                "dyck": [out["arms"][s][k].get("reasoning", {}).get("dyck_close") for s in out["arms"]]}
            for k in out["arms"]["111"]}
    out["summary"] = summ; out["wall_s"] = round(time.time() - t0)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P30] SUMMARY", json.dumps(summ), flush=True); print("[P30] DONE", flush=True)


if __name__ == "__main__":
    main()
