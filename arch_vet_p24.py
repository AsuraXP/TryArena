"""ARCH-VET P24 (cycle 62) — MODULARITY vs SUBSTRATE: a Transformer-of-
experts under the IDENTICAL modular recipe (per-regime experts + P22's
learned causal gate).

WHY: P21-P23 established (a) modular fusion breaks the data-level
fusion wall, (b) a learned 1.4k gate suffices, (c) monolithic
length-generalizing Transformers (NoPE/ALiBi/NAPE) at matched params /
data / steps pass <=1 of 4 bars. The remaining confound: is the unified
row a property of MODULARITY (which a Transformer could also enjoy) or of
the VET SUBSTRATE (Mealy controller x exact counters x exact stack)?
P24 gives the Transformer the SAME recipe: expert A = TF-NAPE (best
in-range TF from P23, d32) trained ONLY on the vanilla P9 pool @4000
steps; expert B = TF-ALiBi (the P23 arm that reached dyck d12 .94) d32
trained ONLY on the deep-mix dyck pool @2000; then P22's Gate (GRU 8,
1,410p) is trained on the same joint pool with the same composed-
mixture CE, hard argmax dispatch at eval. Total params matched to the
VET system (~43k).
PRIOR ART (searched 2026-09-10): arXiv 2410.13964 (SMoE for
compositional generalization: optimal sparsity scales with task
complexity; experts homogeneous FFNs inside one TF); arXiv 2606.14398
(theory: top-1 task routing to task-specific experts in MoE
transformers; expert size scales with task complexity); arXiv 2511.06237
(MoSEs: sub-expert routing for continual learning). None trains
whole-model per-regime Transformer experts with a separately learned
causal token gate and compares to a non-attention substrate under the
same recipe — that is the ablation here.
PREDICTION (falsifiable): if modularity alone explains the row, TF-MoE
should reach >=3/4 bars; if the substrate matters, modk (TF .10-.27 in
P23) and pair should stay below bar even with dedicated experts.
"""
import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
CKPT = os.path.join(REPO, "p21_ckpt")
import arch_vet_p19 as p19
import arch_vet_p13d as p13d
import arch_vet_p21 as p21
import arch_vet_p21c as p21c
import arch_vet_p22 as p22
import arch_vet_p23 as p23
V = p19.V


def tf_expert(pe, d=32, nh=4):
    return p23.TFCtrl(V, d=d, nh=nh, depth=2, mlp=2, pe=pe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="111,222")
    ap.add_argument("--gate_steps", type=int, default=1500)
    ap.add_argument("--d", type=int, default=32)
    a = ap.parse_args()
    t0 = time.time()
    out = {"tag": "ARCH-VET-LM-P24", "protocol": __doc__[:1200], "arms": {"TF-MoE": {"per_seed": {}}}}
    structural = p21c.make_deep_router(3)
    poolA = p19.make_pool(512, 256, 12345)
    poolB = p13d.gen_mix_pool(256, 256, 12345)
    rows = []
    for seed in map(int, a.seeds.split(",")):
        torch.manual_seed(seed); mA = tf_expert("nape", a.d)
        torch.manual_seed(seed); mB = tf_expert("alibi", a.d)
        nA, nB = p19.n_params(mA), p19.n_params(mB)
        print(f"[p24] s{seed} TF-A(nape) {nA}p  TF-B(alibi) {nB}p  total {nA+nB+1410}", flush=True)
        pa = os.path.join(CKPT, f"TFA_s{seed}.pt"); pb = os.path.join(CKPT, f"TFB_s{seed}.pt")
        if os.path.exists(pa):
            mA.load_state_dict(torch.load(pa)["sd"]); hA = torch.load(pa)["hist"]
        else:
            hA = p19.train_arm(f"P24-TFA-s{seed}", mA, poolA, 4000, 8)
            torch.save({"sd": mA.state_dict(), "hist": hA}, pa)
        if os.path.exists(pb):
            mB.load_state_dict(torch.load(pb)["sd"]); hB = torch.load(pb)["hist"]
        else:
            hB = p19.train_arm(f"P24-TFB-s{seed}", mB, poolB, 2000, 8)
            torch.save({"sd": mB.state_dict(), "hist": hB}, pb)
        mA.eval(); mB.eval()
        for m in (mA, mB):
            for p_ in m.parameters(): p_.requires_grad_(False)
        # expert-alone attribution (cheap): A on hard tasks, B on dyck d6/d12
        with torch.no_grad():
            accA = p19.task_acc(mA, 24, 256, random.Random(666), hard=True)
            dB = {d: p13d.rt_close_acc(mB, 4 if d > 6 else 8, d,
                                       256 if p13d.seg_len(d) <= 252 else p13d.seg_len(d) + 16)[0]
                  for d in (3, 6, 12)}
        print(f"[p24 s{seed}] A-alone hard acc {accA} | B-alone close {dB}", flush=True)
        pool = p22.joint_pool(12345 + seed)
        g = p22.train_gate(mA, mB, pool, seed, a.gate_steps)
        torch.save({"sd": g.state_dict()}, os.path.join(CKPT, f"TFG_s{seed}.pt"))
        row = p22.evaluate(mA, mB, g, structural, "TF-MoE", seed)
        p21.causal_route = p21c.make_deep_router(3)
        row.update({"params": {"A": nA, "B": nB, "gate": 1410},
                    "A_alone_hard": accA, "B_alone_close": dB,
                    "hist_A": hA, "hist_B": hB})
        ae = row["task_acc_hard"]
        row["bars"] = int(ae["pair"] >= .717) + int(ae["modk"] >= 1.0) + \
            int(row["len_ratio_1024_over_256hard"] <= .6) + \
            int(row["dyck_close_routed"]["close_d12"] >= .85)
        print(f"[p24 TF-MoE s{seed}] BARS={row['bars']}", flush=True)
        out["arms"]["TF-MoE"]["per_seed"][str(seed)] = row; rows.append(row)
    out["arms"]["TF-MoE"]["summary"] = {
        "bars_passed_per_seed": [r["bars"] for r in rows],
        "pair": [r["task_acc_hard"]["pair"] for r in rows],
        "modk": [r["task_acc_hard"]["modk"] for r in rows],
        "ratio": [r["len_ratio_1024_over_256hard"] for r in rows],
        "dyck_d12": [r["dyck_close_routed"]["close_d12"] for r in rows],
        "joint_pair": [r["joint_mixdd"]["pair"] for r in rows],
        "joint_dyck": [r["joint_mixdd"]["dyck_close"] for r in rows],
        "gate_b_share": [r["gate_b_share"] for r in rows]}
    out["wall_s"] = round(time.time() - t0)
    print("[P24] SUMMARY", json.dumps(out["arms"]["TF-MoE"]["summary"]), flush=True)
    with open("log.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
