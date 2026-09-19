"""P27B: corrected free-running generation probe (family id fixed: all
of track/modk/pair scored; dyck skipped) over cached P27 checkpoints.
n=48 prompts. Unified (10 seeds) vs TF-NAPE/GRU monolith controls (2)."""
import json, os, torch
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
import arch_vet_p27 as p27, arch_vet_p26 as p26, arch_vet_p26b as p26b, arch_vet_p22 as p22, arch_vet_p21 as p21, arch_vet_p23 as p23, arch_vet_p25 as p25
CKPT = p27.CKPT; V0, VB = p26.V0, p26.VB
res = {"unified": {}, "ctrl": {}}
for seed in map(int, p25.DEFAULT_SEEDS.split(",")):
    mA = p21.VETDCC(V0, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V0, 24, k=8, K=8); mC = p26.ByteGRU(VB, 48); g2 = p22.Gate(V0, 8); gm = p26b.GateM(4)
    for m, f in ((mA, "A"), (mB, "B"), (mC, "C"), (g2, "G"), (gm, "GM")): m.load_state_dict(torch.load(f"{CKPT}/{f}_s{seed}.pt")["sd"]); m.eval()
    u = p26.Unified3(mA, mB, mC, p26b.HierGate(g2, gm)).eval()
    r = p27.generation_probe(u, n=48); res["unified"][str(seed)] = r; print(f"[p27b unified s{seed}] {r['exact']} {r['per_family']} n={r['n']}", flush=True)
    if seed == 111: print("[p27b SAMPLE]", repr(r["sample"][:220]), flush=True)
for name in ("tf", "gru"):
    for seed in (111, 222):
        m = p23.TFCtrl(VB, d=56, nh=4, depth=2, mlp=2, pe="nape") if name == "tf" else p26.ByteGRU(VB, 64, 2)
        m.load_state_dict(torch.load(f"{CKPT}/CTRL_{name}_s{seed}.pt")["sd"]); m.eval()
        r = p27.generation_probe(m, n=48); res["ctrl"][f"{name}_s{seed}"] = r; print(f"[p27b ctrl {name} s{seed}] {r['exact']} {r['per_family']}", flush=True)
ex = [v["exact"] for v in res["unified"].values()]
out = {"tag": "ARCH-VET-LM-P27B", "protocol": __doc__, "unified_exact": ex, "unified_mean": round(sum(ex) / len(ex), 4),
       "unified_per_family": {k: [v["per_family"][k] for v in res["unified"].values()] for k in ("track", "modk", "pair")},
       "ctrl": {k: {"exact": v["exact"], "per_family": v["per_family"]} for k, v in res["ctrl"].items()}, "sample_s111": res["unified"]["111"]["sample"]}
open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P27B] DONE", json.dumps({k: out[k] for k in ("unified_exact", "unified_per_family", "ctrl")}))
