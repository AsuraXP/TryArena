"""ARCH-VET P41 (cycle 82) — INFERENCE ENVELOPE: measured, not asserted.
Forward-only (no grad, batch 1, 1 thread) wall time per token and peak RSS for the unified 5-expert system
(A+B+C+K+gates, 116,858 p, all experts evaluated densely = worst case) vs the matched TF-ALiBi control
(42,672 p, 2L d48) at L = 256, 1024, 4096, 16384 on the 2-core / 4 GB box.
Claim under test: the system's per-token cost and state are O(1) in L (registers/counters/stack/bank), the TF's
per-token cost is O(L) and its attention memory O(L^2) — at 16k the TF's score matrices alone are
16384^2 x 4 heads x 4 B = 4.3 GB per layer, i.e. beyond this machine.
Prior art: KV-cache / linear-attention cost analyses (Katharopoulos 2020; Gu-Dao 2023 Mamba); here the point is
an exact-memory architecture with the same O(1) envelope AND the certified exact bars."""
import argparse, json, os, resource, time, gc, torch
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p36 as p36, arch_vet_p26 as p26, arch_vet_p26b as p26b, arch_vet_p22 as p22, arch_vet_p21 as p21, arch_vet_p38 as p38, arch_vet_p23 as p23, arch_vet_p19 as p19, arch_vet_p32 as p32
CKPT = "p21_ckpt"; V0 = p36.V0; VB = p26.VB

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def build_unified(seed=111):
    mA = p21.VETDCC(V0, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V0, 24, k=8, K=8); mC = p26.ByteGRU(VB, 48)
    mA.load_state_dict(torch.load(f"{CKPT}/A_s{seed}.pt")["sd"]); mB.load_state_dict(torch.load(f"{CKPT}/B_s{seed}.pt")["sd"]); mC.load_state_dict(torch.load(f"{CKPT}/CF_s{seed}.pt")["sd"])
    g2 = p22.Gate(V0, 8); g2.load_state_dict(torch.load(f"{CKPT}/G_s{seed}.pt")["sd"]); gm = p26b.GateM(4); gm.load_state_dict(torch.load(f"{CKPT}/GM_s{seed}.pt")["sd"])
    mK = p38.KRBBlocked(V0, 24, 8, b=4, smart=True); mK.load_state_dict(torch.load(f"{CKPT}/P38_R2_SBK4_s{seed}.pt")["sd"])
    u3 = p26.Unified3(mA, mB, mC, p26b.HierGate(g2, gm)); gk = p26b.GateM(4); gk.load_state_dict(torch.load(f"{CKPT}/GKSBK4F_s{seed}.pt")["sd"])
    u = p36.Unified4(u3, mK, gk); u.eval(); return u

@torch.no_grad()
def probe(model, x, reps):
    model(x[:, :64])                                   # warm-up
    gc.collect(); r0 = rss_mb(); t0 = time.time()
    for _ in range(reps): model(x)
    dt = (time.time() - t0) / reps; return dt, rss_mb(), r0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lens", default="256,1024,4096,16384"); ap.add_argument("--tf_max", type=int, default=16384); a = ap.parse_args()
    out = {"tag": "ARCH-VET-LM-P41", "protocol": __doc__, "rows": []}
    u = build_unified(); tf = p23.TFCtrl(V0, pe="alibi"); tf.eval()
    print(f"[p41] unified params {sum(p.numel() for p in u.parameters())} | TF-ALiBi params {p19.n_params(tf)}", flush=True)
    for L in [int(s) for s in a.lens.split(",")]:
        x = torch.tensor(p19.make_pool(1, L, 5)[0][:L]).unsqueeze(0)
        reps = 1 if L >= 4096 else 3
        dt, rss, r0 = probe(u, x, reps); row = {"L": L, "unified_s": round(dt, 3), "unified_us_per_tok": round(1e6 * dt / L, 1), "unified_peak_rss_mb": round(rss, 1)}
        print(f"[p41 L={L}] unified {dt:.3f}s = {1e6*dt/L:.1f} us/tok, peak RSS {rss:.0f} MB", flush=True)
        if L <= a.tf_max:
            try:
                dt, rss, r0 = probe(tf, x, reps); row.update({"tf_s": round(dt, 3), "tf_us_per_tok": round(1e6 * dt / L, 1), "tf_peak_rss_mb": round(rss, 1)})
                print(f"[p41 L={L}] TF-ALiBi {dt:.3f}s = {1e6*dt/L:.1f} us/tok, peak RSS {rss:.0f} MB", flush=True)
            except (RuntimeError, MemoryError) as e:
                row.update({"tf_s": None, "tf_error": str(e)[:120]}); print(f"[p41 L={L}] TF-ALiBi FAILED: {str(e)[:120]}", flush=True)
        out["rows"].append(row); gc.collect()
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P41] DONE", flush=True)

if __name__ == "__main__":
    main()
