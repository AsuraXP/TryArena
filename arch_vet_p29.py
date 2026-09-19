"""ARCH-VET P29 (cycle 65) — END-TO-END CO-ADAPTATION under the certified
gate: unfreeze experts A and B (gate G FROZEN, hard-routed gradient) and
fine-tune on the joint pool at small lr. Question: does co-adaptation
(a) improve the deterministic-position CE / bars, or (b) erode the
certified specialisation (catastrophic drift), as DES-MoE (arXiv
2509.16882, EMNLP 2025) reports for full fine-tuning of routed experts
without isolation? Two arms x 2 seeds x 600 steps:
  COADAPT-HARD : loss through the HARD gate choice only (each position's
                 gradient flows to exactly the selected expert = DES-MoE
                 style gradient isolation by routing);
  COADAPT-SOFT : loss through the frozen gate's soft mixture (both
                 experts get gradient everywhere).
lr 3e-4 (10x below training). Eval before/after: unified 4-bar row,
det/stoch CE (P28 classifier) on 256-hard, joint mixdd pass.
Prediction: HARD keeps bars and lowers det CE slightly (only expert-
consistent gradients); SOFT drifts B toward vanilla statistics and
breaks dyck d12 and/or the length ratio. Tag ARCH-VET-LM-P29.
"""
import copy, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); CKPT = os.path.join(REPO, "p21_ckpt")
import arch_vet_p19 as p19, arch_vet_p13d as p13d, arch_vet_p21 as p21, arch_vet_p22 as p22, arch_vet_p28 as p28
V = p19.V


@torch.no_grad()
def full_row(mA, mB, g):
    mA.eval(); mB.eval(); p21.causal_route = (lambda toks: p22.gate_routes(g, toks))
    acc = p21.routed_task_acc(mA, mB, 24, 256, random.Random(666), hard=True)
    vha = p19.make_batches(8, 256, random.Random(777), hard=True); v1024 = p19.make_batches(2, 1024, random.Random(31415))
    ceh = p21.routed_ce(mA, mB, vha, 256); ce1 = p21.routed_ce(mA, mB, v1024, 1024)
    d12 = p21.routed_close_acc(mA, mB, 2, 12, p13d.seg_len(12) + 16)[0]; d6 = p21.routed_close_acc(mA, mB, 8, 6, 256)[0]
    d = p28.nll_by_class(lambda x: p21.routed_logits(mA, mB, x), vha, 256); ds = p28.det_stoch(d)
    jm = p21.joint_mixed_eval(mA, mB, 24, 256, corpus="mixdd")
    ratio = round(ce1 / ceh, 3)
    return {"pair": acc["pair"], "modk": acc["modk"], "track": acc["track"], "ce_256hard": ceh, "ce_1024": ce1, "ratio": ratio,
            "dyck_d6": d6, "dyck_d12": d12, "det_nll_hard": ds["det_nll"], "stoch_nll_hard": ds["stoch_nll"],
            "joint_pair": jm["pair"], "joint_dyck": jm["dyck_close"],
            "bars": int(acc["pair"] >= .717) + int(acc["modk"] >= 1) + int(ratio <= .6) + int(d12 >= .85)}


def coadapt(mA, mB, g, pool, mode, steps=600, lr=3e-4, batch=8):
    mA.train(); mB.train()
    opt = torch.optim.AdamW(list(mA.parameters()) + list(mB.parameters()), lr=lr); t0 = time.time()
    for step in range(1, steps + 1):
        x = torch.stack([pool[(step * batch + i) % len(pool)] for i in range(batch)]); y = x[:, 1:256]
        la = F.log_softmax(mA(x[:, :256])[:, :255], -1); lb = F.log_softmax(mB(x[:, :256])[:, :255], -1)
        with torch.no_grad(): gl = g(x[:, :255])
        if mode == "hard":
            sel = (gl.argmax(-1) == 1).unsqueeze(-1)
            lp = torch.where(sel, lb, la)
        else:
            w = F.softmax(gl, -1).unsqueeze(-1); lp = torch.log(w[:, :, 0] * la.exp() + w[:, :, 1] * lb.exp() + 1e-9)
        loss = F.nll_loss(lp.reshape(-1, V), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(mA.parameters()) + list(mB.parameters()), 1.0); opt.step()
        if step % 200 == 0: print(f"  [coadapt-{mode}] step {step}/{steps} loss {loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)
    mA.eval(); mB.eval()


def main():
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P29", "protocol": __doc__[:1100], "arms": {}}
    for seed in (111, 222):
        torch.manual_seed(seed)
        A0 = p21.VETDCC(V, 24, k=8, K=8); B0 = p21.STACKDCC2_D12(V, 24, k=8, K=8); g = p22.Gate(V, 8)
        A0.load_state_dict(torch.load(f"{CKPT}/A_s{seed}.pt")["sd"]); B0.load_state_dict(torch.load(f"{CKPT}/B_s{seed}.pt")["sd"]); g.load_state_dict(torch.load(f"{CKPT}/G_s{seed}.pt")["sd"])
        g.eval()
        for q in g.parameters(): q.requires_grad_(False)
        base = full_row(A0, B0, g); print(f"[p29 s{seed} BEFORE] {base}", flush=True)
        out["arms"].setdefault("frozen", {})[str(seed)] = base
        pool = p22.joint_pool(12345 + seed)
        for mode in ("hard", "soft"):
            mA, mB = copy.deepcopy(A0), copy.deepcopy(B0)
            coadapt(mA, mB, g, pool, mode)
            r = full_row(mA, mB, g); print(f"[p29 s{seed} COADAPT-{mode.upper()}] {r}", flush=True)
            out["arms"].setdefault(f"coadapt-{mode}", {})[str(seed)] = r
    summ = {arm: {k: [per[s][k] for s in sorted(per)] for k in ("bars", "pair", "modk", "ratio", "dyck_d12", "det_nll_hard", "stoch_nll_hard", "joint_dyck")} for arm, per in out["arms"].items()}
    out["summary"] = summ; out["wall_s"] = round(time.time() - t0)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P29] SUMMARY", json.dumps(summ), flush=True); print("[P29] DONE", flush=True)


if __name__ == "__main__":
    main()
