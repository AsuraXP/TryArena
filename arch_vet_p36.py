"""ARCH-VET P36 (cycle 72) — FOLD the learned-predicate KRB (expert K = P35
KRB-SEEN, frozen) INTO the certified exact+fluent system via a second
hierarchical gate: one parameter set that holds the four certified OOD
bars, the text CE, AND the MQAR (multi-query associative recall) bar.

Design (L-HIERARCHICAL-GATE-PRESERVES-CERTIFICATION, C63): a modality-
style gate GK (one-hot(304) -> GRU(4) -> 1 logit, ~3.7k p) sits ABOVE
the certified HierGate (G2 + GM); dispatch = K if GK else the certified
3-way choice. GK is trained 1000 steps by the composed-mixture CE with
A, B, C, K, G2, GM all frozen, on a joint pool = the P26B pool + P32-R2
streams. Prior art (searched 2026-09-16): BAR "Train Separately, Merge
Together" arXiv 2604.18473 (independent domain experts + lightweight
router, linear-cost expert addition, no degradation) — same principle
at 7B, dense experts, no exact-memory expert and no certification test.
Bars (per seed): pair>=.717 modk=1 ratio<=.6 dyck_d12>=.85 (certified
row, must stay bit-identical in dispatch: identity 1.0) + NEW MQAR bar:
R2 n4 >= .90, n8 >= .70 (P35 SEEN 3-seed .92/.73) + text CE == C-alone.
Seeds 111/222/333 (K = P35_R2_SEEN_s{seed}). Tag ARCH-VET-LM-P36.
PREDICTION: 6/6 bars on 3/3 seeds; symbolic identity on non-KRB
streams >= .999; K-share on KRB streams >= .95; KRB acc within .02 of
K-alone. If GK leaks onto vanilla streams (pair erodes) -> report.
"""
import argparse, json, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO); CKPT = f"{REPO}/p21_ckpt"
import arch_vet_p26 as p26, arch_vet_p26b as p26b, arch_vet_p22 as p22, arch_vet_p21 as p21, arch_vet_p19 as p19
import arch_vet_p13d as p13d, arch_vet_lm as lm, arch_vet_p32 as p32, arch_vet_p35 as p35, arch_vet_p37 as p37
V0, VB = p26.V0, p26.VB


class Unified4(nn.Module):
    def __init__(self, u3, mK, gk):
        super().__init__(); self.u3, self.mK, self.gk = u3, mK, gk

    def expert_logits(self, x):
        la, lb, lc = self.u3.expert_logits(x); return la, lb, lc, p26.pad48(self.mK(self.u3.sym(x)))

    def logits4(self, x):
        h3 = F.log_softmax(self.u3.g(x), -1); k = self.gk(x); return torch.cat([F.logsigmoid(-k).unsqueeze(-1) + h3, F.logsigmoid(k).unsqueeze(-1)], -1)

    def routes(self, x): return self.logits4(x).argmax(-1)

    def forward(self, x):
        la, lb, lc, lk = self.expert_logits(x); r = self.routes(x).unsqueeze(-1)
        return torch.where(r == 0, la, torch.where(r == 1, lb, torch.where(r == 2, lc, lk)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="111,222,333"); ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--K", default="P35", help="P35 = SEEN expert | P37 = CF85R expert (cycle 75) | BK2/BK4 = blocked-cuckoo P38 expert (cycle 77)"); a = ap.parse_args()
    t0 = time.time(); out = {"tag": "ARCH-VET-LM-P36", "K": a.K, "protocol": __doc__[:1900], "per_seed": {}}
    va_text = torch.stack(p26.pool_of(p26.gen_text_stream, 32, 99, src=p26.TXT_VA)); R2 = p32.regime("R2")
    for seed in map(int, a.seeds.split(",")):
        torch.manual_seed(seed)
        mA = p21.VETDCC(V0, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V0, 24, k=8, K=8); mC = p26.ByteGRU(VB, 48)
        mA.load_state_dict(torch.load(f"{CKPT}/A_s{seed}.pt")["sd"]); mB.load_state_dict(torch.load(f"{CKPT}/B_s{seed}.pt")["sd"]); mC.load_state_dict(torch.load(f"{CKPT}/C_s{seed}.pt")["sd"])
        g2 = p22.Gate(V0, 8); g2.load_state_dict(torch.load(f"{CKPT}/G_s{seed}.pt")["sd"]); gm = p26b.GateM(4); gm.load_state_dict(torch.load(f"{CKPT}/GM_s{seed}.pt")["sd"])
        if a.K.startswith("BK"): mK = p38.KRBBlocked(V0, 24, 8, b=int(a.K[2:])); mK.load_state_dict(torch.load(f"{CKPT}/P38_R2_{a.K}_s{seed}.pt")["sd"])
        elif a.K == "P37": mK = p37.KRBHind(V0, 24, 8, seen=True, cf=True, hseed=85, decouple=True); mK.load_state_dict(torch.load(f"{CKPT}/P37_R2_CF85R_s{seed}.pt")["sd"])
        else: mK = p35.KRBHind(V0, 24, 8, seen=True); mK.load_state_dict(torch.load(f"{CKPT}/P35_R2_SEEN_s{seed}.pt")["sd"])
        for m in (mA, mB, mC, g2, gm, mK):
            m.eval()
            for q in m.parameters(): q.requires_grad_(False)
        u3 = p26.Unified3(mA, mB, mC, p26b.HierGate(g2, gm)); torch.manual_seed(seed); gk = p26b.GateM(4); u = Unified4(u3, mK, gk)
        joint = [torch.tensor(lm.gen_stream(random.Random(1000 + seed + i), 256)[0]) for i in range(96)]
        joint += p19.gen_mixdd_pool(96, 256, 2000 + seed) + p13d.gen_mix_pool(96, 256, 3000 + seed)
        joint += p26.pool_of(p26.gen_chatmix_stream, 128, 4000 + seed) + p26.pool_of(p26.gen_text_stream, 64, 5000 + seed)
        rk = random.Random(6000 + seed); joint += [torch.tensor(p32.gen_stream(rk, R2, 257)) for _ in range(128)]
        joint = [j[:257] for j in joint]
        gkp = f"{CKPT}/GK{'37' if a.K == 'P37' else (a.K if a.K.startswith('BK') else '')}_s{seed}.pt"
        if a.steps == 0 and os.path.exists(gkp): gk.load_state_dict(torch.load(gkp)["sd"]); print(f"[p36] s{seed} GK loaded from checkpoint (re-score mode)", flush=True)
        opt = torch.optim.AdamW(gk.parameters(), lr=3e-3); tt = time.time()
        for step in range(1, a.steps + 1):
            x = torch.stack([joint[(step * 8 + i) % len(joint)] for i in range(8)]); y = x[:, 1:256]
            with torch.no_grad():
                L4 = u.expert_logits(x[:, :256]); P = torch.stack([F.softmax(l, -1) for l in L4], 2)[:, :255]
            w = F.softmax(u.logits4(x[:, :255]), -1).unsqueeze(-1); p = (w * P).sum(2)
            loss = F.nll_loss(torch.log(p + 1e-9).reshape(-1, VB), y.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gk.parameters(), 1.0); opt.step()
            if step % 250 == 0: print(f"  [P36-GK-s{seed}] step {step}/{a.steps} loss {loss.item():.4f} ({time.time()-tt:.0f}s)", flush=True)
        if a.steps > 0: torch.save({"sd": gk.state_dict()}, gkp)
        gk.eval(); u.eval()
        with torch.no_grad():
            row = {"C_alone_text_ce": p26.decomposed_ce(mC, va_text)["text_ce"], "routed_text_ce": p26.decomposed_ce(u, va_text)["text_ce"]}
            same = tot = 0
            for s in p19.make_pool(16, 256, 7) + p13d.gen_mix_pool(16, 256, 7):
                r4 = u.routes(s[:256].unsqueeze(0)).squeeze(0); r3 = u3.routes(s[:256].unsqueeze(0)).squeeze(0)
                same += int((r4 == r3).sum()); tot += r4.numel()
            row["symbolic_dispatch_identity"] = round(same / tot, 4)
            kx = [torch.tensor(p32.gen_stream(random.Random(900 + i), R2, 256, True, None)[:256]) for i in range(16)]
            row["K_share_on_krb"] = round(float(torch.stack([(u.routes(s.unsqueeze(0)) == 3).float().mean() for s in kx]).mean()), 4)
            p21.causal_route = (lambda toks: [min(r, 1) for r in u.routes(torch.tensor(toks).unsqueeze(0)).squeeze(0).tolist()])
            acc = p21.routed_task_acc(mA, mB, 24, 256, random.Random(666), hard=True)
            vha = p19.make_batches(8, 256, random.Random(777), hard=True); v1024 = p19.make_batches(2, 1024, random.Random(31415))
            ratio = round(p21.routed_ce(mA, mB, v1024, 1024) / p21.routed_ce(mA, mB, vha, 256), 3)
            d12 = p21.routed_close_acc(mA, mB, 2, 12, p13d.seg_len(12) + 16)[0]
            uk = lambda x: u(x)[..., :V0]
            class W(nn.Module):
                def forward(self, x): return uk(x)
            w = W(); w.eval()
            mq = {f"n{n}": p32.acc(w, R2, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in (4, 8)}
            mqk = {f"n{n}": p32.acc(mK, R2, 10, 256 if n <= 4 else 320, random.Random(600 + n), True, n) for n in (4, 8)}
        row["guard"] = {"pair": acc["pair"], "modk": acc["modk"], "ratio": ratio, "dyck_d12": d12, "mqar_routed": mq, "mqar_K_alone": mqk}
        row["bars"] = {"pair": acc["pair"] >= .717, "modk": acc["modk"] >= 1, "ratio": ratio <= .6, "dyck_d12": d12 >= .85, "mqar_n4": mq["n4"] >= .90, "mqar_n8": mq["n8"] >= .70}
        row["n_bars"] = sum(row["bars"].values()); row["params"] = p19.n_params(mA) + p19.n_params(mB) + p19.n_params(mC) + p19.n_params(g2) + p19.n_params(gm) + p19.n_params(mK) + p19.n_params(gk)
        out["per_seed"][seed] = row; print(f"[p36 K={a.K} s{seed}] bars={row['n_bars']}/6 {row['guard']} identity={row['symbolic_dispatch_identity']} Kshare={row['K_share_on_krb']} text {row['routed_text_ce']}/{row['C_alone_text_ce']} params {row['params']}", flush=True)
    out["wall_s"] = round(time.time() - t0); open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P36] DONE", flush=True)


if __name__ == "__main__":
    main()
