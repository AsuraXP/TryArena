"""ARCH-VET P26B (cycle 63) — HIERARCHICAL GATE: modality gate M (C vs
{A,B}) trained on top of the FROZEN certified P22 gate G (A vs B).
Fix for L-FLAT-GATE-BLURS-DEPTH (P26: flat 3-way gate regressed the
length-invariance bar to ratio 1.0). By construction, whenever M says
"symbolic", dispatch is exactly the certified 2-way policy, so the
10-seed unified row is preserved unless M mis-fires on symbolic
streams. M = one-hot(304) -> GRU(4) -> 1 logit (~3.7k p). Trained by the
composed-mixture CE on the same joint pool (P26), with G/A/B/C frozen.
Eval = P26 metrics + unified-row guard. Prior art: hierarchical /
two-level routing (arXiv 2604.23108 MoHGE group->expert; arXiv
2507.11181 hierarchical MoE). Tag ARCH-VET-LM-P26B.
"""
import json, os, random, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
CKPT = os.path.join(REPO, "p21_ckpt")
import arch_vet_p26 as p26
import arch_vet_p22 as p22
import arch_vet_p21 as p21
import arch_vet_p19 as p19
import arch_vet_p13d as p13d
import arch_vet_lm as lm
V0, VB = p26.V0, p26.VB


class GateM(nn.Module):
    def __init__(self, h=4):
        super().__init__(); self.gru = nn.GRU(VB, h, batch_first=True); self.head = nn.Linear(h, 1)

    def forward(self, x):
        h, _ = self.gru(F.one_hot(x, VB).float()); return self.head(h).squeeze(-1)   # logit of "C"


class HierGate(nn.Module):
    """Returns 3 logits compatible with Unified3: [A,B,C]."""
    def __init__(self, g2, gm):
        super().__init__(); self.g2, self.gm = g2, gm

    def forward(self, x):
        xs = torch.where(x >= V0, torch.full_like(x, p19.MODS), x)
        ab = F.log_softmax(self.g2(xs), -1)                 # (B,L,2)
        lc = F.logsigmoid(self.gm(x)); lnc = F.logsigmoid(-self.gm(x))
        return torch.stack([lnc + ab[..., 0], lnc + ab[..., 1], lc], -1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="111,222"); ap.add_argument("--steps", type=int, default=1000)
    a = ap.parse_args(); t0 = time.time()
    va_text = torch.stack(p26.pool_of(p26.gen_text_stream, 32, 99, src=p26.TXT_VA))
    va_chat = torch.stack(p26.pool_of(p26.gen_chatmix_stream, 32, 98, src=p26.TXT_VA))
    out = {"tag": "ARCH-VET-LM-P26B", "protocol": __doc__[:900], "arms": {"hier": {"per_seed": {}}}}
    for seed in map(int, a.seeds.split(",")):
        torch.manual_seed(seed)
        mA = p21.VETDCC(V0, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V0, 24, k=8, K=8); mC = p26.ByteGRU(VB, 48)
        mA.load_state_dict(torch.load(f"{CKPT}/A_s{seed}.pt")["sd"]); mB.load_state_dict(torch.load(f"{CKPT}/B_s{seed}.pt")["sd"])
        mC.load_state_dict(torch.load(f"{CKPT}/C_s{seed}.pt")["sd"])
        g2 = p22.Gate(V0, 8); g2.load_state_dict(torch.load(f"{CKPT}/G_s{seed}.pt")["sd"])
        for m in (mA, mB, mC, g2):
            m.eval()
            for q in m.parameters(): q.requires_grad_(False)
        torch.manual_seed(seed); gm = GateM(4); hg = HierGate(g2, gm)
        joint = [torch.tensor(lm.gen_stream(random.Random(1000 + seed + i), 256)[0]) for i in range(96)]
        joint += p19.gen_mixdd_pool(96, 256, 2000 + seed) + p13d.gen_mix_pool(96, 256, 3000 + seed)
        joint += p26.pool_of(p26.gen_chatmix_stream, 128, 4000 + seed) + p26.pool_of(p26.gen_text_stream, 64, 5000 + seed)
        joint = [j[:257] for j in joint]
        u = p26.Unified3(mA, mB, mC, hg)
        opt = torch.optim.AdamW(gm.parameters(), lr=3e-3); tt = time.time()
        for step in range(1, a.steps + 1):
            x = torch.stack([joint[(step * 8 + i) % len(joint)] for i in range(8)]); y = x[:, 1:256]
            with torch.no_grad():
                la, lb, lc = u.expert_logits(x[:, :256])
                P = torch.stack([F.softmax(la, -1), F.softmax(lb, -1), F.softmax(lc, -1)], 2)[:, :255]
            w = F.softmax(hg(x[:, :255]), -1).unsqueeze(-1); p = (w * P).sum(2)
            loss = F.nll_loss(torch.log(p + 1e-9).reshape(-1, VB), y.reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gm.parameters(), 1.0); opt.step()
            if step % 250 == 0: print(f"  [gateM s{seed}] step {step}/{a.steps} loss {loss.item():.4f} ({time.time()-tt:.0f}s)", flush=True)
        gm.eval(); torch.save({"sd": gm.state_dict()}, f"{CKPT}/GM_s{seed}.pt"); u.eval()
        with torch.no_grad():
            row = {"gateM_params": p19.n_params(gm),
                   "C_alone_text_ce": p26.decomposed_ce(mC, va_text)["text_ce"],
                   "routed_text_ce_textstreams": p26.decomposed_ce(u, va_text)["text_ce"],
                   "routed_chatmix": p26.decomposed_ce(u, va_chat),
                   "chatmix_reasoning": p26.chatmix_reasoning(u, 32),
                   "gate_share": {"text": p26.gate_share(u, p26.pool_of(p26.gen_text_stream, 16, 7, src=p26.TXT_VA)),
                                  "chatmix": p26.gate_share(u, p26.pool_of(p26.gen_chatmix_stream, 16, 7, src=p26.TXT_VA)),
                                  "vanilla": p26.gate_share(u, p19.make_pool(16, 256, 7)),
                                  "deepdyck": p26.gate_share(u, p13d.gen_mix_pool(16, 256, 7))}}
            p21.causal_route = (lambda toks: [min(r, 1) for r in u.routes(torch.tensor(toks).unsqueeze(0)).squeeze(0).tolist()])
            acc = p21.routed_task_acc(mA, mB, 24, 256, random.Random(666), hard=True)
            vha = p19.make_batches(8, 256, random.Random(777), hard=True); v1024 = p19.make_batches(2, 1024, random.Random(31415))
            ratio = round(p21.routed_ce(mA, mB, v1024, 1024) / p21.routed_ce(mA, mB, vha, 256), 3)
            d12 = p21.routed_close_acc(mA, mB, 2, 12, p13d.seg_len(12) + 16)[0]
            # is dispatch on symbolic streams BIT-IDENTICAL to the certified 2-way gate?
            same = tot = 0
            for s in p19.make_pool(16, 256, 7) + p13d.gen_mix_pool(16, 256, 7):
                r3 = u.routes(s[:256].unsqueeze(0)).squeeze(0); r2 = g2(s[:256].unsqueeze(0)).argmax(-1).squeeze(0)
                same += int((r3 == r2).sum()); tot += r3.numel()
            row["symbolic_dispatch_identity"] = round(same / tot, 4)
        row["unified_row_guard"] = {"pair": acc["pair"], "modk": acc["modk"], "ratio": ratio, "dyck_d12": d12,
                                    "bars": int(acc["pair"] >= .717) + int(acc["modk"] >= 1) + int(ratio <= .6) + int(d12 >= .85)}
        print(f"[p26b hier s{seed}] text routed {row['routed_text_ce_textstreams']} (C {row['C_alone_text_ce']}) | chatmix {row['routed_chatmix']} | "
              f"reasoning {row['chatmix_reasoning']} | share {row['gate_share']} | identity {row['symbolic_dispatch_identity']} | guard {row['unified_row_guard']}", flush=True)
        out["arms"]["hier"]["per_seed"][str(seed)] = row
    out["wall_s"] = round(time.time() - t0)
    with open("log.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    print("[P26B] DONE", flush=True)


if __name__ == "__main__":
    main()
