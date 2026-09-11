"""ARCH-VET P27 (cycle 64) — 10-SEED CERTIFICATION OF THE 3-EXPERT
(exact + fluent) SYSTEM + OOD bars for the fluency monolith controls
+ a free-running GENERATION probe.

Per seed (111..1010; A/B/G from P25): train C (byte GRU, 4000 steps,
text-only) and M (modality gate, 1000 steps) if missing; evaluate:
 (a) unified-row guard through the hierarchical gate (4 bars),
 (b) symbolic dispatch identity vs G, (c) routed vs C-alone text CE,
 (d) chatmix one-pass reasoning, (e) GENERATION: from a chatmix prompt
     (text turn + reasoning prompt up to the answer marker) sample
     greedily and score exact-match of the answer tokens IN a free-
     running transcript; report the byte-text sample too (logged).
Controls (seeds 111/222, from P26 recipe, re-trained here since P26 did
not save them): TF-NAPE d56 + GRU d64x2 monoliths on chatmix -> the
OOD bars (hard-interval task acc, CE ratio 1024/256hard, dyck d12) on
the 304-vocab symbolic streams, i.e. the same bars as the unified row.
PRIOR ART (2026-09-11): Toolformer arXiv 2302.04761 and tool-call
steering arXiv 2605.07990 delegate exactness to EXTERNAL tools; here
the exact "tools" (counter/register/stack experts) are inside the
parameter set and selected by a learned causal gate. Tag ARCH-VET-LM-P27.
"""
import argparse, json, math, os, random, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
CKPT = os.path.join(REPO, "p21_ckpt"); RUNS = os.path.join(REPO, "runs")
import arch_vet_p26 as p26, arch_vet_p26b as p26b, arch_vet_p22 as p22, arch_vet_p21 as p21
import arch_vet_p19 as p19, arch_vet_p13d as p13d, arch_vet_lm as lm, arch_vet_p23 as p23, arch_vet_p25 as p25
V0, VB = p26.V0, p26.VB
A_MARK, T_TASK, EOS, U = p19.A, p19.T_TASK, p19.EOS, 2


def train_lm(m, pool, steps, tag, every=1000):
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3); torch.manual_seed(0); tt = time.time(); hist = []
    for step in range(1, steps + 1):
        x = torch.stack([pool[(step * 8 + i) % len(pool)] for i in range(8)])
        loss = F.cross_entropy(m(x[:, :256]).reshape(-1, VB), x[:, 1:257].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % every == 0:
            hist.append((step, round(loss.item(), 4))); print(f"  [{tag}] step {step}/{steps} loss {loss.item():.4f} ({time.time()-tt:.0f}s)", flush=True)
    m.eval(); return hist


@torch.no_grad()
def generation_probe(model, n=24, seed=8080):
    """Free-running greedy generation on chatmix prompts: exact-match of
    the answer token(s) after A_MARK for track/modk/pair, with the model
    having generated everything after the prompt cut itself (the cut is
    placed at the A_MARK of the LAST task in the stream)."""
    rng = random.Random(seed); ok = tot = 0; per = {"track": [0, 0], "modk": [0, 0], "pair": [0, 0]}; sample = None
    for k in range(n):
        x = p26.gen_chatmix_stream(rng, 256)
        # find last A_MARK preceded by a task with an answer following
        idx = [i for i in range(len(x) - 3) if x[i] == A_MARK]
        if not idx: continue
        cut = idx[-1] + 1                              # model must produce x[cut:]
        # determine family of this task
        j = cut - 2
        while j > 0 and x[j] != T_TASK: j -= 1
        # back up over the full run of T_TASKs (task header is 1-3 T's)
        while j > 0 and x[j - 1] == T_TASK: j -= 1
        if x[j + 1] != T_TASK: fam = "track"
        elif x[j + 2] == p19.ONE: fam = "modk"
        elif x[j + 2] == T_TASK: continue            # dyck: no A_MARK answer
        else: fam = "pair"
        need = 1 if fam != "pair" else 2
        ctx = torch.tensor(x[:cut]).unsqueeze(0); out = []
        for _ in range(need):
            nxt = int(model(ctx)[0, -1].argmax()); out.append(nxt); ctx = torch.cat([ctx, torch.tensor([[nxt]])], 1)
        hit = int(out == x[cut:cut + need]); per[fam][0] += hit; per[fam][1] += 1; ok += hit; tot += 1
        if sample is None and k == 3:
            # also free-run 80 more tokens of text to show the language side
            for _ in range(80):
                nxt = int(model(ctx)[0, -1].argmax()); ctx = torch.cat([ctx, torch.tensor([[nxt]])], 1)
            gen = ctx[0, cut:].tolist()
            sample = "".join(chr(t - V0) if t >= V0 and 32 <= t - V0 < 127 else f"<{t}>" for t in gen)
    return {"exact": round(ok / max(1, tot), 4), "n": tot,
            "per_family": {k: round(v[0] / v[1], 4) if v[1] else None for k, v in per.items()}, "sample": sample}


@torch.no_grad()
def ood_bars_304(m):
    """Unified-row bars for a 304-vocab monolith (symbolic streams only)."""
    acc = lm.task_acc(m, 24, 256, random.Random(666), hard=True)
    vha = p19.make_batches(8, 256, random.Random(777), hard=True); v1024 = p19.make_batches(2, 1024, random.Random(31415))
    ratio = round(lm.val_ce(m, v1024, 1024) / lm.val_ce(m, vha, 256), 3)
    d12 = p13d.rt_close_acc(m, 2, 12, p13d.seg_len(12) + 16)[0]
    return {"pair": acc["pair"], "modk": acc["modk"], "track": acc["track"], "ratio": ratio, "dyck_d12": d12,
            "bars": int(acc["pair"] >= .717) + int(acc["modk"] >= 1) + int(ratio <= .6) + int(d12 >= .85)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default=p25.DEFAULT_SEEDS); ap.add_argument("--ctrl_seeds", default="111,222")
    a = ap.parse_args(); t0 = time.time()
    text_pool = p26.pool_of(p26.gen_text_stream, 384, 777)
    va_text = torch.stack(p26.pool_of(p26.gen_text_stream, 32, 99, src=p26.TXT_VA))
    rows = {}
    for seed in map(int, a.seeds.split(",")):
        outp = f"{RUNS}/p27_s{seed}.json"
        if os.path.exists(outp): rows[seed] = json.load(open(outp)); print(f"[p27] s{seed} cached", flush=True); continue
        torch.manual_seed(seed)
        mA = p21.VETDCC(V0, 24, k=8, K=8); mB = p21.STACKDCC2_D12(V0, 24, k=8, K=8)
        mA.load_state_dict(torch.load(f"{CKPT}/A_s{seed}.pt")["sd"]); mB.load_state_dict(torch.load(f"{CKPT}/B_s{seed}.pt")["sd"])
        g2 = p22.Gate(V0, 8); g2.load_state_dict(torch.load(f"{CKPT}/G_s{seed}.pt")["sd"])
        torch.manual_seed(seed); mC = p26.ByteGRU(VB, 48); pc = f"{CKPT}/C_s{seed}.pt"
        if os.path.exists(pc): mC.load_state_dict(torch.load(pc)["sd"])
        else: hC = train_lm(mC, text_pool, 4000, f"P27-C-s{seed}"); torch.save({"sd": mC.state_dict(), "hist": hC}, pc)
        for m in (mA, mB, mC, g2):
            m.eval()
            for q in m.parameters(): q.requires_grad_(False)
        torch.manual_seed(seed); gm = p26b.GateM(4); pm = f"{CKPT}/GM_s{seed}.pt"
        hg = p26b.HierGate(g2, gm); u = p26.Unified3(mA, mB, mC, hg)
        if os.path.exists(pm): gm.load_state_dict(torch.load(pm)["sd"])
        else:
            joint = [torch.tensor(lm.gen_stream(random.Random(1000 + seed + i), 256)[0]) for i in range(96)]
            joint += p19.gen_mixdd_pool(96, 256, 2000 + seed) + p13d.gen_mix_pool(96, 256, 3000 + seed)
            joint += p26.pool_of(p26.gen_chatmix_stream, 128, 4000 + seed) + p26.pool_of(p26.gen_text_stream, 64, 5000 + seed)
            joint = [j[:257] for j in joint]
            opt = torch.optim.AdamW(gm.parameters(), lr=3e-3); tt = time.time()
            for step in range(1, 1001):
                x = torch.stack([joint[(step * 8 + i) % len(joint)] for i in range(8)]); y = x[:, 1:256]
                with torch.no_grad():
                    la, lb, lc = u.expert_logits(x[:, :256]); P = torch.stack([F.softmax(la, -1), F.softmax(lb, -1), F.softmax(lc, -1)], 2)[:, :255]
                w = F.softmax(hg(x[:, :255]), -1).unsqueeze(-1); p = (w * P).sum(2)
                loss = F.nll_loss(torch.log(p + 1e-9).reshape(-1, VB), y.reshape(-1))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(gm.parameters(), 1.0); opt.step()
                if step % 500 == 0: print(f"  [P27-M-s{seed}] step {step}/1000 loss {loss.item():.4f} ({time.time()-tt:.0f}s)", flush=True)
            torch.save({"sd": gm.state_dict()}, pm)
        gm.eval(); u.eval()
        with torch.no_grad():
            row = {"C_alone_text_ce": p26.decomposed_ce(mC, va_text)["text_ce"],
                   "routed_text_ce": p26.decomposed_ce(u, va_text)["text_ce"],
                   "chatmix_reasoning": p26.chatmix_reasoning(u, 24),
                   "generation": generation_probe(u)}
            same = tot = 0
            for s in p19.make_pool(16, 256, 7) + p13d.gen_mix_pool(16, 256, 7):
                r3 = u.routes(s[:256].unsqueeze(0)).squeeze(0); r2 = g2(s[:256].unsqueeze(0)).argmax(-1).squeeze(0)
                same += int((r3 == r2).sum()); tot += r3.numel()
            row["symbolic_dispatch_identity"] = round(same / tot, 4)
            p21.causal_route = (lambda toks: [min(r, 1) for r in u.routes(torch.tensor(toks).unsqueeze(0)).squeeze(0).tolist()])
            acc = p21.routed_task_acc(mA, mB, 24, 256, random.Random(666), hard=True)
            vha = p19.make_batches(8, 256, random.Random(777), hard=True); v1024 = p19.make_batches(2, 1024, random.Random(31415))
            ratio = round(p21.routed_ce(mA, mB, v1024, 1024) / p21.routed_ce(mA, mB, vha, 256), 3)
            d12 = p21.routed_close_acc(mA, mB, 2, 12, p13d.seg_len(12) + 16)[0]
        row["guard"] = {"pair": acc["pair"], "modk": acc["modk"], "ratio": ratio, "dyck_d12": d12}
        row["bars"] = {"pair": acc["pair"] >= .717, "modk": acc["modk"] >= 1, "ratio": ratio <= .6, "dyck_d12": d12 >= .85}
        row["n_bars"] = sum(row["bars"].values())
        json.dump(row, open(outp, "w")); rows[seed] = row
        print(f"[p27 s{seed}] bars={row['n_bars']} guard={row['guard']} identity={row['symbolic_dispatch_identity']} text {row['routed_text_ce']}/{row['C_alone_text_ce']} "
              f"chatmix {row['chatmix_reasoning']} gen {row['generation']['exact']} {row['generation']['per_family']}", flush=True)
        if row["generation"]["sample"]: print(f"[p27 s{seed}] SAMPLE: {row['generation']['sample'][:200]!r}", flush=True)
    # ---- controls OOD bars
    chat_pool = p26.pool_of(p26.gen_chatmix_stream, 512, 12345); ctrl = {}
    for name in ("tf", "gru"):
        ctrl[name] = {}
        for seed in map(int, a.ctrl_seeds.split(",")):
            torch.manual_seed(seed)
            m = p23.TFCtrl(VB, d=56, nh=4, depth=2, mlp=2, pe="nape") if name == "tf" else p26.ByteGRU(VB, 64, 2)
            pth = f"{CKPT}/CTRL_{name}_s{seed}.pt"
            if os.path.exists(pth): m.load_state_dict(torch.load(pth)["sd"]); m.eval()
            else: train_lm(m, chat_pool, 6000, f"P27-ctrl-{name}-s{seed}", 2000); torch.save({"sd": m.state_dict()}, pth)
            r = ood_bars_304(m); r["generation"] = generation_probe(m); r["text_ce"] = p26.decomposed_ce(m, va_text)["text_ce"]
            ctrl[name][str(seed)] = r; print(f"[p27 ctrl {name} s{seed}] {r}", flush=True)
    s = sorted(rows); n = len(s)
    summ = {"n_seeds": n, "bars_passed_per_seed": [rows[k]["n_bars"] for k in s],
            "all4": {"k": sum(rows[k]["n_bars"] == 4 for k in s), "n": n, "wilson95": p25.wilson(sum(rows[k]["n_bars"] == 4 for k in s), n)},
            "identity": [rows[k]["symbolic_dispatch_identity"] for k in s],
            "text_ce_routed": [rows[k]["routed_text_ce"] for k in s], "text_ce_C": [rows[k]["C_alone_text_ce"] for k in s],
            "chatmix_modk": [rows[k]["chatmix_reasoning"]["modk"] for k in s], "chatmix_pair": [rows[k]["chatmix_reasoning"]["pair"] for k in s],
            "chatmix_dyck": [rows[k]["chatmix_reasoning"]["dyck_close"] for k in s],
            "gen_exact": [rows[k]["generation"]["exact"] for k in s]}
    out = {"tag": "ARCH-VET-LM-P27", "protocol": __doc__[:1200], "summary": summ, "per_seed": {str(k): rows[k] for k in s}, "controls": ctrl, "wall_s": round(time.time() - t0)}
    with open("log.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    print("[P27] SUMMARY", json.dumps(summ), flush=True); print("[P27] DONE", flush=True)


if __name__ == "__main__":
    main()
