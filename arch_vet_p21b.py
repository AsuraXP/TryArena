# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-P21B (cycle 61, Phase-4 iteration on P21) — WHY DOES THE
ROUTED SYSTEM FAIL THE LENGTH-INVARIANCE RATIO, AND CAN A ROUTER
PATCH FIX IT? P21 (modular fusion, 43,074p) certified pair 3/3,
modk 3/3, dyck d12 3/3 in ONE system and showed pair~1.0 WITH dyck
.85-.99 in the SAME heterogeneous pass — but the routed CE ratio
1024/256hard was 1.058/1.238/0.884 (basin 0/3 <=.6) while expert A
alone holds .457-.538 (P16, same corpus/budget). Routed CE is also
far above A-alone at 256-hard (3.05/2.68/3.13 vs ~2.4-2.5).
HYPOTHESES (falsifiable in this script):
  H1 CONTEXT SHIFT: B was trained on ~pure-dyck streams (P13D
     gen_mix_stream), so inside interleaved vanilla streams its
     token-level predictions on dyck bodies degrade (while its
     CLOSE-TYPE accuracy stays high — cf. P21 close_d12 .95-.98 vs
     routed CE 3.0+).
  H2 DEPTH-2 GAP: the 1024 bank is the EASY regime (dyck depth 2);
     depth 2 has only weight 1/15 in the d-1 depth curriculum, so B
     is weak there -> CE(1024) inflated -> ratio > 1.
  H3 SAMPLING/GRANULARITY artifact: routed CE mixes two models'
     calibrations per position; if most hard-stream CE mass sits on
     non-dyck positions the ratio is driven by A, not by routing.
MEASUREMENT: for each seed, on the SAME banks as eval_full
(256-hard x8, 512 x2, 1024 x2), compute CE for five variants:
  routed      = P21's causal structural router
  A-only      = expert A everywhere (no routing)
  B-only      = expert B everywhere
  deep-route  = routed, but dyck-body positions whose enclosing
                segment depth < 3 go to A instead
  ensemble    = (logitA + logitB)/2 at routed dyck-body positions,
                A elsewhere
and decompose routed/A/B CE by TARGET CLASS (bracket-token target =
dyck body vs all other tokens) so H3 is decidable directly.
DECISION RULE: if ensemble (or deep-route) restores ratio <=.6 while
keeping the P21 answer-accuracy profile, that variant becomes the
certified router (P21C re-run); if no variant moves the ratio, the
failure is in the specialists' context expectations, not the router
-> next patch = co-training B on interleaved (not pure-dyck) streams
(L-CONTEXT-SHIFT), which is a corpus change, not a router change.
Prior art: arXiv 2409.14981 (module specialization needs
task-specific data — the flip side measured here), MoE
load-balancing/calibration literature (logit-scale mismatch across
experts), Zhou et al. 2024 arXiv 2402.09371 (OOD fragility).
Tag ARCH-VET-LM-P21B (diagnostic; no new architecture claimed).
"""
import json, os, random, time, sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(REPO, "p21_ckpt")
T0 = time.time()

import torch
torch.set_num_threads(1)

import arch_vet_p21 as p21
import arch_vet_p19 as p19
import arch_vet_lm as lm

BRK = p19.BRK


def scan_depth(toks):
    """depth of the enclosing segment for each target token index."""
    d = 0
    out = [0] * len(toks)
    for i, t in enumerate(toks):
        if BRK <= t < BRK + 2:
            d += 1
            out[i] = d
        elif BRK + 2 <= t < BRK + 4:
            out[i] = d          # segment depth of this close
            d = max(0, d - 1)
        else:
            out[i] = 0
    return out


@torch.no_grad()
def ce_variants(mA, mB, xs, L):
    """Per-stream masks (each stream has its own router/depth/class
    pattern); returns CE for the five dispatch variants + decomposed
    routed/A/B CE by target class."""
    acc = {}
    for b in range(xs.shape[0]):
        x = xs[b:b + 1, :L + 1]
        y = x[:, 1:L + 1]
        toks = x[0].tolist()
        lgA = mA(x)[:, :L, :]
        lgB = mB(x)[:, :L, :]
        routes = p21.causal_route(toks)[:L]
        rt = torch.tensor(routes, dtype=torch.bool).view(1, -1, 1)
        dep = scan_depth(toks)[1:L + 1]
        is_br = torch.tensor([BRK <= t < BRK + 4
                              for t in y[0].tolist()]).view(1, -1)
        def nll(lg):
            return -torch.log_softmax(lg, -1).gather(
                -1, y.unsqueeze(-1)).squeeze(-1)
        nllA, nllB = nll(lgA), nll(lgB)
        nllE = nll(torch.where(rt, (lgA + lgB) / 2.0, lgA))
        rmask = rt.squeeze(-1)
        deep_mask = rmask & torch.tensor(
            [dd >= 3 for dd in dep]).view(1, -1)
        variants = {"ce_routed": torch.where(rmask, nllB, nllA),
                    "ce_A": nllA, "ce_B": nllB,
                    "ce_deeproute": torch.where(deep_mask, nllB, nllA),
                    "ce_ensemble": nllE}
        for k, v in variants.items():
            acc.setdefault(k, []).append(v)
        acc.setdefault("_routed_dyck", []).append(
            torch.where(rmask, nllB, nllA)[is_br])
        acc.setdefault("_routed_other", []).append(
            torch.where(rmask, nllB, nllA)[~is_br])
        acc.setdefault("_A_dyck", []).append(nllA[is_br])
        acc.setdefault("_B_dyck", []).append(nllB[is_br])
        acc["_n_dyck"] = acc.get("_n_dyck", 0) + int(is_br.sum())
    out = {"L": L}
    for k, vs in acc.items():
        if k == "_n_dyck":
            continue
        v = torch.cat(vs)
        out[k.lstrip("_") if k.startswith("_") else k] = round(
            float(v.mean()), 4)
    out["n_dyck_targets"] = acc["_n_dyck"]
    return out


def main():
    pool_hard = p19.make_batches(8, 256, random.Random(777), hard=True)
    bank512 = p19.make_batches(2, 512, random.Random(31337))
    bank1024 = p19.make_batches(2, 1024, random.Random(31415))
    result = {"tag": "ARCH-VET-LM-P21B",
              "protocol": "P21 router/CE diagnostic: five dispatch "
                          "variants (routed / A-only / B-only / "
                          "depth>=3-route / logit-ensemble) on the "
                          "eval_full banks (256-hard x8, 512 x2, "
                          "1024 x2) + routed/A/B CE decomposed by "
                          "target class (bracket = dyck body vs "
                          "other). Seeds 111/222/333, checkpoints "
                          "from P21. No new training.",
              "arms": {}}
    rows = []
    for seed in (111, 222, 333):
        torch.manual_seed(seed)
        mA = p21.VETDCC(p21.V, 24, k=8, K=8)
        mB = p21.STACKDCC2_D12(p21.V, 24, k=8, K=8)
        mA.load_state_dict(torch.load(
            os.path.join(CKPT, f"A_s{seed}.pt"))["sd"])
        mB.load_state_dict(torch.load(
            os.path.join(CKPT, f"B_s{seed}.pt"))["sd"])
        mA.eval(); mB.eval()
        r256 = ce_variants(mA, mB, pool_hard, 256)
        r512 = ce_variants(mA, mB, bank512, 512)
        r1024 = ce_variants(mA, mB, bank1024, 1024)
        ratios = {}
        for v in ("ce_routed", "ce_A", "ce_B", "ce_deeproute",
                  "ce_ensemble"):
            ratios[v.replace("ce_", "")] = round(
                r1024[v] / max(1e-9, r256[v]), 3)
        row = {"seed": seed, "ce_256hard": r256, "ce_512": r512,
               "ce_1024": r1024, "ratios_1024_over_256hard": ratios}
        rows.append(row)
        print(f"[p21b s{seed}] ratios={ratios}", flush=True)
        print(f"[p21b s{seed}] decomposed 256hard: "
              f"routed_dyck={r256['routed_dyck']} "
              f"A_dyck={r256['A_dyck']} B_dyck={r256['B_dyck']} "
              f"routed_other={r256['routed_other']} "
              f"(n_dyck={r256['n_dyck_targets']})", flush=True)
    result["arms"]["router-variants"] = {
        "per_seed": {str(r["seed"]): r for r in rows}}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open(os.path.join(REPO, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
