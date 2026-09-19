# -*- coding: utf-8 -*-
"""
ARCH-VET-LM-P22 (cycle 62) — LEARNED ROUTER: DOES THE UNIFIED ROW
SURVIVE WHEN THE DISPATCH IS TRAINED INSTEAD OF HAND-SPECIFIED?
P21/P21C certified the UNIFIED ROW (43,074p: pair 3/3, modk 3/3,
length-invariance ratio 3/3, dyck d12 3/3; joint pass pair ~1.0 with
dyck .84-.92) but the dispatcher is HAND-SPECIFIED — a structural
grammar-derived rule (family from tokens <t, dyck to the specialist
iff entering bracket depth >=3). Standing program honesty clause: the
certified mechanism must not depend on injected task knowledge.
P22 = replace the rule with a TRAINED gate:
  FEATURES (causal): one-hot(x_t) only (48 dims) fed to a GRU
    (hidden 8) -> Linear(8, 2) -> softmax weights over {A, B}. The
    gate sees NO grammar labels, NO depth feature, NO family tag —
    just the token stream, exactly as a deployable router would.
    Params ~1.4k (48*8*3 + 8*2 + biases) on top of the frozen
    43,074p experts.
  TRAINING: experts FROZEN (no_grad); only the gate trains, by
    minimising the CE of the PROBABILITY-MIXTURE output
    p = w_A * softmax(lgA) + w_B * softmax(lgB) on a JOINT training
    pool (vanilla P9 streams + mixdd depth-diverse streams + pure
    deep-dyck streams) — the loss signal alone must discover the
    dispatch policy, including the depth dependence that P21B found
    to be essential (L-DEPTH-AWARE-DISPATCH) — the GRU's state is
    the only mechanism it has to track depth.
  EVAL (seeds 111/222, same frozen experts as P21/P21C): (1) unified
    row with HARD gate selection (argmax), (2) joint heterogeneous
    pass, (3) AGREEMENT with the structural depth-aware router, and
    (4) gate-vs-oracle. Success criterion (falsifiable): unified row
    bars within a few points of P21C (pair >=.717, modk 1.0, ratio
    <=.6, dyck d12 >=.85 => >=3/4 bars on both seeds). If the learned
    gate collapses toward A-only (the A baseline is already strong on
    everything except deep dyck), the result is a clean negative:
    loss-driven dispatch does NOT discover the depth boundary from
    the token stream alone at this scale, and the certified unified
    row would then rest on injected structure (logged honestly).
Prior art (searched 2026-09-08/10): Shazeer et al. 2017 sparsely
gated MoE + Fedus et al. 2022 (Switch) — gating learned by the task
loss with a load-balance auxiliary term (P22 omits the auxiliary
term deliberately: with 2 experts and a strong generalist, load
balance must emerge from the loss, not from a hand-set prior;
omission is logged); arXiv 2409.14981 (module specialization needs
task-specific data separation — P22 keeps that: the experts ARE
trained per-regime, only the DISPATCH is learned); arXiv 2408.00508
(block-operations/Multiplexer: routing inductive bias for
compositional generalization); program lineage: P21/P21B/P21C
(this is the claim-integrity ablation of the unified row's router).
Tag ARCH-VET-LM-P22.
"""
import argparse, json, os, random, time

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(REPO, "p21_ckpt")
T0 = time.time()

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)

import arch_vet_p21 as p21
import arch_vet_p21c as p21c
import arch_vet_p19 as p19
import arch_vet_p13d as p13d
import arch_vet_lm as lm

V = p21.V
T_TASK = p19.T_TASK


class Gate(nn.Module):
    """Causal token-only gate: one-hot(x_t) -> GRU(8) -> 2 logits."""
    def __init__(self, V, h=8):
        super().__init__()
        self.gru = nn.GRU(V, h, batch_first=True)
        self.head = nn.Linear(h, 2)

    def forward(self, x):                      # x: (B, L) long
        oh = F.one_hot(x, V).float()           # (B, L, V)
        h, _ = self.gru(oh)
        return self.head(h)                    # (B, L, 2)


def joint_pool(seed, n=192):
    """Training pool: vanilla + mixdd + pure deep-dyck streams."""
    prng = random.Random(seed)
    xs = []
    for i in range(n):
        if i % 3 == 0:
            xs.append(torch.tensor(lm.gen_stream(prng, 256)[0]))
        elif i % 3 == 1:
            xs.append(torch.tensor(p19.gen_mixdd_stream(prng, 256)))
        else:
            xs.append(torch.tensor(p13d.gen_mix_stream(prng, 256)))
    return torch.stack(xs)


def train_gate(mA, mB, pool, seed, steps=1500, batch=8, lr=3e-3):
    torch.manual_seed(seed)
    g = Gate(V, 8)
    opt = torch.optim.AdamW(g.parameters(), lr=lr)
    n_pool = len(pool)
    t0 = time.time()
    for step in range(1, steps + 1):
        sel = [(step * batch + i) % n_pool for i in range(batch)]
        x = torch.stack([pool[i] for i in sel])
        y = x[:, 1:256]
        with torch.no_grad():
            pA = F.softmax(mA(x[:, :256]), -1)[:, :255, :]
            pB = F.softmax(mB(x[:, :256]), -1)[:, :255, :]
        w = F.softmax(g(x[:, :255]), -1).unsqueeze(-1)   # (B,L,2,1)
        p = w[:, :, 0] * pA + w[:, :, 1] * pB
        loss = F.nll_loss(torch.log(p + 1e-9).reshape(-1, V),
                          y.reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(g.parameters(), 1.0)
        opt.step()
        if step % 250 == 0:
            print(f"  [gate s{seed}] step {step}/{steps} loss "
                  f"{float(loss):.4f} ({time.time()-t0:.0f}s)", flush=True)
    g.eval()
    return g


@torch.no_grad()
def gate_routes(g, toks):
    """Hard causal gate decisions for one stream (argmax)."""
    x = torch.tensor(toks).unsqueeze(0)
    lg = g(x)
    return lg.argmax(-1).squeeze(0).tolist()


@torch.no_grad()
def agreement(g, structural, pool, n=24):
    hit = tot = 0
    for i in range(min(n, len(pool))):
        toks = pool[i].tolist()
        gr = gate_routes(g, toks)
        sr = structural(toks)
        for a, b in zip(gr, sr):
            tot += 1
            hit += int(a == b)
    return round(hit / tot, 4) if tot else float("nan")


def evaluate(mA, mB, g, structural, tag, seed):
    p21.causal_route = (lambda toks: gate_routes(g, toks))  # patch
    row = {}
    row["task_acc_hard"] = p21.routed_task_acc(
        mA, mB, 24, 256, random.Random(666), hard=True)
    row["task_acc_train"] = p21.routed_task_acc(
        mA, mB, 24, 256, random.Random(555), hard=False)
    vha = p19.make_batches(8, 256, random.Random(777), hard=True)
    v1024 = p19.make_batches(2, 1024, random.Random(31415))
    ce_h = p21.routed_ce(mA, mB, vha, 256)
    ce_1024 = p21.routed_ce(mA, mB, v1024, 1024)
    row["ce"] = {"256_hard": ce_h, "1024": ce_1024}
    row["len_ratio_1024_over_256hard"] = round(ce_1024 / max(1e-9, ce_h), 3)
    cl = {}
    for d in range(1, 13):
        Ld = 256 if p13d.seg_len(d) <= 252 else p13d.seg_len(d) + 16
        nst = 8 if Ld <= 256 else (4 if Ld <= 4096 else 2)
        cl[f"close_d{d}"] = p21.routed_close_acc(mA, mB, nst, d, Ld)[0]
    row["dyck_close_routed"] = cl
    row["joint_mixdd"] = p21.joint_mixed_eval(mA, mB, 24, 256,
                                             corpus="mixdd")
    # gate health: how often did it pick the specialist overall?
    pools = {"vanilla": p19.make_pool(24, 256, 7),
             "mixdd": p19.gen_mixdd_pool(24, 256, 7),
             "deepdyck": p13d.gen_mix_pool(24, 256, 7)}
    row["gate_b_share"] = {}
    for k, pl in pools.items():
        tot = hit = 0
        for s in pl:
            r = gate_routes(g, s.tolist())
            tot += len(r); hit += sum(r)
        row["gate_b_share"][k] = round(hit / max(1, tot), 4)
    row["agreement_with_structural"] = agreement(g, structural, pools["mixdd"])
    row["agreement_vanilla"] = agreement(g, structural, pools["vanilla"])
    row["agreement_deepdyck"] = agreement(g, structural, pools["deepdyck"])
    row["gate_params"] = sum(p.numel() for p in g.parameters())
    print(f"[p22 {tag} s{seed}] pair={row['task_acc_hard']['pair']} "
          f"modk={row['task_acc_hard']['modk']} "
          f"ratio={row['len_ratio_1024_over_256hard']} "
          f"dyck_d12={cl['close_d12']} joint_pair="
          f"{row['joint_mixdd']['pair']} joint_dyck="
          f"{row['joint_mixdd']['dyck_close']} "
          f"B_share={row['gate_b_share']} "
          f"agree={row['agreement_with_structural']}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="111,222")
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    result = {"tag": "ARCH-VET-LM-P22",
              "protocol": "learned router over frozen experts: causal "
                          "token-only GRU gate (one-hot(x_t) -> GRU(8) "
                          "-> 2 logits, ~1.4k params) trained by the "
                          "composed probability-mixture CE on a joint "
                          "pool (vanilla + mixdd + pure deep-dyck), "
                          "workers frozen; HARD argmax dispatch at "
                          "eval. Metrics: unified row (4 bars), joint "
                          "heterogeneous pass, gate-B-share per corpus, "
                          "agreement with the P21C structural "
                          "depth-aware router. Reference P21C: pair "
                          "3/3 .865, modk 1.0 3/3, ratio 3/3 .473, "
                          "dyck d12 3/3 .969.",
              "arms": {}}
    rows = []
    structural = p21c.make_deep_router(3)
    for seed in seeds:
        torch.manual_seed(seed)
        mA = p21.VETDCC(V, 24, k=8, K=8)
        mB = p21.STACKDCC2_D12(V, 24, k=8, K=8)
        mA.load_state_dict(torch.load(os.path.join(CKPT, f"A_s{seed}.pt"))["sd"])
        mB.load_state_dict(torch.load(os.path.join(CKPT, f"B_s{seed}.pt"))["sd"])
        mA.eval(); mB.eval()
        for p_ in mA.parameters(): p_.requires_grad_(False)
        for p_ in mB.parameters(): p_.requires_grad_(False)
        pool = joint_pool(12345 + seed)
        print(f"[p22] training gate seed {seed} steps {a.steps}",
              flush=True)
        g = train_gate(mA, mB, pool, seed, a.steps)
        torch.save({"sd": g.state_dict(), "seed": seed,
                    "params": sum(p.numel() for p in g.parameters())},
                   os.path.join(CKPT, f"G_s{seed}.pt"))
        rows.append(evaluate(mA, mB, g, structural, "learned", seed))
        p21.causal_route = p21c.make_deep_router(3)   # restore
    n = len(rows)
    md12 = [r["dyck_close_routed"]["close_d12"] for r in rows]
    pairs = [r["task_acc_hard"]["pair"] for r in rows]
    ratios = [r["len_ratio_1024_over_256hard"] for r in rows]
    mods = [r["task_acc_hard"]["modk"] for r in rows]
    summary = {
        "n_seeds": n,
        "dyck_close_d12_dist": md12,
        "basin_dyck_d12_ge_85": sum(1 for x in md12 if x >= 0.85) / n,
        "pair_eval_dist": pairs,
        "basin_pair_ge_717": sum(1 for x in pairs if x >= 0.717) / n,
        "len_ratio_dist": ratios,
        "basin_ratio_le_6": sum(1 for x in ratios if x <= 0.6) / n,
        "modk_eval_dist": mods,
        "basin_modk_eq_1": sum(1 for m in mods if m == 1.0) / n,
        "joint_pair_dist": [r["joint_mixdd"]["pair"] for r in rows],
        "joint_dyck_dist": [r["joint_mixdd"]["dyck_close"] for r in rows],
        "gate_b_share_mixdd": [r["gate_b_share"]["mixdd"] for r in rows],
        "agreement_mixdd": [r["agreement_with_structural"] for r in rows],
        "gate_params": [r["gate_params"] for r in rows],
        "bars_passed_per_seed": [
            int(p >= 0.717) + int(m == 1.0) + int(x <= 0.6)
            + int(d >= 0.85) for p, m, x, d in zip(pairs, mods, ratios, md12)]}
    print(f"[P22 LEARNED ROUTER] SUMMARY {json.dumps(summary)}", flush=True)
    result["arms"]["learned-gate"] = {
        "per_seed": {str(s): r for s, r in zip(seeds, rows)},
        "summary": summary}
    result["wall_s"] = round(time.time() - T0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open(os.path.join(REPO, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
