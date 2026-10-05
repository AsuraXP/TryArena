"""ARCH-VET P43 (cycle 91) — THE IN-RANGE HAZARD NUMBER: token-class CE decomposition.

WHY: frontier item 5 ("in-range parity: 256-hard CE 2.2-2.5 vs TF 1.96") has
been a vague gap since P1. Hard-interval streams at L=256 are dominated by
RANDOM FILLERS (8-way uniform => ln 8 = 2.079 nats/token irreducible), so the
stream-mean CE hides where the gap actually lives. This probe splits the NLL
by TARGET-TOKEN CLASS so the parity gap becomes a number per class:
   filler   : targets 13..20 (irreducible ~2.079 for every model)
   answer   : targets whose previous token is the A marker (track symbol /
              modk m) or the pair answer position (second symbol after a
              pair query) — the tokens the mechanisms exist for
   one-run  : target == ONE inside a modk run (continuation vs stop)
   bracket  : targets 29..32 (dyck open/close)
   marker   : targets in {T_TASK, A, BOS, EOS}
   other    : everything else (track/pair symbols at non-answer positions)
MODELS: unified routed system (seed 111/222 ckpts, 116,858p), expert A alone
(same ckpts), TF-ALiBi control (P23_alibi_s111.pt, 42,672p, trained on the
joint pool, 6000 steps). STREAMS: 256-train (rng 999), 256-hard (rng 777),
1024 (rng 31415) — the p19.eval_full generators, identical batches.
No training. Output: per-class mean NLL and token counts; "hazard number" =
answer-class NLL at 256-hard, unified vs TF.
"""
import sys, json, time, argparse, torch, torch.nn.functional as F, random

sys.argv_backup = sys.argv; sys.argv = ["x", "--lens", "256"]
_p41 = {"__file__": "arch_vet_p41.py"}; exec(open("arch_vet_p41.py").read().split("def main")[0], _p41); sys.argv = sys.argv_backup
build_unified, p19, p23 = _p41["build_unified"], _p41["p19"], None
_p23 = {"__file__": "arch_vet_p23.py"}; exec(open("arch_vet_p23.py").read().split("def main")[0], _p23); TFCtrl = _p23["TFCtrl"]
lm = p19.make_batches.__globals__
V, T_TASK, A, ONE, BOS, EOS, BRK = p19.V, lm["T_TASK"], lm["A"], lm["ONE"], lm["BOS"], lm["EOS"], lm["BRK"]
FILL = set(range(13, 21))


def classes(x):
    """x: [B, L+1] stream. returns class id per target position t (target x[:, t+1]), [B, L]."""
    B, L1 = x.shape; tgt = x[:, 1:]; prev = x[:, :-1]; c = torch.full(tgt.shape, 5, dtype=torch.long)   # other
    c[torch.isin(tgt, torch.tensor(sorted(FILL)))] = 0
    ans = (prev == A)
    # pair answer: stream grammar "T q a ... q a" — the token after a repeated query symbol; approximate as prev in TRACK-range
    # and prev == some earlier token of the same task is costly; we keep the A-marker definition (track/modk) and flag pair
    # answers as targets following a token that equals the pair query symbol seen earlier in the same task:
    for b in range(B):
        seen = {}; task_start = 0
        for t in range(L1 - 1):
            if x[b, t] == T_TASK: seen = {}
            tok = int(x[b, t])
            if tok in seen and seen[tok] is not None and t + 1 < L1: ans[b, t] = True
            if tok not in FILL and tok not in (T_TASK, A, BOS, EOS, ONE) and tok < 29 and t + 1 < L1:
                nxt = int(x[b, t + 1]); seen[tok] = nxt if nxt not in FILL and nxt not in (T_TASK, A) else None
    c[ans] = 1
    c[(tgt == ONE) & (prev == ONE)] = 2
    c[(tgt >= BRK) & (tgt < BRK + 4)] = 3
    c[torch.isin(tgt, torch.tensor([T_TASK, A, BOS, EOS]))] = 4
    return c


NAMES = ["filler", "answer", "one-run", "bracket", "marker", "other"]


@torch.no_grad()
def decompose(model, xs, L):
    lg = model(xs[:, :L + 1]); nll = -F.log_softmax(lg[..., :V] if lg.shape[-1] > V else lg, -1).gather(-1, xs[:, 1:L + 1].unsqueeze(-1)).squeeze(-1)
    c = classes(xs[:, :L + 1]); out = {}
    for k, n in enumerate(NAMES):
        m = c == k; out[n] = (round(float(nll[m].mean()), 3) if m.any() else None, int(m.sum()))
    out["ALL"] = (round(float(nll.mean()), 4), int(nll.numel())); return out


class AOnly(torch.nn.Module):
    def __init__(self, u): super().__init__(); self.u = u
    def forward(self, x): return self.u.u3.expert_logits(x)[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="111,222"); a = ap.parse_args()
    streams = {"256_train": (p19.make_batches(8, 256, random.Random(999)), 256), "256_hard": (p19.make_batches(8, 256, random.Random(777), hard=True), 256),
               "1024": (p19.make_batches(2, 1024, random.Random(31415)), 1024)}
    out = {"tag": "ARCH-VET-LM-P43", "protocol": __doc__[:1500], "rows": {}}
    models = {}
    for s in map(int, a.seeds.split(",")):
        u = build_unified(s); models[f"unified_s{s}"] = u; models[f"A_s{s}"] = AOnly(u)
    tf = TFCtrl(V, pe="alibi"); tf.load_state_dict(torch.load("p21_ckpt/P23_alibi_s111.pt")["sd"]); tf.eval(); models["TF-ALiBi_s111"] = tf
    for mn, m in models.items():
        for sn, (xs, L) in streams.items():
            d = decompose(m, xs, L); out["rows"][f"{mn}|{sn}"] = d
            print(f"[p43 {mn:16s} {sn:9s}] " + " ".join(f"{k}={v[0]}({v[1]})" for k, v in d.items()), flush=True)
    # hazard number
    def g(mn, sn, k): return out["rows"][f"{mn}|{sn}"][k][0]
    for s in map(int, a.seeds.split(",")):
        print(f"[p43 HAZARD s{s}] 256-hard answer-class NLL: unified {g(f'unified_s{s}','256_hard','answer')} | A {g(f'A_s{s}','256_hard','answer')} | TF {g('TF-ALiBi_s111','256_hard','answer')}"
              f" ; filler: unified {g(f'unified_s{s}','256_hard','filler')} TF {g('TF-ALiBi_s111','256_hard','filler')} (ln8={round(torch.log(torch.tensor(8.)).item(),3)})", flush=True)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P43] DONE", flush=True)


if __name__ == "__main__":
    main()
