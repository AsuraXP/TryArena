"""ARCH-VET P23 (cycle 62) — FAIR TRANSFORMER CONTROL: length-generalizing
positional schemes at MATCHED PARAMS and MATCHED DATA vs the unified row.

WHY: every "beats the Transformer" number so far is against TFMicro
(2L d16, sinusoidal PE, 8,144p) — a control that is (a) 5x smaller than
the 43,074p unified system and (b) known to collapse at L>train because
of absolute-PE extrapolation. Claim-integrity item #2: does the unified
row still beat a Transformer when the Transformer is given (i) the
same parameter budget, (ii) the union of both expert corpora, (iii) the
combined step budget (4000+2000), and (iv) a positional scheme that is
KNOWN to length-generalize?

PRIOR ART (searched 2026-09-10):
 - arXiv 2402.01032 "Repeat After Me": "ALiBi and NoPE transformers
   dramatically outperform the RoPE model on longer inputs" (copy task,
   length generalization). -> NoPE + ALiBi are the strongest
   length-generalizing choices for small decoders; RoPE is the weakest.
 - arXiv 2506.16640 (ICLR 2026, ASEntmax/NAPE): "RoPE models poorly
   generalize beyond 4x"; ALiBi induces attention windows with a clear
   cutoff; NoPE induces content-driven sparsity; NAPE = half heads NoPE,
   half ALiBi. -> we include a NAPE arm (mixed heads) as the strongest
   published small-decoder recipe.
 - Kazemnejad et al. 2023 (NoPE) / Press et al. 2021 (ALiBi) originals.
ARMS (all 2L, d48, 4 heads, mlp x2, pre-LN, ~42k params):
   TF-NOPE  : no positional signal at all (causal mask only)
   TF-ALIBI : per-head linear distance bias, slopes 2^-(8h/nh)
   TF-NAPE  : heads 0,1 NoPE ; heads 2,3 ALiBi
TRAIN: joint pool = vanilla P9 pool (512) + mixdd depth-diverse pool
(256) + P13D deep-mix pool (256) => union of expert A's and expert B's
corpora; 6000 steps batch 8 (= A 4000 + B 2000); seeds 111/222.
EVAL: identical unified row — hard task acc (pair>=.717, modk=1),
CE ratio 1024/256hard (<=.6), dyck close d12 (>=.85) via p19.eval_full
+ p19.eval_dyck; bars_passed_per_seed. Reference: unified learned-gate
system [4,4] (pair .8585/.8774, ratio .596/.503, dyck .9838/.9496).
"""
import argparse, json, math, os, random, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
import torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
torch.backends.mha.set_fastpath_enabled(False)   # eval fastpath corrupts per-head float masks (verified: 3.08 max diff)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p19 as p19
import arch_vet_p13d as p13d
V = p19.V


class TFCtrl(nn.Module):
    def __init__(self, V, d=48, nh=4, depth=2, mlp=2, pe="nope"):
        super().__init__()
        self.E = nn.Embedding(V, d); nn.init.normal_(self.E.weight, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d, nh, d * mlp, batch_first=True,
                                       norm_first=True, dropout=0.0)
            for _ in range(depth)])
        self.ln = nn.LayerNorm(d); self.head = nn.Linear(d, V)
        self.nh, self.pe = nh, pe
        slopes = torch.tensor([2.0 ** (-(8.0 * (h + 1) / nh)) for h in range(nh)])
        if pe == "nape":                    # half heads NoPE (slope 0)
            slopes[: nh // 2] = 0.0
        if pe == "nope":
            slopes[:] = 0.0
        self.register_buffer("slopes", slopes)

    def attn_exact_chunked(self, blk, h, q_chunk=512):
        """Memory-bounded EXACT causal attention with per-head ALiBi bias
        (same math as the dense src_mask path; used when L > 2048 where a
        dense nh x L x L float mask would exceed 4 GB — dyck d10-d12)."""
        sa = blk.self_attn; B, L, d = h.shape; nh = self.nh; hd = d // nh
        qkv = F.linear(h, sa.in_proj_weight, sa.in_proj_bias)
        q, k, v = qkv.split(d, -1)
        q = q.view(B, L, nh, hd).transpose(1, 2)
        k = k.view(B, L, nh, hd).transpose(1, 2)
        v = v.view(B, L, nh, hd).transpose(1, 2)
        out = torch.empty_like(q); pos = torch.arange(L, device=h.device)
        for s0 in range(0, L, q_chunk):
            s1 = min(L, s0 + q_chunk)
            sc = torch.matmul(q[:, :, s0:s1], k[:, :, :s1].transpose(-1, -2)) / math.sqrt(hd)
            dist = (pos[s0:s1, None] - pos[None, :s1]).float()
            sc = sc - self.slopes[None, :, None, None] * dist.clamp(min=0)
            sc = sc.masked_fill(dist[None, None] < 0, float("-inf"))
            out[:, :, s0:s1] = torch.matmul(torch.softmax(sc, -1), v[:, :, :s1])
        o = out.transpose(1, 2).reshape(B, L, d)
        return sa.out_proj(o)

    def block_fwd(self, blk, h):
        h = h + self.attn_exact_chunked(blk, blk.norm1(h))
        return h + blk.linear2(blk.activation(blk.linear1(blk.norm2(h))))

    def mask(self, B, L, dev):
        i = torch.arange(L, device=dev)
        dist = (i[:, None] - i[None, :]).clamp(min=0).float()   # L,L
        bias = -self.slopes[:, None, None] * dist[None]          # nh,L,L
        causal = torch.triu(torch.full((L, L), float("-inf"), device=dev), 1)
        m = bias + causal[None]                                  # nh,L,L
        return m.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.nh, L, L)

    def forward(self, x):
        B, L = x.shape
        h = self.E(x)
        if L > 2048:
            for b in self.blocks:
                h = self.block_fwd(b, h)
            return self.head(self.ln(h))
        m = self.mask(B, L, x.device)
        for b in self.blocks:
            h = b(h, src_mask=m)
        return self.head(self.ln(h))


def joint_pool():
    pool = list(p19.make_pool(512, 256, 12345))
    pool += list(p19.gen_mixdd_pool(256, 256, 12345))
    pool += list(p13d.gen_mix_pool(256, 256, 12345))
    return [p[:257] if p.shape[0] >= 257 else
            F.pad(p, (0, 257 - p.shape[0]), value=int(p[-1])) for p in pool]


def bars(row, dy):
    ae = row["acc_eval_interval"]
    return int(ae["pair"] >= .717) + int(ae["modk"] >= 1.0) + \
        int(row["len_ratio_1024_over_256hard"] <= .6) + int(dy["close_d12"] >= .85)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="nope,alibi,nape")
    ap.add_argument("--seeds", default="111,222")
    ap.add_argument("--steps", type=int, default=6000)
    a = ap.parse_args()
    t0 = time.time(); pool = joint_pool(); print(f"[p23] pool {len(pool)}", flush=True)
    out = {"tag": "ARCH-VET-LM-P23", "protocol": __doc__.split("ARMS")[0][-600:],
           "arms": {}}
    for arm in a.arms.split(","):
        per = {}
        for s in map(int, a.seeds.split(",")):
            torch.manual_seed(s); m = TFCtrl(V, pe=arm)
            npar = p19.n_params(m)
            print(f"[p23] {arm} s{s} params={npar}", flush=True)
            hist = p19.train_arm(f"P23-{arm}-s{s}", m, pool, a.steps, 8)
            m.eval()
            with torch.no_grad():
                row = p19.eval_full(m, f"TF-{arm}", s)
                dy = p19.eval_dyck(m, f"TF-{arm}", s)
            b = bars(row, dy)
            print(f"[p23 TF-{arm} s{s}] pair={row['acc_eval_interval']['pair']} "
                  f"modk={row['acc_eval_interval']['modk']} "
                  f"ratio={row['len_ratio_1024_over_256hard']} "
                  f"dyck_d12={dy['close_d12']} bars={b}", flush=True)
            per[str(s)] = {"params": npar, "loss_curve": hist, **row,
                           "dyck_close": dy, "bars": b}
        out["arms"][f"TF-{arm}"] = {
            "per_seed": per,
            "summary": {"bars_passed_per_seed": [v["bars"] for v in per.values()],
                        "pair": [v["acc_eval_interval"]["pair"] for v in per.values()],
                        "modk": [v["acc_eval_interval"]["modk"] for v in per.values()],
                        "ratio": [v["len_ratio_1024_over_256hard"] for v in per.values()],
                        "dyck_d12": [v["dyck_close"]["close_d12"] for v in per.values()]}}
        print(f"[P23 {arm}] SUMMARY {json.dumps(out['arms'][f'TF-{arm}']['summary'])}",
              flush=True)
    out["wall_s"] = round(time.time() - t0)
    with open("log.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    print("RESULT " + json.dumps(out), flush=True); print("DONE", flush=True)


if __name__ == "__main__":
    main()
