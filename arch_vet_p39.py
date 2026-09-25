"""ARCH-VET P39 (cycle 77b) — FLUENCY CAPACITY PROBE for expert C.
Question: is the unified system's text CE (2.67-2.79 bits... nats/byte, == C-alone
on 10/10 seeds) capacity-bound, data-bound, or schedule-bound?
Arms (same text pool as P26, same optimizer, seed 111):
  d48 L1 4000 st  (== C in the unified system, control)
  d48 L1 16000 st (schedule)
  d96 L1 4000 st, d160 L1 4000 st, d96 L2 4000 st (capacity)
Because text CE == C-alone on every seed (gate cost 0), any gain here
transfers 1:1 to the unified model without touching the exact bars.
Prior art: byte-level LM scaling is smooth in params (Xue et al. 2022 ByT5;
Kaplan 2020 scaling laws) — the question is only where THIS 1 MB corpus's
data ceiling sits vs the ~44k-param C.
"""
import argparse, json, os, time, torch, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p26 as p26, arch_vet_p19 as p19

def run(name, d, layers, steps, text_pool, va, fresh=False):
    torch.manual_seed(111); m = p26.ByteGRU(p26.VB, d, layers); n = p19.n_params(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3); torch.manual_seed(0); t0 = time.time(); hist = []
    import random; frng = random.Random(4242)
    for step in range(1, steps + 1):
        # C78c: `fresh` samples 8 NEW streams from the full train split every step (P39 found the 384-stream pool
        # (~98 KB, ~83 epochs in 4000 st) OVERFITS: val 2.666@1k -> 2.786@4k while train 1.84 -> 1.6).
        x = torch.stack([torch.tensor(p26.gen_text_stream(frng, 256)) for _ in range(8)]) if fresh else torch.stack([text_pool[(step * 8 + i) % len(text_pool)] for i in range(8)])
        loss = F.cross_entropy(m(x[:, :256]).reshape(-1, p26.VB), x[:, 1:257].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 1000 == 0:
            m.eval(); ce = p26.decomposed_ce(m, va)["text_ce"]; m.train(); hist.append((step, round(loss.item(), 4), ce))
            print(f"  [P39-{name}] step {step}/{steps} train {loss.item():.4f} val {ce} ({time.time()-t0:.0f}s)", flush=True)
    m.eval(); ce = p26.decomposed_ce(m, va)["text_ce"]
    r = {"params": n, "val_text_ce": ce, "hist": hist, "wall_s": round(time.time() - t0)}
    print(f"[p39 {name}] {r}", flush=True); return r

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arms", default="d48L1x4k,d96L1x4k,d160L1x4k,d96L2x4k,d48L1x16k"); ap.add_argument("--fresh", action="store_true"); a = ap.parse_args()
    text_pool = p26.pool_of(p26.gen_text_stream, 384, 777); va = torch.stack(p26.pool_of(p26.gen_text_stream, 32, 99, src=p26.TXT_VA))
    out = {"tag": "ARCH-VET-LM-P39", "fresh": a.fresh, "protocol": __doc__[:1200], "arms": {}}
    for arm in a.arms.split(","):
        d = int(arm[1:arm.index("L")]); L = int(arm[arm.index("L") + 1:arm.index("x")]); st = int(arm[arm.index("x") + 1:-1]) * 1000
        out["arms"][arm] = run(arm + ("F" if a.fresh else ""), d, L, st, text_pool, va, fresh=a.fresh)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P39] DONE", flush=True)

if __name__ == "__main__":
    main()
