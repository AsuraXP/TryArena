"""ARCH-VET P40 (cycle 79) — expert C retrained on FRESH streams (C78c finding: the 384-stream pool overfits;
val 2.786 -> 2.505 at identical d48 / 43,600 p / 4000 st / AdamW 3e-3 when every step samples new streams from
the full 1 MB train split). Saves p21_ckpt/CF_s{seed}.pt; arch_vet_p36.py --C F loads it. Nothing else changes:
same ctor seed (manual_seed(seed) before ByteGRU), same optimizer, same step budget; only the data pipeline."""
import argparse, json, os, random, time, torch, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p26 as p26, arch_vet_p19 as p19
CKPT = "p21_ckpt"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="111,222,333"); ap.add_argument("--steps", type=int, default=4000); a = ap.parse_args()
    va = torch.stack(p26.pool_of(p26.gen_text_stream, 32, 99, src=p26.TXT_VA))
    out = {"tag": "ARCH-VET-LM-P40", "protocol": __doc__, "per_seed": {}}
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed); m = p26.ByteGRU(p26.VB, 48); n = p19.n_params(m)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3); torch.manual_seed(0); frng = random.Random(4242 + seed); t0 = time.time(); hist = []
        for step in range(1, a.steps + 1):
            x = torch.stack([torch.tensor(p26.gen_text_stream(frng, 256)) for _ in range(8)])
            loss = F.cross_entropy(m(x[:, :256]).reshape(-1, p26.VB), x[:, 1:257].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if step % 1000 == 0:
                m.eval(); ce = p26.decomposed_ce(m, va)["text_ce"]; m.train(); hist.append((step, round(loss.item(), 4), ce))
                print(f"  [P40-CF-s{seed}] step {step}/{a.steps} train {loss.item():.4f} val {ce} ({time.time()-t0:.0f}s)", flush=True)
        m.eval(); ce = p26.decomposed_ce(m, va)["text_ce"]
        torch.save({"sd": m.state_dict(), "hist": hist}, f"{CKPT}/CF_s{seed}.pt")
        r = {"params": n, "val_text_ce": ce, "hist": hist}; out["per_seed"][seed] = r; print(f"[p40 CF s{seed}] {r}", flush=True)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P40] DONE", flush=True)

if __name__ == "__main__":
    main()
