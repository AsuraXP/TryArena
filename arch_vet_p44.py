"""ARCH-VET P44 (cycle 92) — expert C_chat: the byte expert retrained on REAL DIALOGUE (Tier-1 chat).

WHY: chat.py works but expert C was trained on this repo's logs, so it speaks log-ese. The chat
tiers need a conversational lexical expert. Corpus: corpus/chat_dialogues.txt = 1,823 English
dialogues / 3,997 turns / 156 KB flattened from the PyPI package chatterbot-corpus 1.3.3 (English,
BSD; coding/botprofile files dropped). 90/10 split by dialogue.
STREAM: turn-aware — a stream is a sequence of whole dialogues, each turn emitted as the system's own
turn grammar  U <bytes> EOS , so the model learns reply boundaries, not random spans.
MODEL/RECIPE: identical to P40 (ByteGRU VB d48, 43,600 p, AdamW 3e-3, batch 8 x 256, clip 1,
4000 steps, ctor under manual_seed(seed)). Saves p21_ckpt/CCHAT_s{seed}.pt; chat.py --C chat loads it.
HONEST: 156 KB is ~30 epochs at this budget -> the model will partly memorise the corpus; val CE on
the held-out 10% is the number to watch, and the chat quality will be "canned but English".
"""
import argparse, json, os, random, time, torch, torch.nn.functional as F
torch.set_num_threads(1)
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
import arch_vet_p26 as p26, arch_vet_p19 as p19
CKPT = "p21_ckpt"; V0, VB, U, EOS, BOS = p26.V0, p26.VB, p26.U, p26.EOS, p26.BOS

CORPUS = os.environ.get("P44_CORPUS", "corpus/chat_dialogues.txt")   # big = corpus/chat_dialogues_big.txt (Cornell movie pairs via nlpia 0.5.2 + chatterbot)
_dl = [d.split("\n") for d in open(CORPUS, encoding="utf-8").read().strip().split("\n\n")]
_cut = int(len(_dl) * 0.9); DLG_TR, DLG_VA = _dl[:_cut], _dl[_cut:]


def gen_chat_stream(rng, L=256, src=None):
    src = src or DLG_TR; x = [BOS]
    while len(x) < L + 1:
        for t in rng.choice(src):
            x += [U] + [V0 + b for b in t.encode("utf-8")] + [EOS]
    return x[:L + 1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", default="111"); ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--d", type=int, default=48); ap.add_argument("--layers", type=int, default=1); a = ap.parse_args()
    tag = ("" if (a.d == 48 and a.layers == 1) else f"_d{a.d}l{a.layers}") + ("_big" if "big" in CORPUS else "")
    va = torch.stack([torch.tensor(gen_chat_stream(random.Random(99 + i), 256, DLG_VA)) for i in range(32)])
    out = {"tag": "ARCH-VET-LM-P44", "protocol": __doc__, "per_seed": {}, "corpus": {"dialogues": len(_dl), "train": len(DLG_TR), "val": len(DLG_VA)}}
    for seed in [int(s) for s in a.seeds.split(",")]:
        torch.manual_seed(seed); m = p26.ByteGRU(VB, a.d, a.layers); n = p19.n_params(m)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3); torch.manual_seed(0); frng = random.Random(4242 + seed); t0 = time.time(); hist = []
        def val():
            m.eval()
            with torch.no_grad(): ce = float(F.cross_entropy(m(va[:, :256]).reshape(-1, VB), va[:, 1:257].reshape(-1)))
            m.train(); return round(ce, 4)
        for step in range(1, a.steps + 1):
            x = torch.stack([torch.tensor(gen_chat_stream(frng, 256)) for _ in range(8)])
            loss = F.cross_entropy(m(x[:, :256]).reshape(-1, VB), x[:, 1:257].reshape(-1))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if step % 500 == 0:
                ce = val(); hist.append((step, round(loss.item(), 4), ce)); print(f"  [P44-CCHAT{tag}-s{seed}] step {step}/{a.steps} train {loss.item():.4f} val {ce} ({time.time()-t0:.0f}s)", flush=True)
        ce = val(); torch.save({"sd": m.state_dict(), "hist": hist}, f"{CKPT}/CCHAT{tag}_s{seed}.pt")
        r = {"params": n, "d": a.d, "layers": a.layers, "steps": a.steps, "val_chat_ce": ce, "hist": hist}; out["per_seed"][seed] = r; print(f"[p44 CCHAT s{seed}] {r}", flush=True)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P44] DONE", flush=True)


if __name__ == "__main__":
    main()
