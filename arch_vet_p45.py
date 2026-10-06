"""ARCH-VET P45 (cycle 93) — TIER-2 CHAT: exact conversational memory (the K-organ primitive lifted to words).

WHY: C92 showed the 50-100k-param lexical expert produces prompt-independent replies — the user's words are
overwritten in a 48-96-float hidden state long before the reply is finished. The architecture's answer is not
capacity: conditioning on what was said is an EXACT operation done by a memory organ.
MECHANISM (deterministic, O(1) per token, zero learned parameters — same primitive as the certified K expert:
exact key -> value binding with last-write-wins recency):
  WRITE  every user turn is tokenised to words; for each position the bigram key (w_{i-1}, w_i) -> w_{i+1}
         (and the end-of-utterance marker after the last word) is written. The write predicate is hand-wired
         ("the user spoke"), consistent with the project's honest status on predicates.
  READ   the reply is generated word by word. At each word boundary the last two words of the running context
         (the user's final two words when the reply is empty) form a key; on a hit the stored next word is COPIED
         (chain continues while keys keep hitting, stops at the stored end marker); on a miss expert C generates
         the next word byte-by-byte with the full router in the loop (chat.py machinery).
TEST (the brutal one): plant N facts in a long dialogue ("my name is Dana", "i live in Oslo", ...), fill 40+
distractor turns, then ask ("what is my name?"). Score exact presence of the planted value in the reply.
Arms: MEM (C + memory) vs C alone (same checkpoint, memory off). Prediction: C alone ~0; MEM close to 1.0
independent of distance (that is the whole point: distance costs nothing in an exact store).
"""
import sys, re, json, random, time, argparse, torch
sys.argv_backup = sys.argv; sys.argv = ["x"]
_c = {"__file__": "chat.py"}; exec(open("chat.py").read().split("\ndef main():")[0], _c); sys.argv = sys.argv_backup
build_unified, reply, enc, dec, p26 = _c["build_unified"], _c["reply"], _c["enc"], _c["dec"], _c["p26"]
V0, U, EOS, BOS, VB = _c["V0"], _c["U"], _c["EOS"], _c["BOS"], _c["VB"]
END = "<end>"


def words(s): return re.findall(r"[a-z0-9']+|[.?!,]", s.lower())


class WordMemory:
    def __init__(self): self.m = {}; self.writes = 0

    def write(self, text):
        w = words(text) + [END]
        for i in range(1, len(w) - 1): self.m[(w[i - 1], w[i])] = w[i + 1]; self.writes += 1

    def read(self, w1, w2): return self.m.get((w1, w2))


class MemChat:
    def __init__(self, u, mem_on=True, temp=0.6, topk=8, window=224):
        self.u, self.mem_on, self.temp, self.topk, self.window = u, mem_on, temp, topk, window; self.mem = WordMemory(); self.hist = [BOS]; self.stats = {"copied": 0, "generated": 0}

    @torch.no_grad()
    def gen_word(self, ctx_toks):
        """one word from expert C (bytes until space / EOS), full router in the loop via chat.reply machinery."""
        toks = list(ctx_toks); out = []
        for _ in range(24):
            ctx = torch.tensor([toks[-self.window:]]); r = int(self.u.routes(ctx)[0, -1])
            lg = (self.u.u3.mC(ctx)[0, -1] if r != 2 else self.u(ctx)[0, -1]).clone(); lg[1:V0] = -1e9; lg = lg / self.temp
            v, i = lg.topk(self.topk); m = torch.full_like(lg, -1e9); m[i] = v; t = int(torch.multinomial(torch.softmax(m, -1), 1))
            if t == EOS: return dec(out), True
            toks.append(t); out.append(t)
            if t == V0 + 32 and out: return dec(out), False
        return dec(out), False

    def turn(self, text, max_words=16):
        if self.mem_on: self.mem.write(text)
        self.hist = self.hist + enc(text); ctx = list(self.hist) + [U]; uw = words(text); said = []; reply_words = []
        for _ in range(max_words):
            key = (said[-2], said[-1]) if len(said) >= 2 else (uw[-2], uw[-1]) if len(uw) >= 2 else None
            hit = self.mem.read(*key) if (self.mem_on and key) else None
            if hit is not None:
                if hit == END: break
                w = hit + " "; self.stats["copied"] += 1
            else:
                w, end = self.gen_word(ctx); self.stats["generated"] += 1
                if not w.strip(): break
            ctx += [V0 + b for b in w.encode("utf-8")]; reply_words.append(w); said += words(w)
            if hit is None and end: break
        self.hist = ctx + [EOS]; return "".join(reply_words).strip()


FACTS = [("my name is {v}", "what is my name?", ["dana", "milo", "priya", "tomas", "yuki", "abeni", "lars", "noor"]),
         ("i live in {v}", "where do i live?", ["oslo", "lima", "kyoto", "accra", "perth", "quito", "bergen", "malta"]),
         ("my dog is called {v}", "what is my dog called?", ["biscuit", "pepper", "juno", "atlas", "mochi", "rex", "olive", "ziggy"]),
         ("my favorite color is {v}", "what is my favorite color?", ["green", "violet", "amber", "teal", "crimson", "indigo", "ochre", "navy"]),
         ("i work as a {v}", "what do i work as?", ["baker", "pilot", "nurse", "welder", "farmer", "tailor", "diver", "miner"])]
DISTRACT = ["how are you doing today?", "tell me something interesting.", "do you like music?", "it is raining here.", "i am a bit tired.",
            "what do you think about movies?", "that is funny.", "ok.", "really?", "i see.", "what else?", "do you dream?", "nice.", "hmm.", "go on."]


def run_test(u, mem_on, seed, n_distract=40):
    rng = random.Random(seed); bot = MemChat(u, mem_on=mem_on); planted = []
    for tmpl, q, vals in FACTS:
        v = rng.choice(vals); bot.turn(tmpl.format(v=v)); planted.append((q, v))
    for _ in range(n_distract): bot.turn(rng.choice(DISTRACT))
    hits = 0; log = []
    for q, v in planted:
        r = bot.turn(q); ok = v in words(r); hits += ok; log.append((q, v, r, ok))
    return hits / len(planted), log, bot.stats


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--C", default="p21_ckpt/CCHAT_d96l1_big_s111.pt"); ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--distract", default="40"); ap.add_argument("--arms", default="C-alone,MEM")
    a = ap.parse_args(); u = build_unified(111)
    sd = torch.load(a.C)["sd"]; d = sd["head.weight"].shape[1]; layers = sum(1 for k in sd if k.startswith("g.weight_ih_l")); u.u3.mC = p26.ByteGRU(VB, d, layers); u.u3.mC.load_state_dict(sd); u.u3.mC.eval()
    out = {"tag": "ARCH-VET-LM-P45", "protocol": __doc__[:1800], "C": a.C, "arms": {}}
    for nd in map(int, a.distract.split(",")):
        for arm, mem_on in [(x, x == "MEM") for x in a.arms.split(",")]:
            accs = []
            for s in map(int, a.seeds.split(",")):
                t0 = time.time(); acc, log, st = run_test(u, mem_on, s, nd); accs.append(acc)
                print(f"[p45 {arm} d{nd} s{s}] recall {acc:.2f} stats {st} ({time.time()-t0:.0f}s)", flush=True)
                for q, v, r, ok in log: print(f"      {'OK ' if ok else 'MISS'} {q} -> {r!r} (want {v})", flush=True)
            out["arms"][f"{arm}_d{nd}"] = {"recall": accs, "mean": round(sum(accs) / len(accs), 3)}; print(f"[p45 {arm} d{nd}] mean recall {out['arms'][f'{arm}_d{nd}']['mean']}", flush=True)
    open("log.jsonl", "a").write(json.dumps(out) + "\n"); print("[P45] DONE", flush=True)


if __name__ == "__main__":
    main()
