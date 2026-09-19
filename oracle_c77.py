"""C77 oracle: cuckoo placement policies at load 1.0 (R2: 16 keys, 8 slots, n bindings queried once).
Policies: cf1 = consumed-first, 1 kick (P37); blocked(b) = b cells per bucket, S/b buckets, 2 choices, consumed-first; S16 control.
Prior art: Pagh-Rodler 2001 (1/2 threshold), Kirsch-Mitzenmacher-Wieder 2008 (stash), Dietzfelbinger-Weidling 2007 (blocked cuckoo >80% load)."""
import random, torch
import arch_vet_p32 as p32
def hashes(V, S, seed):
    g = torch.Generator().manual_seed(seed); return torch.randn(V, S, generator=g).argmax(-1), torch.randn(V, S, generator=g).argmax(-1)
def sim(n, S, b=1, kicks=1, seed=85, trials=4000, rng=random.Random(0)):
    R = p32.regime("R2"); V = 64
    H1, H2 = hashes(V, S // b, seed); H1 = H1.tolist(); H2 = H2.tolist()
    ok = tot = 0
    for _ in range(trials):
        keys = rng.sample(R["keys"], n); vals = [rng.choice(R["vals"]) for _ in keys]
        tags = [[None] * b for _ in range(S // b)]; used = [[False] * b for _ in range(S // b)]
        def free(bk):
            for j in range(b):
                if tags[bk][j] is None or used[bk][j]: return j
            return None
        def put(k, v, depth):
            for bk in (H1[k % V], H2[k % V]):
                for j in range(b):
                    if tags[bk][j] == k: tags[bk][j] = (k, v); return True
                j = free(bk)
                if j is not None: tags[bk][j] = (k, v); used[bk][j] = False; return True
            if depth >= kicks: return False
            bk = H1[k % V]; j = rng.randrange(b); victim = tags[bk][j]; tags[bk][j] = (k, v); used[bk][j] = False
            return put(victim[0], victim[1], depth + 1) if victim else True
        # tags entries store (k,v) tuples; normalise
        for k, v in zip(keys, vals): put(k, v, 0)
        for k, v in zip(keys, vals):
            hit = False
            for bk in (H1[k % V], H2[k % V]):
                for j in range(b):
                    if tags[bk][j] and tags[bk][j][0] == k: hit = tags[bk][j][1] == v; used[bk][j] = True
            ok += hit; tot += 1
    return ok / tot
if __name__ == "__main__":
    for name, S, b, kk in [("cf1 S8", 8, 1, 1), ("blocked2 S8", 8, 2, 1), ("blocked4 S8", 8, 4, 1), ("blocked2 S8 k2", 8, 2, 2), ("cf1 S16", 16, 1, 1), ("blocked2 S16", 16, 2, 1)]:
        print(f"{name:16s} n4 {sim(4,S,b,kk):.3f}  n8 {sim(8,S,b,kk):.3f}", flush=True)
