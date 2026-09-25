"""C81 theory: exact policy-oracle on the REAL eval streams (not i.i.d. subsets).
Replays p32.gen_stream(hard, n_fixed) with the deployed KRB write/read/evict policy (smart or rand kick, consumed-first,
same hash buffers, same seeds as p32.acc) and reports the fraction of queries whose binding is still in the bank.
This is the number the learned K must match if 'learning follows structure' is exact; any learned surplus over it
must come from the K=8 recency buffer; any deficit is a learning error."""
import random, torch, sys
import arch_vet_p32 as p32
V = p32.V
def hashes(S, b, seed=85):
    NB = S // b; g = torch.Generator().manual_seed(seed)
    return torch.randn(V, NB, generator=g).argmax(-1).tolist(), torch.randn(V, NB, generator=g).argmax(-1).tolist()
def replay(R, n, S, b, smart, L, rng, streams=10):
    H1, H2 = hashes(S, b); NB = S // b; keys = set(R["keys"]); ok = tot = 0
    for _ in range(streams):
        x = p32.gen_stream(rng, R, L, True, n)
        tags = [[None] * b for _ in range(NB)]; used = [[False] * b for _ in range(NB)]
        def free(bk):
            for j in range(b):
                if tags[bk][j] is None or used[bk][j]: return j
        def alt(k, bk): return H2[k] if bk == H1[k] else H1[k]
        def put(k, v):
            for bk in (H1[k], H2[k]):
                for j in range(b):
                    if tags[bk][j] and tags[bk][j][0] == k: tags[bk][j] = (k, v); used[bk][j] = False; return
                j = free(bk)
                if j is not None: tags[bk][j] = (k, v); used[bk][j] = False; return
            bk = H1[k]; vc = 0
            if smart:
                cand = [j for j in range(b) if alt(tags[bk][j][0], bk) != bk and free(alt(tags[bk][j][0], bk)) is not None]
                if cand: vc = cand[0]
            victim = tags[bk][j := vc]; tags[bk][j] = (k, v); used[bk][j] = False
            ab = alt(victim[0], bk); fj = free(ab)
            if fj is not None: tags[ab][fj] = victim; used[ab][fj] = False
        i = 1
        while i < len(x) - 1:
            t = x[i]
            if t == p32.A and i + 2 < len(x):        # query: A key -> value
                k = x[i + 1]; v = x[i + 2]; hit = False
                for bk in (H1[k], H2[k]):
                    for j in range(b):
                        if tags[bk][j] and tags[bk][j][0] == k: hit = tags[bk][j][1] == v; used[bk][j] = True
                ok += hit; tot += 1; i += 3
            elif t in keys and i + 1 < len(x) and x[i - 1] != p32.A:   # write: key value
                put(t, x[i + 1]); i += 2
            else: i += 1
    return round(ok / tot, 4), tot
if __name__ == "__main__":
    for rn, S, arms, ns in [("R2", 8, [(1, False), (2, False), (2, True), (4, False), (4, True)], (4, 6, 8)), ("R5", 16, [(4, True), (8, True)], (8, 12, 16))]:
        R = p32.regime(rn)
        for b, smart in arms:
            print(f"{rn} S{S} b{b} {'smart' if smart else 'rand '}: " + "  ".join(f"n{n} {replay(R, n, S, b, smart, 256 if n <= 4 else 320, random.Random(600 + n))[0]:.4f}" for n in ns), flush=True)
