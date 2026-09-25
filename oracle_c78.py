"""C78 oracle: write-policy lever at fixed geometry (S8, b=2/4). Policies at both-buckets-full:
 rand1  = kick random cell of H1, 1 kick (deployed P38)            | smart1 = kick the victim whose ALTERNATE bucket has a free/consumed cell (1-level BFS)
 rand2  = random victim, 2 kicks                                     | smart2 = smart victim, then rand kick (2 kicks)
 Hall ceiling from theory_c77 (b2 .955, b4 .998 at n8). Prior art: BFS-insertion cuckoo (Fotakis et al. 2005; Li-Andersen-Kaminsky-Freedman 2014 MemC3/'cuckoo path search')."""
import random, torch, sys
import arch_vet_p32 as p32
from oracle_c77 import hashes
def sim(n, S, b, policy, seed=85, trials=4000, rng=random.Random(0)):
    R = p32.regime("R2"); V = 64; NB = S // b
    H1, H2 = hashes(V, NB, seed); H1 = H1.tolist(); H2 = H2.tolist()
    ok = tot = 0
    for _ in range(trials):
        keys = rng.sample(R["keys"], n); vals = [rng.choice(R["vals"]) for _ in keys]
        tags = [[None] * b for _ in range(NB)]; used = [[False] * b for _ in range(NB)]
        def free(bk):
            for j in range(b):
                if tags[bk][j] is None or used[bk][j]: return j
            return None
        def alt(k, bk): return H2[k % V] if bk == H1[k % V] else H1[k % V]
        def put(k, v, depth, maxk):
            for bk in (H1[k % V], H2[k % V]):
                for j in range(b):
                    if tags[bk][j] and tags[bk][j][0] == k: tags[bk][j] = (k, v); used[bk][j] = False; return True
                j = free(bk)
                if j is not None: tags[bk][j] = (k, v); used[bk][j] = False; return True
            if depth >= maxk: return False
            bk = H1[k % V]
            if policy.startswith("smart"):
                cand = [j for j in range(b) if free(alt(tags[bk][j][0], bk)) is not None and alt(tags[bk][j][0], bk) != bk]
                j = rng.choice(cand) if cand else rng.randrange(b)
            else: j = rng.randrange(b)
            victim = tags[bk][j]; tags[bk][j] = (k, v); used[bk][j] = False
            return put(victim[0], victim[1], depth + 1, maxk)
        maxk = int(policy[-1])
        for k, v in zip(keys, vals): put(k, v, 0, maxk)
        for k, v in zip(keys, vals):
            hit = False
            for bk in (H1[k % V], H2[k % V]):
                for j in range(b):
                    if tags[bk][j] and tags[bk][j][0] == k: hit = tags[bk][j][1] == v; used[bk][j] = True
            ok += hit; tot += 1
    return ok / tot
if __name__ == "__main__":
    for b in (2, 4):
        for pol in ("rand1", "smart1", "rand2", "smart2"):
            print(f"S8 b{b} {pol:7s} n6 {sim(6,8,b,pol):.3f}  n8 {sim(8,8,b,pol):.3f}", flush=True)
