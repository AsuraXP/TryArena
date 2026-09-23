"""C77 theory: policy-independent ceiling for the keyed register bank.
For n distinct keys, S cells, 2 hash choices per key (buckets of b cells), any placement policy
(unlimited kicks, any eviction) can retain at most nu(G) bindings, where nu is the maximum
b-capacitated matching of the key->bucket bipartite graph induced by the drawn keys (Hall's theorem /
cuckoo-graph orientability; Pagh-Rodler 2001, Dietzfelbinger-Weidling 2007). Recall_n <= E[nu]/n.
We compute this bound EXACTLY for the R2 key set (16 keys) under the deployed hash (seed 85) by
enumerating subsets, and compare with the oracle policies and the learned K."""
import itertools, statistics, torch
import arch_vet_p32 as p32
from oracle_c77 import hashes
def maxflow(keys, H1, H2, nb, b):
    # tiny augmenting-path max flow: keys -> buckets (cap b)
    load = [0]*nb; match = {}
    def aug(k, seen):
        for bk in (H1[k], H2[k]):
            if bk in seen: continue
            seen.add(bk)
            if load[bk] < b: load[bk]+=1; match[k]=bk; return True
            for k2,bk2 in list(match.items()):
                if bk2==bk and aug(k2, seen): match[k]=bk; return True
        return False
    return sum(aug(k, set()) for k in keys)
def bound(n, S, b, seed=85, V=64):
    R = p32.regime("R2"); K = R["keys"]; nb = S//b
    H1, H2 = hashes(V, nb, seed); H1=[H1[k%V].item() for k in range(V)]; H2=[H2[k%V].item() for k in range(V)]
    vals = [maxflow(sub, H1, H2, nb, b)/n for sub in itertools.combinations(K, n)]
    return statistics.mean(vals), min(vals)
if __name__ == "__main__":
    print("policy-independent ceiling E[nu]/n (min over subsets) — deployed hash seed 85")
    for S,b in [(8,1),(8,2),(8,4),(16,1),(16,2)]:
        print(f"S{S} b{b}: " + "  ".join(f"n{n} {bound(n,S,b)[0]:.3f}(min {bound(n,S,b)[1]:.2f})" for n in (4,6,8)), flush=True)
    print("hash-seed spread S8 b1 n8:", [round(bound(8,8,1,s)[0],3) for s in (1,2,3,4,5,85)])
    print("hash-seed spread S8 b2 n8:", [round(bound(8,8,2,s)[0],3) for s in (1,2,3,4,5,85)])
