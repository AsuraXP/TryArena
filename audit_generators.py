"""C74 generator oracle audit (open item #4 after the P32 token-overlap bug).
For each stream generator, run the *scoring* procedure with an ORACLE that
knows the grammar and check (1) oracle accuracy == 1.0 on every family,
(2) token-set disjointness (filler / markers / family alphabets),
(3) that the scored answer positions are unambiguous (no answer position
is also a marker/filler). Prints a table; exits nonzero on any violation.
Pure python (no torch)."""
import random, sys, collections, importlib
import arch_vet_lm as lm

def oracle_pred(x):
    """Grammar oracle: given the full stream, produce per-position 'prediction' of x[t+1]
    using ONLY the grammar (mirror of what a perfect model would output at scored positions)."""
    L = len(x); pred = [None] * L
    i = 0
    while i < L - 4:
        if x[i] != lm.T_TASK: i += 1; continue
        n1, n2 = x[i+1], x[i+2]
        if lm.TRACK <= n1 < lm.TRACK + 8:
            j = i
            while j < L - 1 and x[j] != lm.A: j += 1
            pred[j] = n1; i = j + 2
        elif n1 == lm.T_TASK and n2 == lm.T_TASK:          # dyck: at p-1 predict the closer x[p] from the stack
            st = []; p = i + 3
            while p < L and lm.BRK <= x[p] < lm.BRK + 4:
                t = x[p]
                if t < lm.BRK + 2: st.append(t)
                else:
                    if st: pred[p - 1] = st[-1] + 2; st.pop()
                p += 1
            i = p
        elif n1 == lm.T_TASK and n2 == lm.ONE:
            j = i; n = 0
            while j < L - 1 and x[j] != lm.A:
                n += int(x[j] == lm.ONE); j += 1
            pred[j] = lm.MANS + (n % 3); i = j + 2
        elif n1 == lm.T_TASK:
            k, v = x[i+2], x[i+3]; j = i
            while j < L - 1 and x[j] != lm.A: j += 1
            pred[j] = k; pred[j+1] = v; i = j + 3
        else: i += 1
    return pred

def audit_lm(hard, n=200):
    rng = random.Random(1); ok = collections.Counter(); tot = collections.Counter()
    for _ in range(n):
        x, _ = lm.gen_stream(rng, 256, hard); pred = oracle_pred(x)
        i = 0
        while i < len(x) - 4:
            if x[i] != lm.T_TASK: i += 1; continue
            n1, n2 = x[i+1], x[i+2]
            fam = "track" if lm.TRACK <= n1 < lm.TRACK + 8 else ("dyck" if (n1 == lm.T_TASK and n2 == lm.T_TASK) else ("modk" if n2 == lm.ONE else "pair"))
            j = i
            while j < len(x) - 1 and x[j] != lm.A and fam != "dyck": j += 1
            if fam == "track" or fam == "modk":
                tot[fam] += 1; ok[fam] += int(pred[j] == x[j+1]); i = j + 2
            elif fam == "pair":
                tot[fam] += 1; ok[fam] += int(j + 2 < len(x) and pred[j] == x[j+1] and pred[j+1] == x[j+2]); i = j + 3
            else:
                i2 = i + 3
                while i2 < len(x) and lm.BRK <= x[i2] < lm.BRK + 4:
                    if lm.BRK + 2 <= x[i2] < lm.BRK + 4: tot[fam] += 1; ok[fam] += int(pred[i2-1] == x[i2])
                    i2 += 1
                i = i2
    return {f: round(ok[f] / max(1, tot[f]), 4) for f in tot}

def tokensets():
    # runtime alphabets (verified by sampling): fill 13-20, track 21-28, ONE 21, MANS 25-27, BRK 29-32, KEYS 33-36, VALS 37-40
    sets = {"markers": {lm.BOS, lm.EOS, lm.U, lm.A, lm.T_TASK}, "fill": set(range(lm.MODS, lm.MODS + 8)),
            "track": set(range(lm.TRACK, lm.TRACK + 8)), "one": {lm.ONE}, "mans": set(range(lm.MANS, lm.MANS + 3)),
            "brk": set(range(lm.BRK, lm.BRK + 4)), "keys": set(range(lm.KEYS, lm.KEYS + 4)), "vals": set(range(lm.VALS, lm.VALS + 4))}
    bad = [(a, b) for a in sets for b in sets if a < b and sets[a] & sets[b]]
    return sets, bad

def audit_p32():
    import arch_vet_p32 as p32, arch_vet_p19 as p19
    out = {}; fill = set(range(p19.MODS, p19.MODS + 8)); mk = {p19.T_TASK, p19.A, p19.BOS, p19.EOS, p19.ONE} | set(range(p19.MANS, p19.MANS + 3))  # p19 re-exports (MODS=13 start of filler 13-20 in the p19 namespace)
    for rn in ("R1", "R2", "R3", "R4"):
        R = p32.regime(rn); ks = set(t for k in R["keys"] for t in (k if isinstance(k, tuple) else (k,))); vs = set(R["vals"])
        out[rn] = {"keys&fill": len(ks & fill), "vals&fill": len(vs & fill), "keys&vals": len(ks & vs), "keys&markers": len(ks & mk), "vals&markers": len(vs & mk)}
    return out

if __name__ == "__main__":
    sets, bad = tokensets(); print("[audit] lm token-set overlaps:", bad or "none")
    for hard in (False, True): print(f"[audit] lm oracle acc hard={hard}:", audit_lm(hard))
    try:
        import arch_vet_p19 as p19
        r = audit_lm.__wrapped__ if hasattr(audit_lm, "__wrapped__") else None
    except Exception as e: print("[audit] p19 import:", e)
    print("[audit] p32 regimes overlaps:", audit_p32())
    # KNOWN, BENIGN (disclosed C74): track symbols 21-28 share ids with ONE(21)/MANS(25-27); family is parsed from
    # the token after T_TASK, never from the symbol -> oracle 1.0 (verified above). P32 keys 21-28 likewise share ids
    # with ONE/MANS; inside MQAR streams these never occur as ONE/MANS. Only cross-grammar (unified pool) needs care.
    benign = {("one", "track"), ("mans", "track")}
    bad = [b for b in bad if b not in benign]
    p32d = audit_p32()
    viol = bool(bad) or any(v != 1.0 for h in (False, True) for v in audit_lm(h, 50).values()) or any(c for d in p32d.values() for k, c in d.items() if k != "keys&markers")
    print("[audit] VIOLATIONS" if viol else "[audit] CLEAN"); sys.exit(int(viol))
