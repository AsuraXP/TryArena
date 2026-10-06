# -*- coding: utf-8 -*-
"""
ARCH-VET parallel runner / harness (cycle 61). OPERATIONAL, not a
mechanism experiment — no new science, no new laws. Purpose: stop
wasting half of the 2-core sandbox. Measured 2026-09-08 on this box:
one 4000-step run at ~21k params L=256 batch 8 single-threaded =
0.72 s/step = 48 min; TWO concurrent single-threaded processes each
finish 40 steps in 29.0 s (identical to solo, wall 31 s, user 60 s
= true 2-core parallelism); THREE concurrent degrade each run to
45.6 s (57% slower) -> 2 is the optimum. Also: runs only progress
while the chat is active and this box has been re-provisioned 8x, so
per-run results must be durable artifacts.

USAGE
  # one arm+seed = one subprocess, writes runs/<run_id>.json
  python3 arch_vet_runner.py run --corpus mixdd --arm VETDCC \
      --seed 111 --steps 4000 [--pool 512] [--dyck-share 0.5]
  # driver: N arms/seeds, at most `--jobs` concurrent (default 2),
  # resumable (skips run_ids that already have a result), then
  # merges into ONE RESULT line + log.jsonl
  python3 arch_vet_runner.py drive --tag ARCH-VET-LM-P21 \
      --specs "mixdd:VETDCC:111 mixdd:VETDCC:222 mixdd:VETDCC:333" \
      --steps 4000 --jobs 2

CORPORA: vanilla (P9 gen_stream), mixdd (P19 depth-diverse equal-rate),
sched (P20 50%-dyck budget). Each corpus is regenerated
deterministically per process from its certified seed (12345), so a
resumed run is bit-identical to an uninterrupted one (P14 precedent).
DURABILITY: every finished run lands in runs/<run_id>.json; `drive`
only reruns missing run_ids, so a wipe/hibernate costs the unfinished
runs, never the whole sweep.
"""
import argparse, json, os, subprocess, sys, time

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(REPO, "runs")


def _load():
    sys.path.insert(0, REPO)
    import arch_vet_p19 as p19
    from arch_vet_p13d import VETDCC
    return p19, VETDCC, p19.STACKDCC2_D12


def build_corpus(kind, size, dyck_share):
    p19, _, _ = _load()
    if kind == "vanilla":
        return p19.make_pool(size, 256, 12345)
    if kind == "mixdd":
        return p19.gen_mixdd_pool(size, 256, 12345)
    if kind == "sched":
        import arch_vet_p20 as p20
        return p20.gen_sched_pool(size, 256, 12345, dyck_share)
    raise SystemExit(f"unknown corpus {kind}")


def build_arm(name, p19, VETDCC, STACKDCC2_D12):
    if name == "VETDCC":
        return lambda: VETDCC(p19.V, 24, k=8, K=8)
    if name == "STACKDCC2":
        return lambda: STACKDCC2_D12(p19.V, 24, k=8, K=8)
    raise SystemExit(f"unknown arm {name}")


def run_one(corpus, arm, seed, steps, size, dyck_share):
    import torch
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    p19, VETDCC, STACKDCC2_D12 = _load()
    ctor = build_arm(arm, p19, VETDCC, STACKDCC2_D12)
    pool = build_corpus(corpus, size, dyck_share)
    torch.manual_seed(seed)               # SEED HYGIENE LAW
    m = ctor()
    t0 = time.time()
    hist = p19.train_arm(f"{arm}-{corpus}-s{seed}", m, pool, steps, 8)
    row = p19.eval_full(m, f"{arm}-{corpus}", seed)
    row["loss_curve"] = hist
    row["dyck_close"] = p19.eval_dyck(m, f"{arm}-{corpus}", seed)
    row.update({"corpus": corpus, "arm": arm, "seed": seed,
                "steps": steps, "params": p19.n_params(m),
                "wall_s": round(time.time() - t0, 1)})
    return row


def run_id(corpus, arm, seed, steps):
    return f"{corpus}_{arm}_s{seed}_n{steps}"


def cmd_run(a):
    os.makedirs(RUNS, exist_ok=True)
    rid = run_id(a.corpus, a.arm, a.seed, a.steps)
    row = run_one(a.corpus, a.arm, a.seed, a.steps, a.pool,
                  a.dyck_share)
    with open(os.path.join(RUNS, rid + ".json"), "w") as fh:
        json.dump(row, fh)
    print(f"[runner] {rid} done in {row['wall_s']}s", flush=True)
    return 0


def cmd_drive(a):
    os.makedirs(RUNS, exist_ok=True)
    specs = [s.split(":") for s in a.specs.split()]
    for c, m, s in specs:
        rid = run_id(c, m, int(s), a.steps)
        p = os.path.join(RUNS, rid + ".json")
        print(f"[drive] {rid}: "
              f"{'cached' if os.path.exists(p) else 'MISSING -> run'}",
              flush=True)
    t0 = time.time()
    pending = []
    for c, m, s in specs:
        rid = run_id(c, m, int(s), a.steps)
        if not os.path.exists(os.path.join(RUNS, rid + ".json")):
            pending.append((c, m, int(s), rid))
    jobs = max(1, a.jobs)
    running = []
    while pending or running:
        while pending and len(running) < jobs:
            c, m, s, rid = pending.pop(0)
            log = open(os.path.join(RUNS, rid + ".log"), "w")
            cmd = [sys.executable, "-u", os.path.abspath(__file__),
                   "run", "--corpus", c, "--arm", m,
                   "--seed", str(s), "--steps", str(a.steps),
                   "--pool", str(a.pool), "--dyck-share",
                   str(a.dyck_share)]
            env = dict(os.environ, OMP_NUM_THREADS="1")
            p = subprocess.Popen(cmd, cwd=REPO, stdout=log,
                                 stderr=subprocess.STDOUT, env=env)
            running.append((p, rid, log))
            print(f"[drive] launched {rid} (jobs={len(running)}/{jobs})",
                  flush=True)
        time.sleep(5)
        for item in list(running):
            p, rid, log = item
            if p.poll() is not None:
                log.close()
                ok = os.path.exists(os.path.join(RUNS, rid + ".json"))
                print(f"[drive] {rid} exit={p.returncode} "
                      f"result={'OK' if ok else 'MISSING'}", flush=True)
                running.remove(item)
    rows = [json.load(open(os.path.join(
        RUNS, run_id(c, m, int(s), a.steps) + ".json")))
        for c, m, s in specs]
    # merge: group arms, summarise basins
    arms = {}
    for r in rows:
        key = f"{r['arm']}-{r['corpus']}"
        arms.setdefault(key, []).append(r)
    result = {"tag": a.tag, "protocol": a.protocol or "",
              "runner": "arch_vet_runner parallel harness "
                        "(2-concurrent opt measured)",
              "arms": {}}
    for key, rs in arms.items():
        rs.sort(key=lambda r: r["seed"])
        d12 = [r["dyck_close"]["close_d12"] for r in rs]
        pairs = [r["acc_eval_interval"]["pair"] for r in rs]
        ratios = [r["len_ratio_1024_over_256hard"] for r in rs]
        mods = [r["acc_eval_interval"]["modk"] for r in rs]
        n = len(rs)
        summary = {
            "dyck_close_d12_dist": d12,
            "basin_dyck_d12_ge_85": sum(1 for x in d12 if x >= 0.85) / n,
            "pair_eval_dist": pairs,
            "basin_pair_ge_717": sum(1 for x in pairs if x >= 0.717) / n,
            "len_ratio_dist": ratios,
            "basin_ratio_le_6": sum(1 for x in ratios if x <= 0.6) / n,
            "modk_eval_dist": mods,
            "basin_modk_eq_1": sum(1 for x in mods if x == 1.0) / n,
            "n_seeds": n,
            "dyck_close_d12_mean": round(sum(d12) / n, 4),
            "pair_eval_mean": round(sum(pairs) / n, 4),
            "len_ratio_mean": round(sum(ratios) / n, 4)}
        result["arms"][key] = {
            "per_seed": {str(r["seed"]): r for r in rs},
            "summary": summary}
        print(f"[drive] {key} SUMMARY {json.dumps(summary)}",
              flush=True)
    result["wall_s"] = round(time.time() - t0, 1)
    print("RESULT " + json.dumps(result), flush=True)
    with open(os.path.join(REPO, "log.jsonl"), "a") as fh:
        fh.write(json.dumps(result) + "\n")
    print("DONE", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for f in ("corpus", "arm"):
        r.add_argument("--" + f, required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--steps", type=int, default=4000)
    r.add_argument("--pool", type=int, default=512)
    r.add_argument("--dyck-share", dest="dyck_share", type=float,
                   default=0.5)
    r.set_defaults(func=cmd_run)
    d = sub.add_parser("drive")
    d.add_argument("--tag", required=True)
    d.add_argument("--specs", required=True,
                   help="space-separated corpus:arm:seed specs")
    d.add_argument("--steps", type=int, default=4000)
    d.add_argument("--pool", type=int, default=512)
    d.add_argument("--dyck-share", dest="dyck_share", type=float,
                   default=0.5)
    d.add_argument("--jobs", type=int, default=2)
    d.add_argument("--protocol", default="")
    d.set_defaults(func=cmd_drive)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
