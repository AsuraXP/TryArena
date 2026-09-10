"""ARCH-VET P25 (cycle 62/63) — 10-SEED CERTIFICATION of the unified
learned-gate system (third-party reproducibility condition).

Seeds 111,222 (P22) + 333 (experts exist) + 444..1010 (7 new).
Stage 1: train missing experts A/B via `arch_vet_p21.py train` in two
concurrent subprocesses (the box has 2 cores; 3 = OOM/thrash).
Stage 2: per seed, train the P22 gate (1500 steps) on frozen experts,
evaluate the unified row (reuses p22.evaluate verbatim), write
runs/p25_s<seed>.json (durable per-seed), skip seeds already done.
Stage 3: aggregate: per-bar basin rate with Wilson 95% CI, mean+-sd of
each metric, and the all-4-bars joint rate. RESULT ARCH-VET-LM-P25.
Bars: pair >=.717, modk = 1.0, CE ratio 1024/256hard <=.6,
dyck close d12 >=.85 (unchanged since P16-P21).
Usage: python3 arch_vet_p25.py [--seeds ...] [--jobs 2]
"""
import argparse, json, math, os, random, subprocess, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = os.path.dirname(os.path.abspath(__file__)); os.chdir(REPO)
CKPT = os.path.join(REPO, "p21_ckpt"); RUNS = os.path.join(REPO, "runs")
os.makedirs(RUNS, exist_ok=True)
DEFAULT_SEEDS = "111,222,333,444,555,666,777,888,999,1010"


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 3), round(c + h, 3))


def stage1(seeds, jobs):
    todo = [(e, s) for s in seeds for e in ("A", "B")
            if not os.path.exists(os.path.join(CKPT, f"{e}_s{s}.pt"))]
    print(f"[p25 stage1] {len(todo)} expert trainings pending: {todo}", flush=True)
    running = []
    env = dict(os.environ, OMP_NUM_THREADS="1")
    while todo or running:
        while todo and len(running) < jobs:
            e, s = todo.pop(0)
            log = open(os.path.join(RUNS, f"p25_train_{e}_s{s}.log"), "w")
            p = subprocess.Popen([sys.executable, "-u", "arch_vet_p21.py", "train",
                                  "--expert", e, "--seed", str(s)],
                                 stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((p, e, s, log, time.time()))
            print(f"[p25 stage1] launched {e} s{s}", flush=True)
        time.sleep(10)
        still = []
        for p, e, s, log, t0 in running:
            if p.poll() is None: still.append((p, e, s, log, t0)); continue
            log.close()
            print(f"[p25 stage1] {e} s{s} exit {p.returncode} in {time.time()-t0:.0f}s", flush=True)
            if p.returncode != 0: todo.append((e, s))   # retry once-ish
        running = still


def stage2(seeds, gate_steps):
    import torch
    torch.set_num_threads(1)
    import arch_vet_p21 as p21, arch_vet_p21c as p21c, arch_vet_p22 as p22
    structural = p21c.make_deep_router(3)
    rows = {}
    for seed in seeds:
        out = os.path.join(RUNS, f"p25_s{seed}.json")
        if os.path.exists(out):
            rows[seed] = json.load(open(out)); print(f"[p25 stage2] s{seed} cached", flush=True); continue
        torch.manual_seed(seed)
        mA = p21.VETDCC(p21.V, 24, k=8, K=8); mB = p21.STACKDCC2_D12(p21.V, 24, k=8, K=8)
        mA.load_state_dict(torch.load(os.path.join(CKPT, f"A_s{seed}.pt"))["sd"])
        mB.load_state_dict(torch.load(os.path.join(CKPT, f"B_s{seed}.pt"))["sd"])
        mA.eval(); mB.eval()
        for m in (mA, mB):
            for q in m.parameters(): q.requires_grad_(False)
        gp = os.path.join(CKPT, f"G_s{seed}.pt")
        if os.path.exists(gp):
            g = p22.Gate(p21.V, 8); g.load_state_dict(torch.load(gp)["sd"]); g.eval()
        else:
            g = p22.train_gate(mA, mB, p22.joint_pool(12345 + seed), seed, gate_steps)
            torch.save({"sd": g.state_dict(), "seed": seed}, gp)
        row = p22.evaluate(mA, mB, g, structural, "learned", seed)
        p21.causal_route = p21c.make_deep_router(3)
        ae = row["task_acc_hard"]
        row["bars"] = {"pair": ae["pair"] >= .717, "modk": ae["modk"] >= 1.0,
                       "ratio": row["len_ratio_1024_over_256hard"] <= .6,
                       "dyck_d12": row["dyck_close_routed"]["close_d12"] >= .85}
        row["n_bars"] = sum(row["bars"].values())
        json.dump(row, open(out, "w"))
        print(f"[p25 stage2] s{seed} bars={row['n_bars']} {row['bars']}", flush=True)
        rows[seed] = row
    return rows


def stage3(rows):
    n = len(rows); s = sorted(rows)
    def dist(f): return [round(f(rows[k]), 4) for k in s]
    metrics = {"pair": dist(lambda r: r["task_acc_hard"]["pair"]),
               "modk": dist(lambda r: r["task_acc_hard"]["modk"]),
               "ratio": dist(lambda r: r["len_ratio_1024_over_256hard"]),
               "dyck_d12": dist(lambda r: r["dyck_close_routed"]["close_d12"]),
               "joint_pair": dist(lambda r: r["joint_mixdd"]["pair"]),
               "joint_dyck": dist(lambda r: r["joint_mixdd"]["dyck_close"]),
               "track_hard": dist(lambda r: r["task_acc_hard"]["track"])}
    def msd(v):
        m = sum(v) / len(v); sd = (sum((x - m) ** 2 for x in v) / max(1, len(v) - 1)) ** .5
        return {"mean": round(m, 4), "sd": round(sd, 4), "min": min(v), "max": max(v)}
    bars = {}
    for b in ("pair", "modk", "ratio", "dyck_d12"):
        k = sum(rows[x]["bars"][b] for x in s)
        bars[b] = {"k": k, "n": n, "rate": round(k / n, 3), "wilson95": wilson(k, n)}
    k4 = sum(rows[x]["n_bars"] == 4 for x in s)
    summ = {"n_seeds": n, "seeds": s, "metrics": metrics,
            "metric_stats": {k: msd(v) for k, v in metrics.items()},
            "bar_basins": bars,
            "all4": {"k": k4, "n": n, "rate": round(k4 / n, 3), "wilson95": wilson(k4, n)},
            "bars_passed_per_seed": [rows[x]["n_bars"] for x in s]}
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=DEFAULT_SEEDS)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--gate_steps", type=int, default=1500)
    a = ap.parse_args(); t0 = time.time()
    seeds = [int(x) for x in a.seeds.split(",")]
    stage1(seeds, a.jobs)
    rows = stage2(seeds, a.gate_steps)
    summ = stage3(rows)
    out = {"tag": "ARCH-VET-LM-P25",
           "protocol": "10-seed certification of the unified learned-gate system "
                       "(VETDCC-big A vanilla@4000 + STACKDCC2-big D12 B deepmix@2000 + "
                       "P22 GRU gate 1500 steps; hard argmax dispatch; bars pair>=.717, "
                       "modk=1, ratio<=.6, dyck d12>=.85; Wilson 95% CIs)",
           "summary": summ, "per_seed": {str(k): v for k, v in rows.items()},
           "wall_s": round(time.time() - t0)}
    print("[P25] SUMMARY", json.dumps(summ), flush=True)
    with open("log.jsonl", "a") as f: f.write(json.dumps(out) + "\n")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
