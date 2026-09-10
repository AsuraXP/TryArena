"""
ARC-2 CYCLE 16 / CONTEXT-WINDOW PROBE: machine v4 (trained at 63 tokens)
at 8192 and 16384 (128x / 256x training length), all four families.
Claim under test: the O(1)-state machine has no context-window limit —
every family should hold its @4096 dCE at 16384, where every TF flavor
already fails at 4096 (4.8-10.6 dCE; the 103k TF cannot even run at
16384 in this sandbox: O(N^2) attention memory).
Evals (2 reps each): echo (total dCE), icl (total|target), mod7 (total),
add (pooled total) + routing acc. Baseline cited: micro_tf_12k @4096
(unified_add.log: echo 4.78, icl 5.39|10.57, mod7 5.63, add 4.82).
USAGE: OMP_NUM_THREADS=1 python3 -u probe_16k.py
"""
import json, random, resource, time
import torch
import torch.nn.functional as F

torch.set_num_threads(1)
t_start = time.time()

g = {"__name__": "p16"}
exec(open("unified_add.py").read().split("\nRESULTS = {}")[0], g)
MachineV4, n_params, eval_task = g["MachineV4"], g["n_params"], g["eval_task"]
gen_echo_t, gen_icl_t, gen_mod7_t, gen_add_t = (
    g["gen_echo_t"], g["gen_icl_t"], g["gen_mod7_t"], g["gen_add_t"])

@torch.no_grad()
def eval_task_long(model, gen, L, task_id, reps=2, tgt=False):
    model.eval()
    bs = 1
    ce = orc = n = 0.0
    tgt_ce = tgt_n = 0.0
    route_acc = 0.0
    for i in range(reps):
        rng = random.Random(900_000 + L + i)
        x, y, o = gen(bs, L, rng)
        t0 = time.time()
        logits, rl = model(x)
        nll = -F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        ce += nll.sum().item(); orc += o.sum().item(); n += y.numel()
        route_acc += (rl.argmax(-1) == task_id).float().mean().item()
        if tgt:
            tgt_ce += nll[:, -1].sum().item(); tgt_n += nll[:, -1].numel()
        print(f"    L={L} rep {i}: dCE so far {round((ce-orc)/n, 4)} "
              f"({time.time()-t0:.0f}s)", flush=True)
    d = round((ce - orc) / n, 4)
    if tgt:
        d = (d, round(tgt_ce / tgt_n, 4))
    d = (d, route_acc / reps) if not tgt else (d[0], d[1], route_acc / reps)
    return d

m4 = MachineV4()
m4.load_state_dict(torch.load("unified_add_final.pt"))
m4.eval()
print(f"[probe] machine v4 final ckpt, params={n_params(m4)}, trained at L=63", flush=True)
RESULTS = {}
for L in [8192, 16384]:
    r = {"echo": eval_task_long(m4, gen_echo_t, L, 0),
         "icl": eval_task_long(m4, gen_icl_t, L, 1, tgt=True),
         "mod7": eval_task_long(m4, gen_mod7_t, L, 2),
         "add": eval_task_long(m4, gen_add_t, L, 3)}
    RESULTS[f"L{L}"] = r
    print(f"  L{L}: {r}", flush=True)

print("\n" + "=" * 92)
print("CONTEXT-WINDOW PROBE — machine v4 (trained L=63), dCE + routing acc")
print("@4096 ref (unified_add.log): echo -0.3004 | icl 0.1351|0.1972 | mod7 0.0036 | add 0.0091")
print("(transient-free 9x ref: icl 0.0072|0.0021) | TF @4096: echo 4.78 | icl 5.39|10.57 |")
print("mod7 5.63 | add 4.82 — TF cannot run @16384 in 2GB (O(N^2) attention).")
print("=" * 92)
for k, v in RESULTS.items():
    print(f"{k:<6} echo {v['echo']}  icl {v['icl']}  mod7 {v['mod7']}  add {v['add']}", flush=True)
print("=" * 92)
final = {"tag": "ARC2-C16-CTX-WINDOW-PROBE-16K", "runs": RESULTS,
         "wall_s": round(time.time() - t_start, 1),
         "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}
print("RESULT " + json.dumps(final), flush=True)
open("log.jsonl", "a").write(json.dumps(final) + "\n")
print("DONE", flush=True)
