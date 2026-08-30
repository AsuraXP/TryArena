# DIRECT PROMPT — short directive for a new agent (paste this)

You are taking over the ARC-2 program (repo AsuraXP/TryArena, branch
arena/01a038ad-tryarena). One goal: **build a token-prediction
architecture that beats the Transformer on reasoning and
generalization — and prove it with micro-scale experiments.**

DO THIS NOW, IN ORDER:
1. Read `HANDOVER_PROMPT.md` in the repo root — it is the ONLY
   authoritative handover. Ignore every queue/plan inside RESUME.md,
   HANDOVER.md, or old log.md blocks (C42-era, STALE, bannered).
   Specifically: do NOT start any "RoPE/SSM hybrid" or "cycle 43"
   work — those lines are dead.
2. `python3 verify_suite.py` → must be 35/35 (if torch missing:
   `pip3 install --break-system-packages torch numpy`, PyPI only).
3. `git fetch origin arena/01a038ad-tryarena` → reconcile per the GIT
   DISCIPLINE section of HANDOVER_PROMPT.md (re-clone recovery
   included; expect non-FF; never force-push).
4. Execute the CYCLE 54 PLAN in HANDOVER_PROMPT.md, in order:
   land P5/P6 (RESULT-tag is the only completion marker; runs are
   deterministic, kill stray `pgrep -af arch_vet` first,
   OMP_NUM_THREADS=1), then P8 VETCAM (the LIFO init-fragility fix),
   P9 VETDCC (the Dyck-3/4 attack), P7 DIV frontier.
5. Every cycle: log.md + log.jsonl + PROBLEM_MAP + 35/35 + commit +
   push. New mechanism → internet search + cite in header first.

THE METRIC (judge every experiment by this): does the VET-lineage
architecture beat the micro-Transformer / Mamba control on
LENGTH INVARIANCE and OUT-OF-TRAIN-INTERVAL accuracy at matched
params — and does it do so ROBUSTLY (multiple inits)? Negative
results must be logged as such (honesty clause). NEVER STOP: when an
experiment lands, immediately start the next in the plan; when the
plan is exhausted, design the next attack on the six open problems
listed there (basin fragility, Dyck, associative capacity, in-range
parity, fluency+reasoning fusion, certificates).
