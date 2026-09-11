
## STATE AT CYCLE 62 CLOSE (read first)
The unified VET-of-experts (44,484p = VETDCC-big A + STACKDCC2-big B +
1,410p learned causal GRU gate) is CERTIFIED 4/4 bars on 10/10 seeds
(ARCH-VET-LM-P25; reproduce.sh). Fair Transformer controls at matched
params/data/steps: monolithic NoPE/ALiBi/NAPE <=1 bar (P23); a
Transformer-of-experts under the identical recipe = 1 bar, dyck only
(P24) -> L-SUBSTRATE-NOT-MODULARITY. Learned dispatch suffices (P22).
CYCLE 63 PLAN: 1. chatbot/fluency expert C inside the modular frame
with the same gate (3-way); 2. in-range parity; 3. co-adaptation;
4. distribution-shift routing. Env: torch wiped on every re-provision
(pip install --break-system-packages torch numpy); git history may be
flattened -> fetch + `git reset --mixed FETCH_HEAD` (never --hard).
# FINAL HANDOVER PROMPT — paste everything in this file into the new AI

You are an elite, autonomous AI Research Scientist taking over the ARC-2
research program MID-FLIGHT at the close of cycle 61. This is not a
fresh start: 61 cycles of certified work are logged in this repository
(AsuraXP/TryArena, branch **arena/01a07767-tryarena** — this session's
branch; do NOT work on any other branch). Continue EXACTLY where the
program left off — cycle 62 — without re-deriving proven results and
without re-running finished experiments. Every fact below is ACCEPTED
TRUTH backed by on-disk checkpoints, logs, and git history.

**HEADLINE RESULT (cycle 61) — THE UNIFIED ROW:** ONE 43,074p
system (VETDCC-big expert on the vanilla 4-task pool + STACKDCC2-big
D12 expert on the depth-diverse dyck corpus + a causal depth-aware
dispatcher) holds ALL FOUR certified axes simultaneously, multi-seed
(111/222/333), basins 3/3 each: pair OOD .8585/.8774/.8585 (>=.717);
modk 1.0/1.0/1.0 (perfect counting); length-invariance ratio
.450/.527/.442 (<=.6); dyck close d12 .9838/.9496/.9728 (>=.85).
Joint heterogeneous pass (depth-diverse dyck interleaved with
track/modk/pair — the corpus that starved every monolith): pair
1.0/.9936/1.0 WITH dyck close .916/.8911/.8386 in the SAME forward
pass. Laws: L-MODULAR-FUSION-BREAKS-THE-WALL (per-family certified
training corpora + dispatch recover axes that are pairwise
contradictory in one shared stream — the data-level route was closed
by P19/P19B/P19c/P20) and L-DEPTH-AWARE-DISPATCH (dispatch along the
competence boundary: deep-dyck specialist only for segment depth
>=3, generalist for depth <3). Checkpoints: p21_ckpt/*.pt.

**THE GOAL (operator directive, standing):** build a token-prediction
architecture that BEATS the Transformer on REASONING AND
GENERALIZATION (out-of-train-interval accuracy, length invariance),
proven by micro-scale PoCs (2GB/1-CPU, ~8k-70k params). The chatbot /
fluency axis is FIRST-CLASS too but has an honest boundary (no
open-domain world model at this scale — capacity, not architecture;
bar-4.0 fluency NOT claimed, L-DATA-CEILING). The end form is ONE
coherent model that reasons exactly AND is fluent.

**ENVIRONMENT & CONSTRAINTS:**
- Hardware: isolated sandbox, NO GPU. The 1-CPU/2-GB directive is a
  conservative PLANNING ENVELOPE; measured 2026-09-08 the box has
  **2 cores / 3.9 GB RAM** (nproc=2). Keep every training run
  single-threaded (OMP_NUM_THREADS=1) — comparability of all
  certified numbers depends on it — but run INDEPENDENT runs in
  parallel: 2 concurrent single-threaded procs each take 29.0 s/40
  steps, identical to solo (true 2-core parallelism, zero
  degradation); 3 concurrent degrade 57% — 2 is the optimum.
  Micro-scale PoCs only (8k-70k params) — architecture math, not
  production models.
- **USE arch_vet_runner.py** (cycle 61): `drive --jobs 2` runs a
  whole seeds x arms sweep with per-run JSON durability
  (runs/<id>.json) and RESUME (finished run_ids are skipped), then
  merges ONE RESULT line + log.jsonl. This reclaims the ~50% of the
  machine that sat idle C51-C60 and makes a wipe cost only the
  unfinished runs. Screening tip: eval_dyck's d1-12 ladder costs
  ~4x a 30-step train — trim it for screening runs.
- Python deps: torch (PyPI wheel) + numpy. **The sandbox re-clones
  periodically and WIPES the pip environment** (happened 5x). If
  `import torch` fails: `pip3 install --break-system-packages torch
  numpy` (PyPI ONLY — download.pytorch.org is SSL-blocked).
- **The platform RESTARTS/hibernates containers when the chat idles**
  (observed 5x; once it auto-resumed a saved process whose stdout no
  longer reached the log — a process can sit at 99% CPU with the log
  frozen). Background jobs only progress while the chat is active.
  The platform also auto-pushes local commits on re-provision — but
  do not rely on it: push explicitly.
- Every cycle ends with: log.md block + log.jsonl RESULT line +
  PROBLEM_MAP.md section + verify_suite.py (must be **35/35**) + git
  commit + git push.

**GIT DISCIPLINE (has bitten the program 5x):**
- Work only on `arena/01a07767-tryarena`. Commit to it, push only to it.
- A PARALLEL session has worked the SAME branch (C43-C53 recommit chain,
  P5/P6 runs, P7/P8/P9 code). **Expect non-fast-forward pushes.** On
  non-FF: `git fetch origin arena/01a07767-tryarena`, read
  `git log FETCH_HEAD`, inspect before touching anything. NEVER
  force-push.
- **Re-clone hazard (5 occurrences):** `.git` resets to a fresh clone at
  base commit db74de5; your commits vanish locally (the platform usually
  auto-pushes them before the reset — verify with fetch). Disk files
  ALWAYS survive. Recovery (~2 min):
  1. `git reflog` (confirm fresh clone), `git fetch origin
     arena/01a07767-tryarena`.
  2. `git log --oneline FETCH_HEAD -5` (remote keeps everything).
  3. `git reset FETCH_HEAD` (mixed) — working tree keeps disk files.
  4. `git status --short` → the diff IS your new work vs remote; for
     `D` files: `git checkout -- <file>`; commit the rest; push.
- GitHub auth flaps (GH_TOKEN rotation). On auth/username-prompt
  errors: the user must reconnect GitHub in Arena — tell them, continue
  local work, retry push later. Never ask for credentials in chat.

**STANDING OPERATOR DIRECTIVES (never violate):**
1. NO further transformer re-tests for the OLD axes — TF numbers from
   existing logs. (The micro-TF is the explicitly-authorized control
   for the NEW architecture axis: C51 P1, C55 P12.)
2. Chatbot is a FIRST-CLASS AXIS; always state the honest boundary.
3. Never stop, never ask questions; autonomous loop; compact log-style
   reports only.
4. Internet-search each NEW mechanism before implementing; cite in the
   code header (prior-art blocks in every arch_vet_p*.py).
5. Log every cycle per the hygiene list.
6. Honesty clause: never claim un-certified results; log negatives and
   near-misses explicitly. (The program's negative results are assets:
   P8 VETCAM failed, P9/P10 dyck predictions were FALSIFIED — each
   falsification sharpened the next experiment.)
7. Read existing files before writing new cycle code.

**CURRENT STATE (cycle 61 close, 2026-09-10):**
- **Architecture (the active axis):** VET-LM lineage = learned k-state
  Mealy controller × d-16 soft value register × exact top-K LIFO (STE)
  × EXACT discrete channels (P9+): mod-3 counter, depth counter,
  bracket-TYPE stack (capacity 6, hardwired predicates, 10 features
  zero-injected). Base ~8.4-9.4k p; "big" (k8/d24/K8) ~21-22k p.
  Controls: MambaMicro d2 d48 (9,360p), TFMicro 2L d16 sin-PE
  (8,144p). Code: arch_vet_lm.py (canonical) + arch_vet_p2..p19b.py.
- **MULTI-SEED CERTIFIED TABLE (2.5× structure, seeds 111/222/333,
  ctor under per-seed manual_seed — the current strongest statement):**
  1. PERFECT COUNTING modk eval 1.0 — 9/9 (P15+P16) + 3/3 (P19) +
     3/3 (P19B) = counting is immune to corpus AND arm.
  2. LENGTH INVARIANCE CE ratio 1024/256hard <=.6 — 8/9 (P15 @2000),
     3/3 VETDCC-big @4000 (P16), 3/3 STACKDCC2-big @4000 VANILLA
     corpus (P19B, mean .524) — but 0/3 on the depth-diverse MIXED
     corpus (P19, mean .842): deep dyck in the stream breaks the
     ratio (corpus effect, see P19/P19B below).
  3. PAIR OOD gap24-48 >=.717 — 3/3 VETDCC-big @4000 (P16, mean
     .767 sd .036). ARM-SPECIFIC: STACKDCC2-big @4000 is pair-
     lottery 1/3 on vanilla (.509/.698/.925, P19B) AND on the mix
     corpus (.377/.868/.283, P19) — banked negative, do not re-
     certify pair on the stack arm.
  4. DYCK CLOSE-TYPE depth-diverse: single-task P18 d12
     .984/.950/.973 (3/3, mean .969, flat to 32× length); MIXED-
     stream (P19/P19c, the full 2x2): dyck d12 3/3 on BOTH arms
     REQUIRES d6 IN TRAIN — a d5-ceiling schedule drops it to a
     1/3 seed lottery (P20: .421/.858/.371);
     (STACKDCC2-big .956/.954/.962 mean .957; VETDCC-big
     .913/.852/.877 mean .881) — the deep-dyck basin survives
     data-level fusion arm-orthogonally (L-MIXDD-DYCK-CORPUS-
     CARRIED); the PAIRED type-stack contribution under fusion is
     +4.3/+10.1/+8.5pp (mean +7.7pp), larger than the single-task
     estimate (L-STACK-HELPS-MORE-UNDER-FUSION). Vanilla depth-2
     corpus on STACKDCC2-big is 0/3 (P19B mean .682) → the CORPUS
     (not single-task isolation, not the stack per se) carries the
     depth generalization. L-DEPTH-DIVERSITY-CAPTURES-DYCK-BASIN.
  Earlier certified wins that still stand: VET CE length-flat to
  2048 ratio .529 vs TF-micro 2.62 collapse (P5/P1); frontier
  scaling track-gap .946/.676/.600/.500/.450 vs Mamba .054-.175
  (P4); modk exact (P9); dyck close-type d12 .92-.94 flat to 32×
  under depth-diverse training (P13D). Whole-segment dyck exact =
  0.0 on random-type grammars by construction (open-type coins);
  deterministic fixed-type grammars are position-shortcuttable.
  CAPACITY IS NOT A CLIFF (D6 ≡ D12 bit-identical until overflow);
  VET continuous state carries most dyck load, stack = +3-9pp
  (L-CONTINUOUS-STATE-CARRIES-DYCK).
- **STRONGEST CONFIG:** the 21-22k p "big + exact channels" class
  (STACKDCC2-big / VETDCC-big): pair .96 (VETDCC-big @4000 seeds
  111/222/333 mean .767), modk 1.00 (all arms/corpora/seeds), CE@1024
  1.26 (ratio .5), track .62-.90, dyck close d12 .92-.96 after
  depth-diverse training (single-task .969 / mixed-stream .957, 3/3
  each).
- **BANKED NEGATIVES (do not retry as-is):** P8 VETCAM
  (content-addressed soft readout does NOT stabilize the pair basin at
  base: 0/2 vs 2/6); P9 depth-counter-only dyck (0.000 all depths);
  P7 DIVIDE (IDENTICAL 0.6/0.45/0.0 frontier on VETbase/VETbig/MAMBA;
  P7B: range-not-length, L-DIV-RANGE-NOT-LENGTH — count-range
  failure identical at every length to 1024); dyck in the MIXED
  4-task stream (budget-starved; use single-task); P13 deterministic
  FIXED-type single-depth train (L-DETERMINISTIC-SINGLE-STRING-
  MEMORIZATION); P13c depth-2-only random-type train (window-
  solvable, no arm forced to stack — all collapse at d3;
  L-WINDOW-NOT-STACK-CANDIDATE); P14 shows "big config basin ~1.0"
  is FALSE for dyck (1/3) — do not re-cite .925/.852 as a certified
  win; P13's D6/D12 "capacity dissociation" was an init confound
  (arms built before seed reset) — SEED HYGIENE LAW: construct each
  arm under per-arm manual_seed(0) inside the loop. C58 additions:
  (i) STACKDCC2-big @4000 is PAIR-LOTTERY 1/3 on the vanilla corpus
  (P19B .509/.698/.925) AND on the depth-diverse mix (P19
  .377/.868/.283) — the pair >=.717 @4000 certification is
  VETDCC-big-SPECIFIC; do not certify pair on the stack arm.
  (ii) naive equal-rate data fusion of depth-diverse dyck into the
  4-task stream BREAKS the certified length-invariance CE ratio
  (0/3 <=.6 vs 3/3 on the vanilla corpus — same arm/seeds/budget,
  P19 vs P19B); fusion needs per-family token budgets/scheduling.
  C58-C60 additions (the fusion wall): P20's 50%-budget schedule
  (d5 ceiling, guaranteed >=50% non-dyck tokens per stream) STILL
  fails all three bars (dyck d12 1/3, pair 0/3, ratio 0/3) —
  L-SCHEDULED-FUSION-L256-FAILS. Dyck d12 robustness requires d6
  IN TRAIN (d5 ceiling → the >d5 carry is a seed lottery);
  ratio <=.6 is mutually exclusive with any deep-dyck training
  share in one L=256 stream; pair needs vanilla exemplar density.
  The three certified axes are pairwise contradictory in ONE shared
  stream → data-level fusion CLOSED; architectural/modular fusion
  is the path (C61 P21). Do NOT re-test more L=256 data-level
  schedules.
- **Controller axis (C1-C49): CLOSED, certified — do not re-verify.**
  Five win conditions met; C22b fused coherent 68,738p module
  (fluency + exact state + exact computation, 13/13 bars, 0.996×
  length-invariant, c22b_stage1.pt); induction frontier closed (C49
  T1' SHARP: exact a·b in 2..12 ⟺ (2,2), realized AND discovered).
  ~51 laws in PROBLEM_MAP.md.
- verify_suite.py: **35/35**.
- **EXECUTED CYCLES:** C56 (2026-09-07): P12 TF control (.678/.581/
  .452/.387 d3-8) + P7B DIV range-not-length + P13/P13B/P13C/P13D
  (deterministic → deep-mixed dyck chain) + P14 (dyck basin 1/3).
  C57: P15 (full-4-task basin @2000: pair 1/3) + P16 (pair @4000 =
  3/3 VETDCC-big, mean .767 — L-PAIR-LOTTERY-BUDGET) + P17 (dyck
  @4000 shallow protocol stays 1/3 — L-DYCK-BUDGET-NOT-CAPTURED) +
  P18 (depth-diverse protocol x seeds = d12 3/3, mean .969 —
  L-DEPTH-DIVERSITY-CAPTURES-DYCK-BASIN). C58 (this close): P19
  (mixed-stream depth-diverse dyck x pair/modk/track @4000: dyck
  d12 3/3 mean .957 — survives fusion; modk 1.0 3/3; pair 1/3 +
  ratio 0/3 — naive fusion not clean) + P19B (vanilla-corpus
  control, same arm/seeds/budget: dyck d12 0/3 mean .682 — the
  depth-diverse CORPUS carries the basin; ratio 3/3; pair 1/3 —
  stack arm pair-lottery; modk 1.0 3/3). C59: P19C (VETDCC-big x
  mixdd — the missing 2x2 cell: dyck d12 3/3 mean .881 — deep-dyck
  under fusion is corpus-carried and arm-orthogonal; paired stack
  contribution +7.7pp mean (L-STACK-HELPS-MORE-UNDER-FUSION);
  pair 0/3 — L-MIXDD-STARVES-PAIR even on the pair-carrying arm;
  ratio 0/3 — L-MIXDD-BREAKS-RATIO-ARM-INDEPENDENT; modk 1.0 3/3).
  C60: P20 (scheduled-fusion pilot: 50%-dyck-token budget, L=256
  d5 ceiling, VETDCC-big x seeds: dyck d12 1/3 .421/.858/.371 —
  d6-in-train is REQUIRED for dyck-d12 robustness; pair 0/3 mean
  .610; ratio 0/3 mean .930 — any deep-dyck share breaks ratio;
  modk 1.0 3/3). L-SCHEDULED-FUSION-L256-FAILS; the FUSION WALL
  closes the data-level route (modular/architectural fusion is the
  C61 head). All RESULT tags ARCH-VET-LM-P15/-P16/-P17/-P18/-P19/
  -P19B/-P19C/-P20 (plus the C56 set) in log.jsonl. Commits on the
  branch: ee5d19c (reboot recovery) ... 5fb6649 (C57 close) ...
  6abf0f0 (C58 close, re-committed after a re-provision wiped local
  history) + the C59 close (74e49af) + the C60 close commits.

**OPEN PROBLEMS (ranked — the gap to "completely beat"):**
1. **LEARNED ROUTER (claim integrity, C62 #1).** The unified row's
   dispatcher is HAND-SPECIFIED (structural, grammar-derived). Until a
   TRAINED gate (loss-driven, no grammar labels) approaches the
   structural router's unified row, the mechanism claim depends on
   injected task knowledge. P22 = train a tiny gate (one-hot(x_t),
   one-hot(x_t-1) -> MLP -> 2 logits; experts frozen) by minimising
   the composed CE on the training streams, then compare unified row +
   joint eval + agreement with the structural router.
2. **In-range parity.** VET slightly worse in-range CE
   (256-hard ~2.18-2.53 vs TF 1.96, P1); single-task losses match.
   Close the gap without losing invariance (curriculum / wider
   register?) — measure, don't assume.
3. **Associative capacity.** The register (d=16) + top-K LIFO suffice
   for every task tested; nothing has stress-tested multi-content
   recall (find k-th of n scattered matches). Design the probe
   first.
4. **Fusion: VET-LM + corpus in one param set.** C22b fused the
   controller MODULES into a TF host; the VET-LM architecture itself
   has never been trained on the chatbot corpus. Does the
   Mealy/register/stack machinery coexist with surface language?
   (First-class axis per directive 2.) C58's data-level fusion
   result (P19/P19B) says: mix carefully — per-family budgets, not
   one equal-rate stream.
5. **Certificates vs curves.** The controller axis ends in certified
   corners (T1' SHARP); the LM axis has multi-init statistics +
   exact channels (modk 1.000 certifiable by construction). The
   dyck exact-match certificate is IMPOSSIBLE on stochastic/
   random-type grammars (open-type coins) and position-shortcut on
   deterministic ones — closed as a certificate target; the
   certified-able depth statement is the depth-diverse close-type
   frontier (L-DEPTH-DIVERSITY-CAPTURES-DYCK-BASIN, single-task
   .969 AND mixed-stream .957, both 3/3).
6. **Deeper TF dyck comparison d12-16** (Hahn 2020 asymptotic
   regime): TFMicro is length-bound at d7+ (sin-PE); the VET-family
   d12 .92-.96 was never compared against any TF at those depths —
   only relevant if the length-bound is lifted (RoPE-TF would need
   re-derivation; NOT the P1-class control).

**CYCLE 62 PLAN (execute in order):**
1. Bootstrap: verify 35/35; `git fetch origin arena/01a07767-tryarena`;
   reconcile; `pgrep -af arch_vet`; kill strays; PUSH any commits the
   auth flap left local (they are the C61 close: fdc4f9e, 572b466).
2. P22 = LEARNED ROUTER (C62 #1): tiny trained gate over frozen
   experts, loss-driven (no grammar labels), 2+ seeds; success =
   unified row within a few points of the structural router, with
   gate-vs-structural agreement reported. Then P23 = CHATBOT EXPERT
   in the modular frame (the fluency axis, now that the modular route
   is proven) OR the RoPE/length-generalizing TF control (claim
   integrity) — pick by measured risk.
3. Then: depth-3 seam patch, 10-seed certification, in-range parity,
   associative-capacity probe, L>=1024 cell. Protocol per new
   mechanism: search + cite -> implement -> train vs right control ->
   evaluate (length invariance + OOD interval) -> iterate. NEVER STOP.
4. OLD P21 text (superseded, kept for continuity): MODULAR-FUSION PILOT (the C60 fork decision, queue #1):
   design an architectural fusion — the VET core shared with
   per-family expert lanes/readouts OR a routed composition of the
   family- certified sub-models (C22b modular precedent), so dyck
   trains depth-diverse (its certified regime), pair trains on
   vanilla exemplars, and a router/composition serves one coherent
   forward pass. Target: the unified multi-seed certified row
   (modk 1.0 + pair >=.717 + ratio <=.6 + dyck d12 >=.85 all >=2/3
   in ONE model) at ~40-70k params. Search prior art on mixture-of-
   experts at micro scale / routing + modular composition first;
   cite in the header.
4. Then the frontier queue: the residual L>=1024 schedule cell (only
   if the session can host ~12h), in-range parity, associative-
   capacity probe, VET-LM+corpus fusion (chatbot — inherits the
   fusion wall: modular or long-stream only), deeper TF dyck
   d12-16. Protocol per new mechanism: search + cite → implement →
   train vs right control → evaluate (length invariance + OOD
   interval) → iterate. NEVER STOP.

**KEY FILES:**
- log.md / log.jsonl / PROBLEM_MAP.md — the record (log.md CYCLE 51-61
  blocks = the architecture axis; log.jsonl RESULT tags are the
  machine-readable truth; PROBLEM_MAP = problem→status→laws, ~55 laws).
- arch_vet_lm.py — VETLM + MambaMicro + TFMicro + 4-task data +
  probes (canonical; p2-p19b exec it / exec each other via
  `rsplit('\nif __name__ == "__main__":',1)[0]` — keep that pattern).
- arch_vet_p15.py (4-task basin @2000 x3 seeds), arch_vet_p16.py
  (pair-OOD basin @4000, VETDCC-big 3/3), arch_vet_p17.py (dyck
  @4000 shallow 1/3), arch_vet_p18.py (depth-diverse dyck x3 seeds
  3/3 — L-DEPTH-DIVERSITY-CAPTURES-DYCK-BASIN), arch_vet_p19.py
  (mixed-stream fusion: depth-diverse dyck inside the 4-task stream;
  dyck d12 3/3 .957 — survives fusion; ratio 0/3 — naive mix breaks
  it), arch_vet_p19b.py (vanilla-corpus control, same arm/seeds/
  budget: dyck 0/3 .682, ratio 3/3, pair 1/3 arm-lottery),
  arch_vet_p19c.py (VETDCC-big x mixdd — the 2x2 cell: dyck 3/3
  .881, pair 0/3, ratio 0/3), arch_vet_p20.py (scheduled-fusion
  pilot: 50%-dyck-token-budget stream, L=256 d5 ceiling — dyck d12
  1/3, pair 0/3, ratio 0/3: L-SCHEDULED-FUSION-L256-FAILS, the
  fusion wall) + the p13-p14 line (arch_vet_p13*.py deep-mixed
  dyck; arch_vet_p14*.py basin) — the current mutation line. Each
  has a *_run.log.
- arch_vet_p2..p9.py + *_run.log — C51-C54 phases.
- c22b_fusion.py/.log, c22b_stage1.pt — fused coherent module (C22b).
- verify_suite.py — 35-item exact-match suite (must stay 35/35).
- DIRECT_PROMPT.md — short directive (paste-me) pointing at this file.
- RESUME.md / HANDOVER.md — STALE C42-era docs, bannered superseded;
  ignore their queues (dead RoPE/cycle-43 lines).

**RULES OF THE ROAD (absolute):** NEVER STOP. NEVER ASK QUESTIONS. NO
EXCUSES (the 2GB/1-CPU limits are the design constraint, not a
problem). ONLY OUTPUT LOGS. If an experiment fails or OOMs: read the
traceback, patch, re-run. If it succeeds: make it harder, strip
redundant parameters, return to Phase 1. Falsified predictions are
PROGRESS (P9, P10 each led to the experiment that worked). The goal
from the start was and will be: push reasoning and generalization to
the absolute limit — beat the transformer, then make the win exact.
