# HANDOVER PROMPT — paste this into the new conversation

You are resuming the ARC-2 research program — an autonomous machine-building
project that has been running for 38 cycles in this repository. Work inside
the checkout of AsuraXP/TryArena on branch arena/01a02c9d-tryarena. Do all
work on that branch only; commit and push to it each cycle.

## 0. FIRST ACTIONS (in order, before anything else)
1. Read HANDOVER.md (the canonical state index; updated cycle 38).
2. Read the last 2–3 blocks of log.md (most recent cycle detail) and the
   C22/C22-R sections of PROBLEM_MAP.md, plus SCOREBOARD.md.
3. Run verify_suite.py (OMP_NUM_THREADS=1 python3 verify_suite.py). It MUST
   be 35/35. If torch is missing, the VM was reset — reinstall with:
   pip3 install --break-system-packages torch numpy
   If git state looks rolled back, recover with:
   git fetch origin arena/01a02c9d-tryarena && git reset --hard FETCH_HEAD
   (working-tree files usually survive resets; rescue modified tracked files
   to /tmp before any hard reset).
4. State in one line: verify result, current queue head, and what you will
   do this cycle. Then do it.

## 1. WHAT THE PROJECT IS
Beat the transformer architecturally (never by budget) on a constrained box:
2 CPU cores, ~4GB RAM, 1-thread torch. We build heterogeneous machines:
shared embeddings + learned per-example router over ISOLATED branches; each
branch = small linear SSM host + an EXACT MEMORY ORGAN (learned readout over
mechanism-computed state features). Capabilities are certified against
analytic oracles (dCE = CE minus exact per-token entropy), probed at
4096/16384 (256x train length), frozen verify suite 35/35.

Certified so far: machine v6/v7 (5 task families), P4-DISC open-ended
program discovery (c24k), C25a iterated subtraction, and — as of cycle 38 —
C22 chatbot machine v9c (c22r8.pt): all 7 bars D1–D7 (state recall,
overwrite, 16k length, math +/-, chat, routing 1.0, exact dialogue).

## 2. CURRENT STATE (cycle 38 close)
- HEAD pushed on arena/01a02c9d-tryarena (cycle 38 commit: C22-R CERTIFIED).
- ACTIVE QUEUE HEAD: C29 new-machinery results (next slot). Then C22b
  fluency-into-chatbot fusion, C23 router hardening, C21b d64 fluency host.
- C26 variable binding is PLATEAU-BLOCKED v4 (30+ runs, four repair methods
  converged on the same attractor — do NOT retry row/cell-level repair;
  re-enter ONLY with fundamentally new attack: new tape geometry / larger
  state space / value-encoded transport).
- Key checkpoints: c22r8.pt (chatbot champion), c24k_crispfix.pt (P4-DISC),
  c25a_sub.py/.pt (subtraction), c33_merge.pt (binding plateau champion
  326/500), c37_single.pt (binding single-path 154/500).

## 3. STANDING OPERATOR DIRECTIVES (do not violate)
1. NO transformer baselines/comparisons — reaffirmed three times. Cite old
   TF numbers from logs only; never re-run TF arms.
2. Chatbot is a FIRST-CLASS axis; open-ended P4 discovery is THE main track.
3. Never stop, never ask questions in autonomous mode; compact log-style
   reports only. State win/fail VERDICTS explicitly up front.
4. Internet-search each new mechanism before implementing (log the cite).
5. Every cycle: log.md block + log.jsonl RESULT line + PROBLEM_MAP.md update
   + verify_suite 35/35 + git commit + push (push auth flaps — retry each
   cycle; never ask the user for credentials).
6. Honesty clause: never claim an un-certified result; log negatives and
   near-misses; ambiguous judge inputs get "?" not guesses.
7. Long runs: state wall-clock estimate BEFORE starting; hard bound every
   experiment (≤ ~45 min); max TWO 1-thread processes at once.
8. Don't re-verify things already verified (no wasted time).

## 4. PROTOCOL CONSTANTS
- dCE vs analytic oracle; probes at 4096/16384; train windows 63; AdamW
  3e-3 typical; seeds logged in every RESULT line of log.jsonl.
- Laws banked in log.md/HANDOVER §5 (21+ laws incl. the cycle-38 set:
  L-ORACLE-COMPLETE, L-EMIT-TIMING, L-DECAY-DRIFT, L-TRAIN-LENGTH-MISMATCH,
  L-ORGAN-GAIN). Read them before designing new machinery — they are paid
  for in compute.

## 5. ENVIRONMENT PITFALLS
- The sandbox VM resets occasionally: pip torch is wiped AND git rolls back
  to the last pushed commit. Recovery recipe is in §0.3. Check ps before
  relaunching anything; children can survive.
- Many experiment scripts are monolithic with module-level run sections
  (dialog_chat.py etc. have NO __main__ guard) — reuse them by exec-ing the
  source up to a marker (pattern in c22r.py/c22r2.py), never import them.
- Probe generators take INT ops (mod.PLUS/MINUS), not strings.
- Everything runs single-thread (OMP_NUM_THREADS=1) and must fit ~4GB.

## 6. AUTONOMOUS MODE (when I paste the research-scientist block)
Continuous execution: pick the queue head, research → design → implement →
debug yourself and rerun → certify against bars → log/commit/push → next.
No questions mid-loop; technical log-only output; if a campaign plateaus
after genuinely distinct attacks, grade it BLOCKED honestly, bank the laws,
and advance the queue.

Begin now with the FIRST ACTIONS.
