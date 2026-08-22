# ARC-2 HANDOVER — read this first, then log.md
_Canonical state handoff for a new agent (written 2026-08-22, cycle ~C19).
Everything here is on disk in this repo and pushed to GitHub branch
`arena/01a02263-tryarena`. `log.md` (34 cycle blocks) is the full historical
record; this file is the index + in-flight state + directives + pitfalls._

## 1. WHAT THIS PROGRAM IS
Beat the transformer — architecturally, not by budget — on a constrained
box: 2 CPU cores, ~4GB RAM, 1-thread torch, everything runs in the sandbox.
We build heterogeneous machines: a shared embedding table + learned
per-example router over ISOLATED branches, each branch = a small linear
SSM host (d16) + an EXACT MEMORY ORGAN (a small learned readout over
exact, mechanism-computed state features). Every capability is certified
against an analytic oracle (dCE = CE minus exact-entropy oracle), probed
at 4096 and 16384 (256x training length), on a frozen verify suite
(35/35). The win is proven architectural by the 796k-param TF control
that still loses (C11) and by the fact that TFs cannot even allocate
memory at 16384 while our O(1)-state machines hold cert-level.

## 2. OPERATOR STANDING DIRECTIVES (do not violate)
1. NO further transformer re-tests of any kind (C17 directive): "there is
   no way to make a transformer generalize and reason on this hardware,
   don't waste my time, just improve our system." Every cycle builds OUR
   architecture. TF numbers are cited from existing logs only.
2. Chatbot is a FIRST-CLASS AXIS (latest directive): "it also needs to
   nail it as a chatbot, we can't be limited to certain things." The
   machine must be a chatbot: fluency (C21 LM host) + perfect state
   memory + exact computation, fused. Always state the honest boundary:
   no open-domain world model at 2GB (capacity, not architecture, for ANY
   design); we claim the structurally different chatbot.
3. Never stop, never ask questions; autonomous loop; compact log-style
   reports only (minimal conversational text).
4. Internet-search each new mechanism before implementing (log the cite).
5. Log every cycle: log.md block + log.jsonl RESULT line + PROBLEM_MAP.md
   section + verify_suite.py (35/35) + git commit (push when it works).
6. Honesty clause: world knowledge/alignment/multimodality are
   data/scale problems, not contested from a 2GB box. Never claim an
   un-certified result; log negatives and near-misses explicitly.
7. Read existing files before writing new cycle code; build on the
   logged next steps of the relevant line.

## 3. IMMEDIATE WORK QUEUE (in order; all scripts exist in the repo)
### C19 — machine v7 depth-k<=8 readout (IN FLIGHT, was reset-killed ~step 6.5k)
- `unified_kstack8.py` (Machine V7, 38,479p): v6 organ scaled top-4 ->
  top-8 features (s-bits 8->16, M 17x10, A 36 rows), Q0..Q7 (VOCAB 83),
  k uniform in 1..min(8, depth). 12k cycling over 5 tasks (echo, kstack,
  icl, mod7, add), ckpts 3k/6k/9k/12k, per-ckpt eval incl. @16384 probes
  and a per-k answer-CE diagnostic (the deep columns k=5..8 are the claim).
- On disk: `unified_kstack8_3000.pt`, `unified_kstack8_6000.pt`
  (last logged: lm 1.5920, routing 0.0000, organ mass 1538.9 at 6k).
- RESUME (the script has RESUME=1 support: loads latest ckpt,
  fast-forwards the data rng exactly, continues to 12k):
  `cd TryArena && OMP_NUM_THREADS=1 RESUME=1 nohup python3 -u unified_kstack8.py >> unified_kstack8.log 2>&1 &`
  (~13 min train + ~8 min eval). If the reset already wiped the .pt
  files (check `ls unified_kstack8_*.pt`), run fresh without RESUME.
- SUCCESS: (i) kstack(k<=8) dCE @4096 <= 0.01 at 12k (v6 got 0.0037 at
  k<=4); (ii) @16384 within 3x @4096 every ckpt; (iii) echo <= -0.25,
  icl tgt @4096 <= 0.005, mod7 <= 0.01, add <= 0.02 at 12k; (iv) routing
  1.0; (v) per-k answer CE <= 0.05 for ALL k=1..8 (else a deep column
  silently failed).
- After: log block + PROBLEM_MAP + commit + push.

### C20 — generalization probes (READY: `probes_c20.py`, eval-only, ~10 min)
Runs on `unified_kstack8_final.pt` (i.e. after C19): controls (in-machine
mod7 ~0.0034, ICL single ~0.0 must reproduce) + zero-shot: P1 mod5 walk
(novel wrap transition — ring transfer), P2 mod6, P3 ICL 3-query/row
(register persistence), P4 ICL redefinition (latest value wins), P5
kstack bottom (k=depth, depth<=8) + deep k=8 under load (per-depth
answer CE = the exact capability boundary), P6 subtraction zero-shot on
the add vocabulary (expected FAIL -> certifies transition-specificity,
defines the borrow-organ build). Tag ARC2-C20-GEN-PROBES.

### C21 — LM host, the chatbot fluency engine (IN FLIGHT, killed early; RERUN)
- `lm_host.py`: SSMBlock(d32) (the machine's host verbatim) + tied
  embedding, 35,968p, byte-level BPE (vocab 768, `bpe_tok.py`, cached in
  `corpus/tok_cache.pkl`). Corpus `corpus/corpus_full.txt` = 1.0MB real
  text (public-domain English prose, Austen, fetched via web tool +
  program's own code/prose), 542,719 tokens, 90/10 train/val.
- Rerun fresh (had no ckpt at kill time):
  `cd TryArena && OMP_NUM_THREADS=1 nohup python3 -u lm_host.py > lm_host.log 2>&1 &`
  (~31 min). RESUME=1 supported if it dies after a ckpt.
- SUCCESS: (i) val CE @256 <= 4.0 at 12k (uniform prior ln768=6.644);
  (ii) CE @16384 within 1.3x of CE @256 (LENGTH INVARIAVARIANCE on real
  text = the architectural claim); (iii) two logged generations
  (prose + code prompt) are coherent word sequences.
- NOTE: this trains on CPU in parallel with C19 — two 1-thread processes
  on 2 cores is fine; do not run a third.

### C22 — THE CHATBOT MACHINE (design; build after C21)
Fuse into ONE artifact: LM host (fluency) + SRAM/stack organs (exact
state: names/facts/commitments) + carry/kstack organs (exact math in
conversation) + per-example router. First certified form: dialogue
streams with injected state — assistant replies must reproduce state
values at 16k context (exact oracle on state tokens) while reply tokens
stay fluent (CE). This is the operator's chatbot: speaks, remembers
perfectly, computes exactly, at a context length TFs cannot allocate.
State the honest boundary in the report (no world model at 2GB).

### C23 — router hardening (StickyMoE-style load/consistency) + borrow organ
(subtraction: c=(a-b-borrow) mod 10, 1-bit exact transition, same pattern
as the carry organ — 15-min build if C20 P6 fails as expected).
### C24 — P4 multi-pass machine (open-ended iteration — unsolved by
anyone; the deepest remaining novelty).
### C25 — P3 full multi-digit algorithms (division etc. = carry/borrow
organs + iteration).
### C26 — P6 variable-binding probes (register file is native; likely win).

## 4. CERTIFIED MACHINE STATE (the crown; full detail in log.md)
MACHINE v6 (26,891p, C18): ONE model, 4 isolated branches (r0 SSM host +
stack organ [top-4 exact features, additive + state-x-query bilinear
readout], r1 host + SRAM organ [16 slots, content-addressed], r2 host
only [finite-state], r3 host + carry organ [1-bit exact transducer]),
learned 4-way per-example router (MLP on first 3 token embeddings,
direct CE), shared token-disjoint embedding (VOCAB 79), dual-gating
(exp head-gate per branch + exp organ-gate on SRAM). All heads
zero-init (L-GATE-INIT); every logit term carries a learnable scale.
@4096 dCE / @16384: echo -0.3198 / -0.3198; icl target 0.0 / 0.0;
mod7 0.0034 / 0.0034; add 0.0154 / 0.0155; kstack(k<=4) 0.0037 / 0.0032;
routing 1.0; transient-free across 3k-12k ckpts. 12k steps, 19 min wall,
723MB peak, 1 thread.
MACHINE v7 (38,479p, C19 in flight): v6 with the k<=8 organ.
TF controls (why this is architectural): micro TF 104,843p/12k loses
15-40x @4096 (echo 4.78, icl 5.39-10.57, mod7 5.63, add 4.82); STRONG TF
796,571p/10k/8x-compute/peak 2001MB still loses (mod7 0.0068@64 ->
8.30@4096 = length decay; dyck 3.42; echo 21.98) and cannot run @16384.
Training-time axis: our 21k machine 505s vs 796k TF 1935s — and the TF
still loses. Hahn TACL-2020 finite-ops bound is the mechanism.
Standalone certs (the bar the machine must match internally): echo
-0.2935; SRAM 0.0217-0.0270 (4,353p); mod7 0.0025-0.0071; dyck-3 0.0068.

## 5. LAWS (16; one line each; full proofs/derivation in log.md)
L-LINEAR-HOST: sub-quadratic linear hosts are length-invariant,
near-oracle on finite-state tasks. L-ROUTING-BEATS-FUSION: per-example
routing over isolated experts beats one fused model. L-STACK-NECESSITY:
non-regular reads need an explicit-stack organ. L-STATE-SUFFICIENCY: the
organ's exact state is sufficient (host is optional ballast).
L-TWO-HOP: query-then-read needs two mechanism hops. L-STABLE-PARTIAL:
partial mechanism + learned readout is the stable regime.
L-RELIABLE-EXACT: direct-gradient table classes have zero
crystallization lottery (9/9 seeds). L-MARKOV-COMPLETION: a readout that
is a function of (top, empty, prevC) alone can only complete, not query.
L-ORCHESTRATION: one model = shared embedding + router + isolated
branches (zero sharing => zero interference; vocabularies are disjoint).
L-DUTY: every branch must get full-batch pure-task steps (cycling),
else duty starvation. L-ORGAN-GATE: a ported organ's readout needs a
learned exp-scale (soft start) + its own alphabet (L-ORGAN-ALPHABET).
L-NO-CTX-LIMIT: trained L=63, cert-level at 16384 = 256x (size state
caps to the task's max depth before length probes — C16 cap lesson).
L-DUAL-GATE: a co-trained head + learned organ sharing a logit sum need
per-term learnable scales; frozen scales => checkpoint oscillation +
length-dependent error (the C17 transient fix). L-QUERY-READOUT: a
readout serving query-keyed retrieval over exact state needs a learned
state x query joint term (bilinear over exact one-hot features);
zero-init the interaction; keep its inputs exact (full-rank gradient
from step 1). L-ENCODING: task hardness class is representation-
relative (carry became a KR-mode automaton under the right encoding).

## 6. PROTOCOL CONSTANTS
dCE vs analytic oracle (oracle = exact generation entropy per token;
masked/random positions use their true entropy). Eval lens 64/512/2048/
4096 (+16384 probes). Machine tasks: batch 32, L=63, 12k cycling steps,
ckpts 3k/6k/9k/12k, AdamW 3e-3, clip 1.0, seed 0, data rng =
random.Random(17), OMP_NUM_THREADS=1. LM: L=256, same lr/seed.
TF reference config (cite-only, never run): sinusoidal pos + tied
embedding. Per cycle: declare SUCCESS criteria BEFORE launch; report
verdict honestly (near-misses and negatives included); append log.md
block; append log.jsonl RESULT; PROBLEM_MAP.md section; verify_suite.py
must stay 35/35; git commit (+ push). Files: <name>.py + <name>.log +
<name>_<step>.pt ckpts in repo root; corpus/ for data.

## 7. ENVIRONMENT PITFALLS (this box is hostile; respect it)
1. ENVIRONMENT RESETS HAPPEN (VM fork): kills all background processes,
   wipes pip installs (torch!), and re-cloned the git repo at the base
   commit once (files on disk always survive). Response procedure:
   (a) `python3 -c "import torch"` -> if missing:
   `pip3 install --break-system-packages torch numpy`;
   (b) for any in-flight script: check its on-disk ckpts and relaunch
   with `RESUME=1` (C19/C21 scripts support it);
   (c) git: `git add -A && git commit` from disk (the files ARE the
   record), then push;
   (d) full history is now on GitHub branch `arena/01a02263-tryarena`
   (pushed 2026-08-22) — a fresh sandbox will have everything on clone.
2. Bash default cwd = /home/user, NOT the repo — always `cd TryArena`.
3. nohup launch quirk: `cd X && nohup ... & echo; sleep; grep` — the `&`
   backgrounds the WHOLE chain and the tail commands run in the wrong
   cwd / hang until timeout. Launch with a bare
   `cd TryArena && OMP_NUM_THREADS=1 nohup python3 -u script.py > log 2>&1 &`
   (expect the call to time out — that's normal), then VERIFY with a
   fresh call: `ps aux | grep [s]cript.py` + `tail log`.
4. nohup log can lag 1-2 min before the first line appears.
5. Direct HTTP is TLS-blocked (only pypi egress works): use the
   fetch_page tool for web content (wikisource/gutenberg reachable;
   gutenberg direct 504s).
6. Budget: 2 cores total — run at most TWO 1-thread processes; each
   machine run ~13-35 min wall, peak ~0.7GB.
7. pkill self-match: `pkill -f script.py` kills the shell too — use
   precise PIDs.
8. git push works (auth restored 2026-08-22); if it fails, tell the
   operator the GitHub connection needs attention.
9. In smoke tests: never index-assume positions (the C18/C19 smoke
   caught real bugs: generator depth guarantee, idx widths, q-state
   index collisions, in-place param writes). Unit-test organ wiring with
   hand-set table cells before any multi-minute launch.

## 8. KEY FILES
log.md (canonical record, 34 cycles) | PROBLEM_MAP.md (problem state +
attack order) | JUDGE_CARDS.md + answer_key.json (frozen public test
items; victory condition = 100% exact-match vs frontier chatbots,
operator judges) | verify_suite.py (35/35) | results.jsonl, log.jsonl.
Machines: unified_stable.py (v5, 21,309p), unified_kstack.py (v6,
26,891p), unified_kstack8.py (v7, 38,479p), unified_add.py (v4),
unified_iso3.py (v3), unified.py (v2 data layer + SSMBlock). In flight:
probes_c20.py, lm_host.py, bpe_tok.py, corpus/. TF reference (DO NOT
RUN): strong_tf.py, tf_patience.py (on hold since C17).

## 9. WHERE THE BATTLE STANDS (one-paragraph score)
Beaten, proven architectural: long-context exact reasoning (carry chains
4096, dyck, k-th stack queries, mod walks) at 15-40x over the best TF
we could run, with the 796k control ruling out budget; length
generalization to 256x training length where TFs cannot allocate;
content-addressed exact memory (ICL) at 0.0 vs ln16 for every TF flavor;
training time/compute/memory (505s/723MB vs 1935s/2001MB — and the TF
still loses); stability (transient-free by construction since C17).
Open for everyone: P4 open-ended iteration. Our remaining certification
debt: within-family transfer probes (C20), deep-k columns (C19), and
the chatbot axis (C21/C22) — after which the only honest un-won axis is
open-domain world modeling, which is a capacity game no 2GB box can
contend for, in any architecture.
