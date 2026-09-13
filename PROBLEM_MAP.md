# ARC-2 PROBLEM MAP (locked, cycle 1)

## A. TARGET PROBLEMS - architectural weaknesses of SOTA models (scale does not fix)
P1 Length/context decay ............ BEATEN at machine scale (C16):
                                     all 4 organ families cert-level
                                     dCE at 16384 = 256x training len;
                                     TF cannot run @16384 (O(N^2) mem)
P2 Exact state tracking (TC0) ...... BEATEN at micro-scale, certified
P3 Counting & arithmetic (carries) . CORE MECHANISM IN MACHINE (C15):
                                     carry organ (1-bit exact transducer)
                                     dCE 0.0091 @4096 vs TF 4.82; full
                                     multi-digit algorithms remain open.
P4 Iteration / adaptive compute .... CORE NOVELTY (multi-pass machine needed;
                                     unsolved by us AND by them)
P5 Compositional generalization .... partial; needs dedicated benchmark
P6 Variable binding / symbols ...... register file native; untested, likely win
P7 Verifiable reasoning ............ our certification pipeline; deepest moat
P8 Memory cost (KV cache) .......... won by construction (O(1) state)

## B. INTERNAL BLOCKERS - what we must solve to claim the crown
P9  Crystallization lottery (~50%/seed) -> CLOSED FULLY (C10): 9/9 seeds
    deterministic on the end-to-end line (ssm_d16_1 5/5, echo-organ 4/4,
    zero restarts); TF baseline = consistent 0/3 loss, not a lottery.
    Law L-RELIABLE-EXACT. Best-generalizer blocker eliminated.
P10 Feedback-class learnability (carries) -> currently compiler-dependent
P11 Short-range fluency gap vs RoPE (1.1 nats) -> SOLVED BY REFRAME (C8):
P13 Content-addressed retrieval / few-shot ICL (attention's home turf)
    -> SOLVED BY CONSTRUCTION (C12): the SRAM organ (exact per-context
    register file, 4,353p w/ host) reads the 16-key cipher mapping with
    target CE 0.022-0.027 @4096 (ln16=2.773 for every transformer flavor,
    incl. the 796k strong TF lineage, even at training length).
    Length-invariant, 2 seeds deterministic. Law: exact associative memory
    is a first-class unit attention does not have. C9's "not decidable"
    referred to attention/SSM hosts alone; the organ line decides it.
    a host property, not a paradigm property. Linear (sub-quadratic) hosts
    are length-invariant and near-oracle on finite-state tasks at 3.2k params
    (L-LINEAR-HOST); routing beats fusion (L-ROUTING-BEATS-FUSION); beyond-
    finite-state (non-regular) reads need an explicit-stack organ (L-STACK-
    NECESSITY, dyck-echo: organ dCE 0.0106 vs ssm 0.6057 vs tf 2.42 @4096).
P12 Training-speed engineering (scan is proven, implementation is loop)
P14 Memory orchestration (which exact memory serves which sequence)
    -> SOLVED BY CONSTRUCTION (C13): unified machine, 6,197p, ONE model =
    shared linear host + exact-stack organ + SRAM organ + learned
    per-example router (100% routing acc @4096 on a 3-family mixed stream,
    deterministic). Beats the 103k-param TF by 15-40x on ALL three tasks
    (echo -0.3019 vs 10.17; icl-target 0.2057 vs 2.78; mod7 0.0113 vs
    7.27 @4096). Law L-ORCHESTRATION.
P15 Multiplexing of exact-memory organs -> CLOSED BY CONSTRUCTION (C14):
    three real limitations, three design fixes: (1) shared-backbone
    interference (C13 knee: echo -0.30 -> +1.13 @20k) -> per-task
    parameter isolation (L-ORCHESTRATION); (2) duty starvation (60k rows
    @batch24 > 80k rows @batch8) -> task cycling, full-batch pure-task
    steps (L-DUTY); (3) organ porting bugs (linear readout needs a
    learned soft-start scale + its own alphabet: 1.44 -> 0.0012 target)
    -> L-ORGAN-GATE + L-ORGAN-ALPHABET. Machine v3 (unified_iso3.py,
    13,011p): AT/BELow standalone cert on all 3 families inside ONE
    model, stable 1x->1.33x budget, routing 1.0, 40-1400x over the 8x TF.

## C. OUT OF SCOPE (honesty clause)
World knowledge, chat alignment, multimodality: data/scale problems, not
architecture problems. Not contested from a 2GB box.

## VICTORY CONDITION
Frozen public test items (JUDGE_CARDS.md). Operator pastes them into frontier
chatbots and records their scores; sandbox-trained machines must reach 100%
exact-match on the same items. Win = measured resource inversion.

## ATTACK ORDER
C1: suite + judge cards (this cycle) -> C2: T2/T4 quick certifications ->
C3-C6: the carry problem (P3+P10) -> C7+: multi-pass machine (P4) ->
parallel track: P9 reliability throughout.

## CYCLE-2 STATUS UPDATE
P3 (counting/arithmetic-addition): SOLVED — LSB pair-token reduction makes carry a
KR mode automaton; certified 12.5x length generalization (8->100 digits), 14/14 on
frozen suite. P10 note: carry was never feedback-class under the right encoding —
encoding choice is an architectural decision (new law: L-ENCODING — task hardness
class is representation-relative). T3 multiplication remains the true P4 fight
(nested iteration). Frontier column of SCOREBOARD.md awaits operator judging.

## CYCLE-3 STATUS UPDATE
P4 (iteration/adaptive compute): DEMONSTRATED — IFT (iterated learned transducer)
computes O(N^2)-work multiplication via input-dependent pass count. T3 certified
(200/200 @ 25-50 digits) and 5/5 on frozen items. Suite: 19/19 machine-side.
Remaining open on the map: P5 (compositional benchmark), P9 (lottery - note: zero
restarts needed in ARC-2 so far; direct-gradient table classes are lottery-free),
P11 (fluency gap). Frontier-LLM column = operator's move.

## CYCLE-4 STATUS UPDATE
P5 (compositional generalization): SOLVED — L-COMPOSE-EXACT (exactness composes;
23/23 suite incl. nested 30-digit expression items). P9 (reliability): CLOSED for
direct-gradient table classes — L-DIRECT-GRADIENT, zero restarts across all ARC-2
training runs, 4-seed sweep all-exact. Map remaining: P11 (fluency gap - hybrid
engineering), P4 open-ended extension (learned pass-programs for NEW algorithms).

## CYCLE-5 STATUS UPDATE
P4 extension: sorting hosted with NATURAL encoding (no design) — substrate generality
evidenced. Suite 26/26 (parity, cups, addition, multiplication, nested expressions,
sorting). ARC-2 restart count still ZERO. Remaining: P11 (LM-host hybrid), division/
GCD as further substrate instances, operator's frontier column.

## CYCLE-6 STATUS UPDATE
Arithmetic set COMPLETE on the substrate family: + (KR stream), x (iterated IFT),
/ (KR stream), nested composition (learned dispatch over certified organs), plus
parity, state-chains, sorting. Suite 29/29 machine-side; ~1.7M params total across
all seven machines; total training wall-clock across the entire program: <5 min.
Open: P11 (LM-host hybrid), big/big division, operator's frontier column.

## CYCLE-13 STATUS UPDATE
C7-C12 recap (details in log.md): capstone 3.2k-param machine beats 85k
TF at 64p-48p (C8-C10, 8 laws incl. L-RELIABLE-EXACT 9/9 seeds);
STRONG-TF 796k/10k-steps still loses both axes => limitation is
architectural, not budgetary (C11); SRAM organ closes P13 (ICL) by
construction (C12).
C13: P14 SOLVED — the four certified units (host, stack organ, SRAM
organ, hard router) trained as ONE 6,197-param model on a 3-family mixed
stream; per-example learned routing 100% @4096; wins 15-40x over the
103k TF on all three tasks. P15 OPEN: 20k diagnostic shows the
multiplexing cost is TASK INTERFERENCE on shared parameters (flat loss
hides drift: echo -0.30 -> +1.13, ICL target 0.21 -> 0.44, mod7 0.011 ->
0.006; router 1.0 throughout). Best config 10k/8-16-8.
C14: gradient isolation per routed task. Suite 35/35.

## CYCLE-15 STATUS UPDATE
P3 core mechanism: SOLVED BY CONSTRUCTION — carry organ (arithmetic
transducer: 1-bit state, exact mechanism transition, learned (carry,a,b)
-> sum readout in its own alphabet) added as branch r3 of the machine;
cascading 4096-length carry chains at dCE 0.0091 (TF 4.82). Machine v4:
21,305 params, 4-way learned per-example routing (100% acc), all four
families (finite-state / bounded-stack / exact-associative-memory /
arithmetic-transducer) at or beyond standalone cert inside ONE model.
C14 laws in force: L-ORCHESTRATION, L-DUTY, L-ORGAN-GATE, L-ORGAN-
ALPHABET. Known transient: SRAM branch ICL oscillates in a training
window (0.0021 -> 0.1972 between 9x and final ckpt; logged).

## CYCLE-16 STATUS UPDATE
Context window: machine v4 (21,305p, trained L=63) at 16384: echo
-0.2947 (cap-sized), icl 0.0046|0.0025, mod7 0.0028, add 0.0096,
routing 1.0 — no decay at 256x length (L-NO-CTX-LIMIT). Caveats logged:
mechanism depth cap is a design constant; host-organ coupling makes
transient organ states length-sensitive. Next (C17): depth-k stack
readout organ (query the k-th stack element — beyond (top,empty,prevC)
Markov completion) + SRAM transient-stability study.

## CYCLE-17 STATUS UPDATE
SRAM transient: CLOSED — machine v5 (unified_stable.py, 21,309p) dual-
gating: learned exp-scale on every branch's host head (logits_r =
exp(head_gate[r]) * head_r(h_r) + organ_r; init 0 = neutral, symmetric to
organ_gate). ICL target @4096 now MONOTONE across ckpts 0.0202 -> 0.0013
-> 0.0001 -> 0.0002 (v4: 0.0228 -> 0.0021 -> 0.1972, 80x swing); @16384
within 1.1-2x of @4096 at EVERY ckpt (v4 final: 8x, length-sensitive).
Final: echo -0.3153, icl 0.0051|0.0002 (100x below standalone cert
0.0218 and now the stable state), mod7 0.0034, add 0.0095, routing 1.0,
wall 533s, peak 708MB. LAW L-DUAL-GATE: a co-trained head and a learned
organ sharing a logit sum need per-term learnable scales; frozen scales
couple their magnitude dynamics into checkpoint oscillation +
context-length-dependent error. (Honest note: the head gates OPENED,
1.7-2.5 — the mechanism is per-term scale/direction decoupling in head-
organ co-training, not host shut-off.) Operator directive (C17): no
further TF re-tests; every cycle builds the machine. Next (C18): depth-k
stack readout organ (query the k-th element — first task beyond
(top,empty,prevC) Markov completion).

## CYCLE-18 STATUS UPDATE
New capability: DEPTH-K STACK READOUT — machine v6 (unified_kstack.py,
26,891p): r0 organ replaced by top-4 exact stack features (value+valid
bits per depth + prevPop) x query state (none-push/none-pop/Q0..Q3);
readout = learned additive table (0.1-randn) + learned state x query
BILINEAR (zero-init; inputs are exact features => full-rank gradient from
step 1). Task T5 (triplet stream op/Qk/ans, k uniform in
1..min(4, depth-after-op)): dCE 0.0781 -> 0.0143 -> 0.0103 -> 0.0037
@4096 (3k/6k/9k/12k), 0.0032 @16384 — length-invariant by construction
(answer depends only on the top-4 elements). First task in the program
that is unexpressible by the (top,empty,prevC) Markov-completion organ
family (L-MARKOV-COMPLETION's boundary, now pushed). No regressions:
echo -0.3198 (best in the program — the richer organ also helps echo),
icl target 0.0 @4096/16384, mod7 0.0034, add 0.0154 (duty cost of the
5th family, within bar), routing 1.0, transient-free 3k-12k, wall
1121s, peak 723MB. LAW L-QUERY-READOUT: an organ readout serving
query-keyed retrieval over exact state needs a learned state x query
joint term (bilinear over exact feature one-hots); a readout that is a
function of state alone (or query alone) cannot express "the k-th
element"; zero-init the interaction and keep its inputs exact. Next
queue: (a) generalization probes (mod7->mod5, bottom-of-stack echo, ICL
permutation, P3 full multi-digit); (b) organ scale-up (top-8 features,
k <= 8 queries: s-bits 8->16, M 9x6->17x6); (c) router hardening
(StickyMoE); (d) recover the duty overhead (add 0.0095->0.0154).

## CYCLE-19/20/21/22 STATUS UPDATE (chatbot axis + generalization closed)
C19 MACHINE v7 (38,479p): depth-k<=8 readout. per-k answer CE @4096
0.0025..0.0045 (k1..k8, all <= 0.05); kstack @4096 0.013 / @16384
0.0097 (better at 4x — length-invariant); echo -0.313, mod7 0.0036,
add 0.0043, routing 1.0; icl tgt 0.0054 (near miss; 9k transient closed
by 12k). L-QUERY-READOUT extended: deeper k = small monotone CE
increment, no phase change; organ scales by widening exact features.
C20 GENERALIZATION PROBES (zero-shot, v7 final): controls reproduce
(mod7 0.0029/0.0, icl single 0.0052/0.0009). TRANSFER: ICL multi-query
(0.0052->0.0028 @16k, ans 0.0) + ICL redefinition (ans 0.0008 — organ
write is mechanism-level LATEST-WINS, expresses zero-shot while host is
confidently wrong mid-stream: dCE 20.6, dual-gating keeps it off the
answer) + kstack bottom/deep (per-depth 0.0022-0.0050, exposure cap =
8 s-bits exact). FAIL: mod5/mod6 walks (4.40/3.37 — ring exact, not
modulus-general) + subtraction (ans 6.13 — transition-specificity
certified; defines the borrow organ).
C21 LM HOST (chatbot fluency axis, 35,968p, 768-byte BPE, 1.0MB real
text): length-invariance PASS on real text (CE @16384 = 1.007x @256,
ce1024 3.045 best); CE @256 4.2704 vs 4.0 bar = MISS (flat 4.31->4.27 =
d32 capacity ceiling on mixed corpus); gens ~20 coherent in-dist words
then degradation (capacity limit, logged). C21b = scale host (d64) /
lengthen L.
C22 CHATBOT MACHINE v8 (20,518p, IN FLIGHT): 3 branches on a 36-vocab
dialogue surface — r0 STATE organ (mechanism-computed conversational
slots NAME/CODE + query-keyed bilinear readout, incl. OVERWRITES),
r1 MATH organ (plus 2-digit + mod-10 minus = borrow organ pulled
forward), r2 CHAT host-only echo; learned router, dual-gated heads.
Bars D1-D7 (state 0.01 / overwrite 0.05 / length 4096->16384 / math
plus 0.02 minus 0.05 / chat 0.02 / routing 1.0 / logged dialogue
exact). Next: C22b fluency-into-chatbot fusion (load lm host d32+768-
emb as 4th branch), C21b d64 fluency host, C23 router hardening, C24
multi-pass, C25 multi-digit, C26 variable binding.
C22 CHATBOT MACHINE v8 RESULTS (2026-08-23, relaunched after reset): PARTIAL.
PASS: D3 length-invariance (state 0.2269@16384 vs 0.2271@4096), D5 chat
0.0002, D6 routing (rt CE 0.0000), D7-greedy name/code queries correct.
MISS: D1 state @4096 0.2271 vs <=0.01 bar (FLAT floor 3k-12k = readout not
crystallized, not a transient), D2 overwrite 1.05 (latest-wins not
expressed), D4 math-plus 0.0519 @12k (9k ckpt 0.0005 PASSED then regressed
= L-DUAL-GATE oscillation), math-minus 0.0515 borderline (9k: 0.0027).
Mechanism: state bilinear mass still growing @12k (587->1059) = undertrained.
C22-R repair queued behind C24 (operator P4 priority); C22b fusion waits for
the state bar.
C22-R CHATBOT REPAIR (2026-08-24, cycle 38): CERTIFIED — all 7 bars D1-D7 on
champion c22r8.pt (machine v9c, 20,518p). Two latent defects fixed, five
compounding. (1) EVAL BUG: probe oracle never subtracted the iid turn-choice
entropy at U-turn-start positions (H=1.667 nats) — D1 bar 0.01 unreachable
for any model; v8 was already -0.027 on corrected oracle. (2) MECHANISM BUG:
v8 organ emitted PRE-update so query one-hots fired at the answer-TOKEN
position while probes score the A-marker position (off-by-one) — bilinear
organ contributed nothing where CE is scored. Round-2 v9 staged query machine
(arm at NAME/CODE, fire at A; code-ones at d1) + math organ from A + corrected
oracle: PASS D1/D3/D4/D5/D6; math-minus fixed 0.059->0.004. Rounds 3-8 then
fixed: length collapse (host0 SSM decay log_a drift to 0.986 + head blowup ->
CLAMP a<=0.90, organs own persistence), D7 dialogue math (math turns into
state family + math organ in host0 branch), long-range organ push (long-window
L512/L1024 fine-tune + overwrite-distance curricula + st_m gain x2.4 total).
FINAL: D1 -0.065 | D2 0.039 | D3 -0.070 | D4 -0.000/0.000 | D5 0.000 |
D6 1.0 | D7 exact (dave/it/1-2/fine/6/4-2). Robust over seeds and at 8192.
New laws: L-ORACLE-COMPLETE, L-EMIT-TIMING, L-DECAY-DRIFT,
L-TRAIN-LENGTH-MISMATCH, L-ORGAN-GAIN. Champion c22r8.pt. Next: C22b fusion.
C24 MULTI-PASS (P4, 2026-08-23): CERTIFIED via armB (orbit-supervised pass):
iterated increment, input-driven pass count = k EXACT, 16x depth (k=64) and
8x length (120 digits) generalization, 100% exact at all scales, mechanism
halt (fixpoint). NEGATIVE (honest): end-to-end terminal-contract-only
learning (armA/A2) did not crystallize the data pass — open-ended protocol
discovery remains the unsolved P4 frontier. New laws: L-MECHANISM-HALT,
L-ORBIT-COVERAGE. M5 CA-k stretch deferred to C24b.
C24b CA-k LOOP (2026-08-23): CERTIFIED — second instance of the multi-pass
mechanism (rule-90 light-cone pass, lookahead head): 16x depth + 8.5x length,
100% exact, pass count = k+1 exact. Run 1 joint 96/100 = honest miss; repair
(3k steps, +L21 stage) = ALL PASS. P4 input-driven iteration now generic
across two task instances; e2e protocol discovery still open.
C24c-k P4-DISC (2026-08-23): DONE — ALL BARS PASS (ARC2-C24K-P4-CRISPFIX).
Discovered program: search found the counter protocol in 2 edits (MARK->SEP
dissolution); SGD learned the digit pass (+1 mod 10 on LSB-first digits,
carry = state persistence on 9). 500/500 in-dist; 200/200 k=16; 100/100
k=64 (4x unseen); 100/100 joint k=64xL=120; passes=k+1 exact; crisp
execution. Laws: L-NEEDLE-SEARCH, L-SUPERPOSITION-HIDE, L-EVAL-FIDELITY,
L-NEUTRAL-BRIDGE, L-FITNESS-OVERFIT.
C25a P3-LOOP (2026-08-23): DONE — iterated subtraction (x-k) CERTIFIED, all
bars; counter protocol reused with ZERO rows changed (L-MODULAR-REUSE);
borrow organ = mirror of carry organ (persist on 0). C25b stride-2 (x+2k)
FAILED (needs 2 digit states: L-CARRY-IN-STATES); queue c25b-R, then C26.
C21b FLUENCY SCALE-UP (2026-08-25, cycle 39): NEGATIVE. d64 (91,648p)
ce256 4.40 WORSE than d32 4.2704; ceiling is corpus-limited (1MB, ~100
epochs, train 1.73 vs val 4.40 = memorization), not capacity. Length
invariance holds (1.007x). Bar 4.0 not met; honest claim = length-invariant
fluency engine. L-DATA-CEILING banked. Fluency fusion (C22b) may carry the
d32 engine as-is. Strategy reset (operator): win = one coherent model under
the box; after coherence, generalization/reasoning to the absolute limit.
Next: C26 binding wall re-entry via VALUE-ENCODED TRANSPORT (new machine
class; discrete table family exhausted, L-PLATEAU-ATTRACTOR).
C26 BINDING WALL (2026-08-25, cycle 40): BROKEN by new machine class
VALUE-ENCODED TRANSPORT (VET): control Mealy (5 states) x mechanism value
register. All bars PERFECT: 500/500 in-dist; 200/200 nd=16 (was 9/200);
100/100 nd=32 & nd=64-joint (were 0/100); passes=nd+1 exact; one-mark
trace; stretch exact at nd=128/256/512 (depth-unlimited by construction).
Diagnosis: wall = joint (flags x value) state encoding budget in discrete
tables, not task hardness. LAW: L-VALUE-CHANNEL. Cycle 41: DISCOVERED by search from blank genome in 877 evals/20s; all bars
perfect. C26 BINDING CERTIFIED under the VET class. PLATEAU-BLOCKED v4
verdict VACATED (representation artifact, not task hardness). Laws:
L-VALUE-CHANNEL, L-DISCOVERABILITY-BY-CLASS. C29 objective achieved.
CYCLE 42 (2026-08-25) — REASONING FRONTIER probe 1: REVERSAL binding
(tgt_i <- d_{nd-1-i}) = NEGATIVE, PROVABLY. Theorem L-TRANSPORT-DIRECTION:
single-head left-to-right tape machines transport values monotonically
RIGHTWARD (writes only at/ahead of head; passes restart from the left),
so any permutation needing leftward moves — reversal needs them for half
the pairs — is unsolvable in the VET class (indeed ANY single-head LTR
class) at any state/register budget. Measured: 12,023-eval VET+counter
search stalls at 0.3985 (rightward-feasible ceiling 0.556); the pure-
leftward target tgt0 = 6/30 while rightward targets partial. CLOSED; do
NOT retry on this class. Next attack (cycle 43): LIFO / bidirectional
geometry (machine-v6 stack organ push/pop, second head, or tape rotation).
CYCLE 43 (2026-08-26) — REASONING FRONTIER probe 2: REVERSAL via
VET+S (VET + mechanism-owned persistent LIFO stack = machine-v6 stack
organ in the tape class) = capability CERTIFIED, ALL bars: 500/500
in-dist, 200/200 nd=16, 100/100 nd=32/64-joint, passes=nd+2 exact +
one-mark trace (spot nd=1..64), stretch EXACT to nd=512 (depth-
unlimited by construction). C42's L-TRANSPORT-DIRECTION wall broken
by the minimal class extension (one new mechanism channel): the wall
was a property of the linear-tape class, not of the task. LAW:
L-LIFO-OVERHEAD (n = nd+2, not nd+1 — LIFO output order is the reverse
of head target order; the push pass cannot emit; the one-pass price of
leftward transport). Smoke caught a real parity off-by-one pre-launch
(post-SEP index starts at 0: sources s-even, targets s-odd). ARM B
discoverability NEGATIVE at C41 budget (27,555 evals, best train
fitness 0.8350; best genome S1 115/500, S2-S4 0, trace discipline
True) — joint mark+scan+pop discipline beyond that budget; capability
unaffected. NEXT: C44 discovery re-entry (enlarged / staged /
contract-decomposed search, c24c-k precedent), then probe 3 =
arbitrary permutations (transport-distance analysis classifies
solvability per geometry).
CYCLE 44 (2026-08-26) — VET+S DISCOVERABILITY: CERTIFIED. Staged
contract-decomposed hill-climb (c24c P4-DISC precedent) from blank
genome discovers the reversal program in 266 evals (M1 4, M2 91, Sa
165, Sb 6) vs 27,555 evals/0.8350 FAIL for C41-protocol undirected
search (C43 ARM-B). Discovered genome passes ALL C43 bars: 500/500
in-dist, 200/200 nd=16, 100/100 nd=32/64, passes=nd+2 + one-mark
trace, stretch exact nd=512. L-DISCOVERABILITY-BY-CLASS confirmed
for the LIFO class. Basin: k=1 7/8, k=2 8/8, k=4 8/8 (attractor-
stable). LAW: L-CONTRACT-PURITY — stage fitnesses must enforce the
PER-PASS invariants downstream stages depend on: graded, cumulative,
structurally closed, precondition-bearing; terminal properties admit
lazy/parasitic solutions (3 instances forensically caught this
cycle: SEP-destroyer, lazy-scan, invariant-drift; plus zero-
gradient all-or-nothing scoring and a vacuous per-pass hole).
Companion process law: decompose to 2-entry needles before scaling
budget. Total laws ~29. NEXT: probe 3 = arbitrary permutations
(transport-distance analysis classifies solvability per geometry).
CYCLE 45 (2026-08-26) — REASONING FRONTIER probe 3: ARBITRARY
PERMUTATIONS on VET+S = CLASSIFIED. Two-level result. (A)
SCHEDULE level (exact DFS enumeration, wait-passes first-class):
ALL of S_n reachable — 24/24 @ n=4, 120/120 @ n=5, min passes
6-9 (L-LIFO-COMPLETENESS). Forensics: first model (consecutive
blocks only) under-approximated 21/24; search discovered
[2,0,1,3] with A=0, exposing the bug. (B) CONTROL level (5-state
value-agnostic (symbol,state) table; staged 6-stage discovery x
3 seeds): STRICT SUBSET — 14/24 @ n=4 (12 exact + 2 effective,
all verified 60/60; 10 plateau at a universal 0.4167 attractor
= reversal skeleton), 2/6 @ n=5, n=8 head-front DISCOVERED
(2n+1 passes), n=8 two-block NOT realized (0.300 @ 1505ev x3)
(L-STATE-BUDGET: pass number = mark-count chain <= 5 phases;
boundary pi need >= 6 distinct phase states — structural
argument + plateau evidence). Length generalization: exactly TWO
nontrivial universal families — reversal (n+2) and head-front
(2n+1), both n-invariant-gate schedules, one control verified
n=4..32 each; necessity probe [1,3,2,0]@n8-embedding = 0.0
confirms the rest are length-specific (L-LIFO-UNIQUENESS,
refined). Patch log: 5 items incl. an S1b cap artifact (nd+12 <
2n+1 at n=16) that briefly falsified head-front generalization.
Total laws ~32. NEXT: probe 4 = indirection / nested binding
(VET register organ under the discovered control).
CYCLE 46 (2026-08-26) — REASONING FRONTIER probe 4: INDIRECTION /
NESTED BINDING on VET+S (C45 mechanism + RSET peek / REM emit /
ACT_CLR + ADIG index class, 44 symbols) = CLASSIFIED. 1-hop
T_i := V_{a_i}: REALIZABLE at n=4 (certified hand control,
400/400 exact, tables intact, passes 2n) and LOCKED there over
n=3..9 (fx 0.000 at all other n, 40-tape sweeps) —
L-INDIRECTION-N4-LOCK; the earlier derived MOD-5 STRIDE exclusion
is REFUTED by the sweep (n=9, L%5=1 like working n=4, fails
identically; the L%5=0 cases fail by entry alignment, and co-
phased blocks would bind when the entry hits 3 — the per-digit
ADIG rows disambiguate, no reader exclusivity needed) — realiz-
ability at n!=4 (incl. n=3) is formally OPEN. 2-hop T2_i :=
V_{I_{a_i}}: UNREALIZABLE (L-INDIRECTION-DEPTH-1: the
intermediate value must be re-exposed to address table 2; at the
T1->V boundary the V entry is data-independent and the written
BDIG_v is invisible (Mealy-on-original); the only forward value
channel is the 5-value state (5 < 10 digits), r is opaque) —
derived, structural. DISCOVERY: 1-hop n=4 NOT found in 23,925
evals x 3 seeds (best 0.5017, ver 0.000) — plateau forensics:
MARK/SEP rows blank (no mark discipline, no entry), fs unstable
across tape sets (0.225 vs 0.717) = coincidental write
collisions, zero structural components; the ~100-entry needle is
INTRINSICALLY COUPLED (branch + RSET + REM must coexist for any
gradient) so staged decomposition is impossible — L-INDIRECTION-
UNDISCOVERABILITY: capability/discoverability separation (the
same search infrastructure finds the 5-entry LIFO needle in 266
evals, C44). Laws banked: OPACITY (r/S value-opaque; tape digits
only value-visible memory), REDUNDANCY (n^2 table replication per
reader at the 5-state budget), OVERHEAD (2n passes: RSET
selectivity forces T-state to hit the REM-eligible 0 shared with
cleared marks — cf. L-LIFO-OVERHEAD), N4-LOCK, DEPTH-1,
UNDISCOVERABILITY. Patch log: 6 items incl. RSET-hold bug
(re-peeked whole chain -> r = last V), stale-register trace
harness, impossible 4-stage decomposition, hill-climb
star-degeneracy on zero plateaus (fixed by plateau walk), and the
mod-5 refutation. Total laws ~38. NEXT: probe 5 = induction /
recursion.
CYCLE 47 (2026-08-26) — REASONING FRONTIER probe 5: INDUCTION /
RECURSION (unary data-dependent iteration) on VET+S = CLOSED at
the certified level. Depth-1 induction REALIZABLE and CERTIFIED:
REPEAT(k,v) = v^k, k in 1..4, hand control 400/400 value-
agnostic, passes = k+1, k=5 fails at the mod-5 collision (4/5,
L-INDUCTION-GATING edge measured). Depth-2 (a*b, a^b as
(a,b)-uniform computations) UNSETTLED: search (2-stage M1+Q,
plateau walk, 3 seeds) found OVERFIT ATTRACTORS only — in-sample
best=1.0, same-geometry ver ~0.87, but geometry-diverse
generalization 4-24/40-50; forensics: the "MUL(3,3) solution"
fills ~m/2 of the output region (9 at m=17 = 3*3 by coincidence;
7 at (2,3)/m=14, 10 at (4,3)/m=20), pure REM broadcast (no
push/pop, marks never cleared) — the fill count is an emergent
function of the GEOMETRY, not the value product. REFUTED: the
derived channel-decoupling bound (totals in {a,b,a+b}) — the
TAPE-ORBIT mechanism self-sustains REM writes far beyond a+b
(9, 14 measured). New laws: L-INDUCTION-TAPE-ORBIT (countdown =
the (symbol,state) trajectory over the EVOLVING tape; filled
BDIG cells re-route the state each pass; the classical 1C-vs-2C
counter separation does not transfer because the tape is value-
visible evolving memory) + L-INDUCTION-DEPTH-1 (certified).
Standing verify rule (from the same-geometry ver artifact):
verify bars must be GEOMETRY-DIVERSE (other (a,b) AND other
output-region size m), not just fresh values. Patch log: 4 items
incl. an uncapped-fs overproduction parasite (C44
L-CONTRACT-PURITY) and a score-offset bug (a+2 vs a+1+b) caught
before any claim. Prior art: 1C vs 2C separation (Minsky;
Hartmanis-Stearns unary squares), cited. NEXT: C48 = attempt a
genuine value-agnostic MUL hand control via the tape-orbit
construction (or bank the barrier); then C22b fluency fusion.
CYCLE 48 (2026-08-26) — REASONING FRONTIER probe 6: DEPTH-2
INDUCTION / value-agnostic MUL on VET+S = BARRIERED at scale 2..12:
certified unrealizable outside the 4-pair T1 corner {(2,2), (2,3),
(3,2), (4,2)} (T1: fills <= a+4 or m via L1 mark-pass budget
[312.5M classes x2] + L2 one-fill/pass + L3 tail max-prefix 4
[500k combos]; T2 mode-P: <= a+b+10); corner empirically blocked
(0.883 value-defective plateau: the (2,4) attractor is exact for
v=0..7 and dead at v=8 — a dead digit row; off-pair +-1 noise).
Rank-1 CERTIFIED realizable: (a,1) REPEAT a<=4 (mod-5 edge);
(1,2..4) ONESHOT per-b (front-clock transient, Ph[BLK] = F) + ONE
JOINT control (B_S4d witness: F=[2,0,3,4,4] 4-closed-and-fixed,
G={0,1,2,3}, PhDIG=[1,2,0,0,0], s0=0 — prefix confinement); (1,5)
unreachable (max fill 4). L-POP-COLLISION (new): the pop channel
cannot target the output region (emptied template steals pops,
first-eligible-BLK rule; POP-LOOP forensics 0/75, 100 template
fills) — REM is the only output writer. L-INDUCTION-FOUR: every
realizable data-value loop runs to at most 4. Total laws ~41.
NEXT: C22b fluency fusion.
CYCLE 49 (2026-08-26) — C22b FLUENCY FUSION: ONE COHERENT MODEL
UNDER THE BOX — WIN. FusionBot (single nn.Module, 68,738p = 1/11.6 of
the 796k TF): the entire C22-R champion (v9c, 20,518p) FROZEN on the
dialog surface (all 8 D bars re-measured through the fusion within
1e-4: D1 -0.0651 D2 0.0389 D3 -0.0704 D4 -0.0004/0.0002 D5 0.0001
D6 1.0 D7 exact) + the d32 fluency engine (lm_host_final.pt, 768 BPE,
STOCK SSM forward, a_max 0.923 as trained) as the 4th branch + a
LEARNED surface-router row 3 over the frozen 16-dim champion front
(b3 init -5; never steals a dialog stream: route_d 1.0 throughout;
4-way routing 32/32). Smoke: both surfaces BIT-EXACT vs their
standalone models at init (maxdiff 0.00). The engine carried as-is
and improved: fused vs standalone ce 4.3155/4.3199 @256,
2.1938/2.2153 @1024, 4.299/4.3547 @16384 (0.996x = length-invariant
inside the fused model). Honest boundary: bar-4.0 fluency NOT
claimed (L-DATA-CEILING). Operator win condition (fluency + exact
state + exact computation in one parameter set) MET. Files:
c22b_fusion.py/.log, c22b_stage1.pt.
CYCLE 50 (2026-08-26) — C49 INDUCTION CORNER RESOLUTION: the C48
rank-2 corner is CLOSED at the certified level. T1' SHARP (replaces
the loose C48 "a+P" bound): total REM-mode fills = max(r, K') <=
max(a, 4) — each r-phase fill shifts the front index by 1 (the
filled BDIG prefix walks F), so r-phase fills do NOT add to the
tail run; K' = L3 open prefix (machine-re-verified max 4, 500k
classes). Exact a*b <=> a*b <= max(a,4) <=> ONLY (2,2) in 2..12.
(2,2) REALIZED (hand: both marks clear pass 1, tail fold d=0,
F=[1,2,3,4,4], G={0,1,2,3}; 100/100 + value sweep 10/10, passes 5,
contiguous) AND DISCOVERED (hybrid v-deterministic search: 0.5
exact + 0.5 partial credit over all 10 values — 3,363 evals, ver
1.0/1.0 all values; the C48 protocol lands on a trace-1.0 dead-
row attractor, v=8, ver 0.9). (2,3)/(3,2)/(4,2) CERTIFIED
unrealizable (T1'; (2,3) also by mode-P pop-steal 5<6). COMPLETE
realizable value-agnostic MUL at scale 2..12 = rank-1 family +
(2,2) = 7 cells; (2,3) search consistency 0.583 < 1.0. L-DEAD-ROW-
ATTRACTOR (new search law, extends L-CONTRACT-PURITY to per-value
invariants + partial credit). Total laws ~44. Files:
c49_corner.py/.log, c49_corner_discovered.pt.
## ARCH-VET (cycle 51) — NEW ARCHITECTURE AXIS (VET-LM)
Novel-architecture PoC (operator directive): native learned
k-state Mealy controller x soft value register x exact top-4
LIFO (STE) x state x query bilinear readout, as a TOKEN-PREDICTION
model (8,372p) vs MambaMicro depth-2 d_state=48 (9,360p) vs
TFMicro 2L d16 sinusoidal-PE (8,144p), 4-task reasoning stream
(TRACK/MODK/DYCK/PAIR, V=48 L=256), 2000 steps, seed 0.
Prior art: Mamba-3 ICLR2026 arXiv 2603.15569 (SSM state-tracking
TC0 collapse line) + FSC line post-hoc only (arXiv 2602.08734;
ETH HRNN-LM; OpenReview S1gOpsCctm) -> gap: no native learned
Mealy x register x LIFO LM. P1: length-invariant CE = VET
1.316/1.257/1.295 @256/512/1024 (ratio .596) vs TF-micro COLLAPSE
1.346/3.865/5.144 (2.619, PE extrapolation) vs Mamba flat 1.402/
1.329/1.378; per-task eval corners: MAMBA-modk .423, TF-track
.512 in-range, VET-pair .057 best + best CE at length; DYCK
depth 3-4 unresolved for all arms at 8-9k. P2 ablations
(subset chain): A1 ctrl+query (5,534p) = counting only
(modk-eval .365; no track/pair) -> counting is a controller-
STATE property; A2 +soft register (7,150p) carries track/pair
+ CE flatness (pair-ev .189); A3 +LIFO (8,372p) marginal,
init-dependent (pair-ev .094 in P2 init). P3 (3rd base init,
seed-0 pre-construction; bit-parity False by design — fresh
default torch RNG is entropy-seeded, L-ENTROPY-RNG-NO-BIT-
PARITY): pair-ev .604(!) — the LIFO+stack basin EXISTS; eval-
acc init-fragile (track .302-.512, modk .212-.423, pair
.057-.604 over 3 inits), CE@1024 robust (1.294-1.296).
L-LIFO-INIT-FRAGILE. P4 frontier (single-task TRACK, train gap
4-16 -> eval 32-64..192-256): VETbase .595/.514/.450/.475/.275
(gentle decay, no cliff); VETbig k8/d24/K8 20,697p .946/.676/
.600/.500/.450 (near-saturation at first OOD band, 0.450 at
16x train gap); MAMBA .054/.108/.100/.175/.175 — VETbig beats
Mamba 6-27x at every point (L-STRUCT-SCALING). VERDICT: H1
SUPPORTED WITH NUANCE — structural LM wins length invariance +
frontier scaling at matched params; Mamba keeps modk-eval
corner; init-fragility + dyck-3/4 the open edges. Laws:
L-VALUE-CHANNEL-CARRIES, L-LIFO-INIT-FRAGILE, L-STRUCT-SCALING,
L-ENTROPY-RNG-NO-BIT-PARITY. Files: arch_vet_lm.py/p2/p3/p4 +
_run.log. NEXT: VETbig full 4-task @4000 steps (dyck-3/4?);
multi-seed basin rate of pair-ev .604 basin.
## ARCH-VET P5 (cycle 52) — VETbig full 4-task @4000
VETbig k8/d24/K8 20,697p, 4000 steps, seed 0: CE flat to 2048
(1.287/2.403/1.232/1.275/1.271 @256tr/hard/512/1024/2048, ratio
.529 — improves on base .596). Eval: track .488 / modk .212 /
dyck 0.000 / pair .717 = BEST-IN-PROGRAM. Key: the pair basin
(base: 1 of 3 special inits, .604) is captured under PLAIN seed-0
at 2.5x structure -> L-BASIN-SCALE-CAPTURE (structure scaling
stabilizes the basin; L-LIFO-INIT-FRAGILE reclassified as a
BUDGET property). DYCK 3-4 still 0.000 even at 2.5x structure +
2x steps -> structural limit of soft k-state; P9 VETDCC designated
attack (exact counter channels). MODK-eval .212 (corner stays
Mamba's, P1 .423; P9 mod channel also targets it). Files:
arch_vet_p5_run.log + RESULT ARCH-VET-LM-P5.
## ARCH-VET P6 (cycle 53) — basin rate quantified
Base 8,372p, 3 fresh inits (111/222/333): pair-ev .207/.434/.604
(rate >= .5: 1/3). Full 6-init sample (incl P1/P2-A3/P3): 2/6 reach
the .6 basin. L-LIFO-INIT-FRAGILE QUANTIFIED ~= 1/3 at base budget
vs ~1.0 at 2.5x (P5, n=1). CE@1024 stable 1.28-1.30 across ALL 6
inits (length invariance fully robust). modk-ev .269-.404 (base
matches Mamba corner under some inits). dyck-ev still 0.00-.018.
P8 VETCAM = the base-budget basin fix test (running). P10
candidate: VET-STE-DECOUPLED (arXiv 2410.13331 decoupled
tau_f/tau_b + 1611.01144 annealing + VQ-STE++ index-collapse
analogue) — schedule-level fix, no structure cost.
## ARCH-VET P7/P8/P9 (cycle 54) — mutation round 2
P8 VETCAM (content-addressed LIFO readout, 8,373p, seeds 0/111):
pair basin NOT stabilized (0/2 vs base 2/6) — L-LIFO-INIT-FRAGILE
holds at base budget; side: seed-111 modk .462 (beats Mamba's .423
corner, single sample unconfirmed); CE flat. P9 VETDCC (exact
mod-3 + depth-6 counters, zero-init injection; 8,902p/21,257p):
modk 1.000/1.000 BOTH arms = first perfect-score task
(L-EXACT-CHANNEL-PERFECT); dyck 0.000 at ALL depths 3-10 incl.
in-clamp on both arms — sharp prediction FALSIFIED: depth counter
!= type-order stack (L-DYCK-NEEDS-CONTENT-STACK); pair .226
base / .962 big = best-in-program (L-BASIN-SCALE-CAPTURE now
2/2 big inits). P7 DIVIDE frontier (train n 4-12, eval 13-24):
IDENTICAL 0.6/0.45/0.0 on VETbase/VETbig/MAMBA — no separation;
boundary is data-range not architecture (L-DIV-NO-SEPARATION);
CE@1024evaln diverges all arms (15.2/14.2/7.5) -> P7b L=256
control queued. Strongest config: VETDCC-big (pair .962, modk
1.000, CE@1024 1.263, ratio .5). NEXT: P10 exact bracket-type
stack (dyck content attack), P7b DIV length isolation, 2.5x basin
multi-seed.
## ARCH-VET P10/P11 (cycle 55) — dyck content attack
P10 STACKDCC (VETDCC + exact type-6 stack, 7 features; 9,273p /
21,649p): sharp prediction FALSIFIED — dyck 0.000 at ALL depths
3-10 on both arms (unit tests prove the features are correct).
Diagnosis: (1) dyck never learned beyond ~.22 even at TRAIN
interval by ANY arm — mixed-stream budget starvation
(L-DYCK-BUDGET-STARVED); (2) P10 features one position late
(pre-top vs needed post-top; L-STACK-FEATURE-LATENCY). Positive:
track-eval lifts to .628/.698 (channels add general structure).
P11 (launched): single-task dyck x 6 arms, STACKDCC2 (post-top
fix, 10 features), per-position open/close accuracy to separate
stack use from the grammar coin-flip ceiling.

## ARCH-VET P12/P13/P7B (cycle 56) — dyck TF control + capacity frontier + DIV length isolation
- P12 LANDED: TFMicro (8,144p) on the P11 single-task stochastic
  dyck protocol: close-type .678/.581/.452/.387/.345 d3/d4/d6/d8/d10
  — BELOW STACKDCC2-big (.925/.852/.724/.595 P11 citation; .791/
  .738/.624/.543/.488 same-session re-run), ABOVE Mamba/VETbase at
  d3-4. Exact 0.0 all arms (grammar ceiling). P11's dyck win over
  micro-TF CONFIRMED at matched protocol WITH init-sensitivity
  caveat (re-run differs from citation; arm ctor RNG order differs
  between p11/p12 scripts). L-DYCK-TF-PARITY-AT-MICRO not triggered.
- P13: deterministic dyck (every token state-determined → exact is a
  real bar). TRAIN-CORPUS LAW L-DETERMINISTIC-SINGLE-STRING-
  MEMORIZATION: fixed-depth deterministic training = ONE distinct
  string → all arms memorize (loss ~0.01), exact 0.0 everywhere,
  capacity D6/D12 indistinguishable on exact. CAPACITY SIGNAL (D12
  close-type flat ~.67 to d12 vs D6 decay to ~.001) CONFOUNDED by
  ctor-before-seed-reset (different inits). P13b (queued): depth-
  mixed train {1,2,3} + matched seed ctor → clean capacity test.
- P7B: L-DIV-RANGE-NOT-LENGTH. VET length invariance intact to
  L=1024 (flat acc, no compounding); DIV n21-24 ≈ 0.0 at every
  length = out-of-train-RANGE counting, not length. DIV axis closed.
- Protocol law: arm ctor AFTER manual_seed(0) (P13b pattern) so
  seed-0 = matched init across arms/scripts.
- OPEN QUEUE (ranked): 1. dyck certifiable win: exact-match ~1.0 in
  capacity via depth-mixed deterministic training (P13b) — the
  single open #1 problem; 2. basin multi-seed robustness (P14, 2.5x
  budget seeds 111/222/333 VETDCC-big/STACKDCC2-big); 3.
  associative-capacity probe; 4. in-range parity; 5. fusion;
  6. certificates.

## ARCH-VET P13B (cycle 56) — depth-mixed deterministic dyck, seed-clean capacity test
- Capacity purity CONFIRMED: D6/D12 bit-identical weights + d1-6
  identical (capacity = pure runtime buffer). P13's close-type
  capacity signal was init luck (ctor confound), now conclusively.
- L-DETERMINISTIC-POSITION-SHORTCUT: fixed-type deterministic
  grammar is NOT stack-essential OOD — S(d) embeds trained S(3)
  verbatim; TFMicro OOD close d4/d6 .85/.70 > stack arms .50/.51.
  Deterministic fixed-type grammar = dead end for the stack-vs-
  attention separation (problem 1). P13c: fixed shape + random
  per-node types (kills shortcut; exact depth d → clean D+1
  overflow boundary).
- In-train: depth-mixed training induces mastery (close d2/d3 1.0
  all arms; STACKDCC2-big exact d2 1.0 best). VETDCC-big OOD
  close below chance (anti-correlates; +L-DYCK-NEEDS-CONTENT-
  STACK).
- OPEN QUEUE (ranked): 1. dyck certifiable win → P13c (fixed-shape
  random-type exact-depth capacity frontier); 2. basin robustness
  → P14 (111/222/333, P11 stochastic protocol); 3-6 unchanged.

## ARCH-VET P13C (cycle 56) — exact-depth random-type dyck: stack channel never forced
- Capacity-frontier predictions FALSIFIED (D6 and D12 identical,
  both collapse at d3; capacity 12 irrelevant) → D_STACK buffer not
  the binding frontier. Root cause: depth-2 train corpus is
  K-window-solvable (VETDCC w/o type stack also scores d2 1.0) →
  no arm ever forced to drive the stk channel → no LIFO policy
  learned → d3+ fails for all arms.
- L-WINDOW-NOT-STACK-CANDIDATE (synthesis P11-P13c): all prior
  "stack wins" are consistent with the K=8 content window + fixed
  structure, NOT the exact type-stack doing OOD work (never
  observed under seed-clean tests). The P11 .925 claim needs a
  mechanism-level test.
- P13d (queued): train on DEEP stochastic dyck (target depths 3-6
  mixed) so train segments genuinely nest >2 → window fails in
  train → stack must engage → VETDCC-big is the discriminating
  control (must fail in-train deep segments).
- OPEN QUEUE (ranked): 1. dyck certifiable win → P13d (stack-
  forcing deep-mixed train) + P14 basin (in flight); 2. basin
  robustness → P14 running; 3-6 unchanged.

## ARCH-VET P14 (cycle 56) — basin rate at 2.5x structure on dyck: 1/3, claim downgraded
- STACKDCC2-big close_d3 by seed .655/.763/.631 → basin 1/3 at the
  TFMicro .678 bar, 0/3 at .85. VETDCC-big .641/.290/.403 → 0/3.
  L-BASIN-SCALE-CAPTURE does NOT extend to the dyck axis; the P11
  .925 citation = lucky basin (2/3 seeds below micro-TF bar).
- Content-stack survives only as mean margin (STACKDCC2-big d3/d4/
  d6/d8 means .683/.650/.544/.454 vs VETDCC .445 and vs TFMicro
  .678/.581/.452/.387) → downgrade dyck claim to "mean advantage
  with 1/3 basin".
- Combined with L-WINDOW-NOT-STACK-CANDIDATE (P13c): the stack
  channel's causal contribution is UNPROVEN. P13d (deep-mixed
  random-type, train {2..6}) is the mechanism discriminator: if
  STACKDCC2 cannot fit deep train segments that VETDCC fails, the
  stack is not usable by the learned controller at this budget.
- OPEN QUEUE (ranked): 1. dyck axis → P13d (stack forcing, in
  flight next); 2. basin on the OTHER certified axes (pair-eval
  .717/.962 n=2 also needs multi-seed before claiming robustness);
  3-6 unchanged.

## ARCH-VET P13D (cycle 56) — deep-mixed random-type dyck: depth-diversity is the lever
- STACKDCC2-big D6/D12 (bit-identical weights) in-train d4-6
  .983/.994/.980; OOD flat d7-12 .92-.98 (D12) / .92-.96 (D6) —
  length-invariant to 32x (L=16396). VETDCC (no stack) .89-.96;
  TFMicro .66-.85 d3-6 (below all VET arms, length-bound).
- L-CONTINUOUS-STATE-CARRIES-DYCK: the discrete type-stack is a
  +3-9pp enhancer, NOT the indispensable organ; capacity is not a
  cliff (D6 keeps .92+ past overflow; D12-D6 gap only 1-2pp).
- Depth-diverse training (d4-6) induced deep generalization in ALL
  arms (vs P13c depth-2 train: all fail) → Zhou et al. 2023
  training-depth-diversity reproduced at micro scale. Dyck axis
  re-scoped win: depth-generalizing, length-invariant close-type
  tracking in the VET family > micro-TF.
- OPEN QUEUE (ranked): 1. basin robustness on OTHER certified axes
  (pair-eval .717/.962 is n=2 seed-0 — multi-seed needed); 2.
  in-range parity; 3. associative-capacity probe design; 4.
  fusion pilot; 5. certificates. (dyck-axis problem downgraded:
  mechanism now understood and re-scoped as a VET-family win.)

## ARCH-VET P15 (cycle 57) — basin multi-seed on the full protocol: pair = lottery, counting+invariance robust
- VETDCC-big pair_eval .302/.302/.887 (basin@.717 1/3); STACKDCC2-
  big .057/.208/.925 (1/3). L-BASIN-SCALE-CAPTURE FALSIFIED for the
  pair axis at 2.5x (P9 .962 / P5 .717 = n=1 lucky basins; train
  pair 1.0 all 6 runs → the lottery is OOD-eval generalization).
- INIT-ROBUST certified properties (big config): modk eval 1.0 6/6
  (L-EXACT-CHANNEL-PERFECT), CE@1024/256hard ratio <=.6 5/6
  (length invariance). Stack organ does not help pair OOD (STACKDCC2
  mean .396 < VETDCC .497).
- Strongest-config table DOWNGRADED (pair .96 and dyck .925 are
  single-seed samples; modk + invariance are the certified pair).
- OPEN QUEUE (ranked): 1. pair-OOD basin stabilization (train 1.0,
  eval ~1/3 basin at any scale — the headline reasoning axis; P16 =
  search prior art → curriculum vs SRAM-style exact assoc readout
  vs 4000-step); 2. P13D depth-diverse dyck win multi-seed (is the
  .92-.94 close-type d12 init-robust like modk or lucky like pair?);
  3. in-range parity; 4. associative-capacity probe; 5. fusion;
  6. deeper TF dyck d12-16.

## ARCH-VET P16-P18 (cycle 57) — budget/structure basin diagnosis: pair budget-captured, dyck depth-diversity-captured
- P16: pair-OOD basin @4000 steps = 3/3 at .717 (mean .767 sd .036;
  P15 @2000 was 1/3). L-PAIR-LOTTERY-BUDGET: the pair lottery was
  2000-step underconvergence.
- P17: dyck (depth-2 shallow stochastic protocol) basin @4000 stays
  1/3 (mean .674 ≈ TFMicro .678). L-DYCK-BUDGET-NOT-CAPTURED:
  structural, not budget. C56 downgrade of .925 re-confirmed at 2x
  budget.
- P18: the DEPTH-DIVERSE protocol (P13D, train depths 2-6) is
  multi-seed ROBUST: STACKDCC2-big D12 close d12 .984/.950/.973
  (3/3 @.85 and @.90, mean .969 sd .014, flat to 32x length all
  seeds). L-DEPTH-DIVERSITY-CAPTURES-DYCK-BASIN. FIRST multi-seed
  dyck win.
- MULTI-SEED CERTIFIED TABLE (2.5x structure, seeds 111/222/333):
  modk 1.0 9/9 (P15+P16) | length-invariance ratio<=.6 8-9/9 |
  pair gap24-48 >=.717 3/3 @4000 (mean .767) | dyck close-type
  d12>=.85 3/3 @2000 depth-diverse (mean .969). The 4-task axis
  now has multi-seed certified wins on ALL families under the right
  protocol (budget for pair, depth-diversity for dyck).
- OPEN QUEUE (ranked): 1. P19 = mixed-stream fusion: does the
  depth-diverse dyck corpus coexist with pair/modk/track in the P9
  4-task stream (data-level fusion, the pre-fusion question)? 2.
  in-range parity; 3. associative-capacity probe; 4. VET-LM+corpus
  fusion (chatbot); 5. deeper TF dyck d12-16 (needs RoPE-TF
  re-derivation — NOT P1-class).

## ARCH-VET P19/P19B (cycle 58) — mixed-stream fusion: dyck survives, naive mix breaks the other families' OOD
- P19 (mixed-stream depth-diverse dyck x track/modk/pair, STACKDCC2-big D12
  @4000, seeds 111/222/333): dyck close d12 .9563/.9535/.9621 = 3/3 @.85
  (mean .9573, -1.2pp vs P18 single-task) — L-MIXED-DEPTH-DIVERSE-DYCK-OK,
  the depth-diverse dyck win survives data-level fusion. modk 1.0 3/3.
  BUT pair .377/.868/.283 (1/3) and CE ratio .798/.873/.856 (0/3) —
  prediction (b) falsified: naive equal-rate mixing is not clean.
- P19B (same arm/seeds/budget, VANILLA P9 corpus — the arm x corpus 2x2):
  dyck d12 .504/.791/.751 (0/3, mean .682) vs P19 3/3 .957 — the CORPUS
  carries deep-dyck (vanilla depth-2 training stays 0/3-deep at 4000 on
  this arm; L-DEPTH-DIVERSITY-CAPTURES-DYCK-BASIN re-confirmed corpus-
  level, not single-task-training). modk 1.0 3/3 (12/12 across P19/P19B).
  pair .509/.698/.925 (1/3, mean .711) — pair OOD stays 1/3 on vanilla
  too: NEW BANKED ARM NEGATIVE — STACKDCC2-big @4000 is pair-lottery 1/3;
  the P16 pair-budget win is VETDCC-big-SPECIFIC. ratio .508/.564/.501
  (3/3 <=.6) vs P19 0/3 — the deep-dyck mix BREAKS the certified length-
  invariance ratio (corpus effect: d5/d6 segments = 50-100% of a 256
  stream starve the other families / inflate hard-256 CE).
- FUSION LAW (answers queue #1): depth-diverse dyck is corpus-compatible
  for its own axis (-1.2pp), but NAIVE equal-rate data fusion is not clean
  for pair/ratio OOD; the certified configs must be reached per-family
  (budget for pair on VETDCC-big, depth-diverse corpus for dyck, 3/3 ratio
  survives only without the deep-dyck mix). Multi-task fusion therefore
  needs per-family token budgets/scheduling, not one shared mix — the
  VET-LM+corpus (chatbot) fusion pilot (queue #4) inherits this.
- OPEN QUEUE (ranked): 1. in-range parity (the last un-run P9 axis); 2.
  P19c = VETDCC-big x mixdd corpus @4000 (does the pair-carrying arm keep
  pair 3/3 under the depth-diverse mix? the missing cell of the 2x2);
  3. scheduled fusion pilot: cap dyck segment size / per-family token
  shares in one stream and re-test ratio+pair+dyck basins (C59 head);
  4. VET-LM+corpus fusion (chatbot); 5. associative-capacity probe; 6.
  deeper TF dyck d12-16 (NOT P1-class).

## ARCH-VET P19C (cycle 59) — fusion 2x2 completed: no naive config reaches all three certified bars
- P19C (VETDCC-big x mixdd corpus @4000, seeds 111/222/333): dyck
  close d12 .9129/.8521/.8772 = 3/3 @.85 (mean .881) — the deep-
  dyck basin under fusion is corpus-carried and holds WITHOUT the
  type stack (L-MIXDD-DYCK-CORPUS-CARRIED); pair .547/.623/.340 =
  0/3 and CE ratio .755/1.075/1.064 = 0/3; modk 1.0 3/3.
- PAIRED STACK CONTRAST (same corpus/seeds/budget, P19 vs P19c):
  type stack = +4.3/+10.1/+8.5pp (mean +7.7pp) on dyck d12 —
  L-STACK-HELPS-MORE-UNDER-FUSION (+7.7pp > the +3-9pp single-task
  estimate): the explicit type stack earns its keep under stream
  dilution.
- 2x2 COMPLETE (basin @4000): pair >=.717 only VETDCC-vanilla 3/3;
  ratio <=.6 only VANILLA corpus (3/3 both arms, 0/3 both arms
  mixdd) — L-MIXDD-BREAKS-RATIO-ARM-INDEPENDENT; dyck d12>=.85
  only DEPTH-DIVERSE corpus (3/3 both arms). L-MIXDD-STARVES-PAIR:
  deep d5/d6 dyck segments (d5 = half a stream, d6 = whole stream)
  collapse the other families' exemplar density — even the P16
  pair-carrying arm drops to 0/3 under the mix.
- FUSION LAW (queue #1 CLOSED): naive equal-rate data fusion cannot
  hold dyck+pair+ratio simultaneously on ANY arm; multi-task fusion
  needs per-family token budgets/scheduling (reserve vanilla-share
  pair/modk/track exemplar density, controlled depth-diverse dyck
  share). Inherited by the VET-LM+corpus (chatbot) fusion pilot.
- OPEN QUEUE (ranked): 1. SCHEDULED-FUSION PILOT: one stream with
  per-family token shares (dyck share capped so d5/d6 segments do
  not dominate; pair/modk/track exemplar density held at vanilla
  levels) x seeds 111/222/333 — target: dyck d12 + pair + ratio
  basins 3/3 in ONE config (the first whole-model multi-family
  certification); 2. in-range parity; 3. associative-capacity
  probe; 4. VET-LM+corpus fusion (chatbot — inherit the scheduled-
  fusion constraint); 5. deeper TF dyck d12-16 (NOT P1-class).

## ARCH-VET P20 (cycle 60) — scheduled-fusion pilot FAILS at L=256: the fusion wall (data level closed)
- P20 (VETDCC-big x 50%-dyck-token-budget schedule @4000, seeds
  111/222/333): dyck d12 .421/.858/.371 = 1/3 (mean .55; d2-5
  in-train .94-.98, the >d5 carry is a seed lottery), pair
  .679/.566/.585 = 0/3 (mean .610), ratio .955/1.009/.825 = 0/3
  (mean .930), modk 1.0 3/3. L-SCHEDULED-FUSION-L256-FAILS.
- MECHANISM FINDINGS: (a) dyck-d12 robustness REQUIRES d6 in train
  (d5 ceiling -> 1/3 lottery; d6 in train -> 3/3 in P18/P19/P19c);
  (b) pair is exemplar-density-limited under deep-dyck streams
  (0/3 mean .610 vs vanilla 3/3 .767); (c) ratio <=.6 is broken by
  ANY deep-dyck training share (0/3 mixdd AND 0/3 sched-50%) vs
  vanilla 3/3 — ratio and depth-diverse dyck are mutually exclusive
  in one L=256 stream (extends L-MIXDD-BREAKS-RATIO).
- FUSION WALL (queue #1 CLOSED at the data level): dyck d12 needs
  d6-in-train (~>=50% of a 256 stream), pair needs vanilla exemplar
  density, ratio needs a near-vanilla dyck share — pairwise
  contradictory in ONE shared stream; no data-level schedule
  reconciles them. FORK DECISION (per C60 plan): fusion must be
  ARCHITECTURAL (per-family routes/experts/compartments in a shared
  host — C22b modular precedent), not data-level mixing.
- OPEN QUEUE (ranked): 1. C61 = modular-fusion pilot design:
  per-family route/compar- tment architecture (VET core shared,
  family-specific registers/readouts or gated expert lanes) that
  trains each family in its certified regime and routes at
  inference — target the unified multi-seed certified row; 2. the
  residual L>=1024 schedule cell (d6 @ ~25% of a long stream; ~12h;
  suspect given finding (c) — run only in a hosting session); 3.
  in-range parity; 4. associative-capacity probe; 5. VET-LM+corpus
  fusion (chatbot — inherits the fusion wall: modular or
  long-stream only); 6. deeper TF dyck d12-16 (NOT P1-class).

## OPERATIONAL (cycle 61) — parallel harness: the box has 2 cores, use them
- MEASURED: 2 concurrent single-threaded runs each take 29.0 s for 40
  steps (identical to solo; wall 31 s, user 60 s = real 2-core
  parallelism, zero degradation); 3 concurrent -> 45.6 s each
  (57% slower) = do not exceed 2. Hardware: nproc=2, 3.9 GB RAM
  (the 1-CPU/2 GB directive is a conservative envelope).
- arch_vet_runner.py: per-run JSON durability (runs/<id>.json),
  resumable `drive --jobs 2`, RESULT merge. Runs remain
  single-threaded, seed-hygienic and bit-identical to the certified
  protocol -> comparability preserved; only scheduling changed.
- EVAL-COST: eval_dyck d1-12 (L to 16396) ~= 4x a 30-step train ->
  screening runs should use a trimmed ladder (d1-8) first.
- Consequence: 3-seed certifications ~2.9 h -> ~1.6 h; reclaims the
  ~50% of machine capacity idle for C51-C60. No certified number
  changes.

## ARCH-VET P21/P21B (cycle 61) — MODULAR FUSION BREAKS THE DATA-LEVEL WALL; unified row 3/4 bars; router patch identified
- P21 (43,074p = VETDCC-big on the vanilla P9 pool @4000 + STACKDCC2-big D12 on the P13D depth-diverse corpus @2000, causal structural router, seeds 111/222/333): UNIFIED ROW pair .8585/.8774/.8585 (3/3), modk 1.0 (3/3), dyck d12 .9838/.9496/.9728 (3/3, routed = B-level) — but CE ratio 1.058/1.238/0.884 (0/3). JOINT heterogeneous pass (mixdd, the corpus that starved every monolith): track 1.0 | modk 1.0 | pair 1.0/.9936/1.0 | dyck_close .9908/.8924/.8491 SIMULTANEOUSLY (joint_both_ok 2/3 at pair>=.717 & dyck>=.85). L-MODULAR-FUSION-BREAKS-THE-WALL: per-family certified training corpora + causal dispatch recover axes that are pairwise contradictory in one stream (arXiv 2409.14981: modules need task-specific data to specialize).
- P21B (router/CE diagnostic, 5 variants): ratios 1024/256hard — routed 0.789/1.070/0.810 (0/3), A-only 0.457/0.538/0.448 (3/3), B-only 3.606/3.131/2.216 (fail), DEEP-ROUTE (dyck body to B only when segment depth>=3) 0.475/0.559/0.464 (3/3), ensemble 0.622/0.788/0.604 (marginal). DECOMPOSITION: at 1024/easy (depth-2 dyck) A is far better at dyck targets (0.49-0.52 vs B 1.22-4.28) because depth 2 has weight 1/15 in B's d-1 curriculum; at 256-hard (depth-3) B is better (2.80-3.33 vs A 4.47-7.32). => L-DEPTH-AWARE-DISPATCH: the router must be depth-sensitive (B only for depth>=3), which restores the invariance bar WITHOUT retraining.
- NEXT (P21C, in flight): re-evaluate the unified row + joint evals with the causally-implementable depth-threshold router (T=3 primary, T=2 secondary) -> target the unified row 4/4 bars 3/3.

## ARCH-VET P21C (cycle 61 close) — THE UNIFIED ROW: 4/4 certified bars, 3/3 seeds, one 43,074p system
- deep-router T=3 (causal depth-threshold dispatch, no retraining):
  pair .8585/.8774/.8585 (3/3), modk 1.0 (3/3), length-invariance
  ratio .450/.527/.442 (3/3 <=.6), dyck close d12 .9838/.9496/.9728
  (3/3 >=.85) => bars_passed_per_seed [4,4,4]. JOINT heterogeneous
  pass (mixdd): pair 1.0/.9936/1.0 WITH dyck .916/.8911/.8386 in the
  SAME pass (T=2: joint dyck 3/3 but ratio 2/3).
- L-MODULAR-FUSION-BREAKS-THE-WALL (P21+P21C): the axes that are
  pairwise contradictory in one stream are simultaneously certified
  by two per-family-certified experts + causal dispatch, at 43k
  params with no extra training beyond the experts.
- L-DEPTH-AWARE-DISPATCH (P21B): the dispatch boundary must follow
  the competence boundary (depth >= 3 -> specialist; depth < 3 ->
  generalist), because the deep specialist is weak at depth 2
  (curriculum weight 1/15) while the generalist is strong there.
- HONEST BOUNDARIES: router is HAND-SPECIFIED (learned-router
  ablation = C62 #1); depth-3 seam in the ladder (close_d3 .60-.76
  at T=3); track eval noisy (.95/.77/.35, never certified); baseline
  still sin-PE TFMicro (RoPE/length-generalizing control = C62 #2);
  3 seeds only.
- OPEN QUEUE (ranked): 1. LEARNED ROUTER (replace the structural
  router with a trained/distilled gate; if it approaches the
  structural router, the mechanism claim stops depending on injected
  grammar knowledge); 2. length-generalizing/RoPE TF control at
  matched params (claim integrity for "beats the Transformer");
  3. 10-seed certification of the unified row (tighten basin
  estimates, report CIs); 4. close the depth-3 seam (per-position
  threshold or 3-way gate); 5. CHATBOT expert C on the corpus in the
  SAME modular frame (the fluency axis; the modular path is now the
  proven route); 6. in-range parity; 7. associative-capacity probe;
  8. L>=1024 data-level schedule cell (suspect, low priority).

## ARCH-VET P22 (cycle 62) — LEARNED ROUTER: the unified row survives without injected grammar knowledge
- Learned causal token-only GRU gate (1,410p, one-hot(x_t) -> GRU(8)
  -> 2 logits) over FROZEN 43,074p experts, trained by the composed
  probability-mixture CE on a joint vanilla+mixdd+deep-dyck pool
  (no labels/tags/depth features), hard argmax dispatch:
  unified row bars_passed_per_seed [4,4] (pair .8585/.8774, modk 1.0,
  ratio .596/.503, dyck d12 .9838/.9496); joint pass pair 1.0/.9936
  WITH dyck .9908/.9108 (=/+ structural T=3). Gate share: vanilla
  .03/.005, mixdd .54/.53, deep-dyck .98/.98.
- L-LEARNED-DISPATCH-SUFFICES: the modular-fusion result does not
  depend on injected grammar knowledge; a 1.4k causal gate trained by
  the loss discovers the depth-diverse dispatch policy. It also
  REPAIRS the P21C depth-3 seam (close_d3 .913/.862 vs .605/.757).
- RECIPE (program-level, now claim-clean): per-regime certified
  experts + a tiny learned causal dispatcher.
- BOUNDARIES: 2 seeds; ratio marginal on s111 (.596); in-family
  dispatch only (no distribution-shift routing); experts frozen (no
  end-to-end co-adaptation); sin-PE TF baseline unchanged.
- OPEN QUEUE (ranked): 1. CHATBOT EXPERT C in the modular frame
  (fluency axis — the modular route is proven, so an expert on the
  corpus + the same learned gate is the direct test; honest
  L-DATA-CEILING boundary applies); 2. length-generalizing/RoPE TF
  control at matched params (claim integrity vs "beats the
  Transformer"); 3. 10-seed certification of the unified row with
  the learned gate (fix the marginal ratio estimate, report CIs);
  4. end-to-end co-adaptation (unfreeze experts + gate) vs the
  frozen-expert boundary; 5. in-range parity; 6. associative-capacity
  probe; 7. L>=1024 data-level schedule cell (low priority).

## ARCH-VET P23 (cycle 62) — FAIR TRANSFORMER CONTROL closed
- 2L d48 4h Transformers (42,672p) with NoPE / ALiBi / NAPE, trained
  6000 steps on the union of both expert corpora (matched params,
  data, steps vs the unified system): bars [0,0] / [1,1] / [0,1].
  Unified learned-gate system: [4,4]. No TF run holds 2 bars together.
- modk (long-span counting) is the sharpest separator: TF .10-.27 vs
  VET 1.0. ALiBi gives length-invariant CE and dyck d12 .94 (1 seed)
  but kills pair recall (.02-.13).
- L-FAIR-TF-CONTROL: the Transformer gap is not a PE artefact.
- Eng: torch MHA eval fastpath corrupts per-head float masks
  (disable it); exact chunked attention for L>2048.
- OPEN QUEUE: 1. modular TF-of-experts with the learned gate
  (modularity vs substrate); 2. chatbot expert C; 3. 10-seed
  certification; 4. end-to-end co-adaptation; 5. in-range parity.

## ARCH-VET P24 (cycle 62) — MODULARITY vs SUBSTRATE closed
- Transformer-of-experts (TF-NAPE d32 vanilla@4000 + TF-ALiBi d32
  deepmix@2000 + the SAME learned gate; 41,954p): bars [1,1] —
  dyck d12 .952/.952 passes (modularity win shared by TF), pair
  .54/.63, modk .40/.21, ratio .97/1.34 fail. VET-of-experts [4,4].
- L-SUBSTRATE-NOT-MODULARITY: pair / modk / length-invariance are
  VET substrate properties; dyck at depth is a modularity property.
- Groundbreakingness conditions: fair control ✓ (P23+P24), learned
  dispatch ✓ (P22), one unified model ✓ (P21C/P22), third-party
  reproducibility ✗ (next: 10-seed + CIs + reproduce.sh).
- OPEN QUEUE: 1. reproduce.sh + 10-seed certification of the unified
  learned-gate row with CIs; 2. chatbot expert C in the modular frame;
  3. end-to-end co-adaptation; 4. in-range parity (256-hard CE).

## ARCH-VET P25 (cycle 62 close) — 10-SEED CERTIFICATION
- Unified learned-gate system, 10 seeds: bars [4]x10, all-4 rate
  10/10 (Wilson 95% [.722, 1]); pair .889+-.049, modk 1.0, ratio
  .499+-.051, dyck d12 .968+-.016; joint pair .999 WITH dyck .942.
- L-UNIFIED-ROW-10-SEED. reproduce.sh = single-command third-party
  reproduction (verify 35/35 -> P25 -> summary).
- All four groundbreakingness conditions closed (P22/P23/P24/P25).
- OPEN QUEUE (C63): 1. chatbot/fluency expert C in the modular frame
  (the end-form "exact + fluent" requirement; inherits L-DATA-CEILING);
  2. in-range parity (256-hard CE); 3. end-to-end co-adaptation;
  4. distribution-shift routing (gate on unseen stream mixtures);
  5. associative-capacity probe.

## ARCH-VET P26 (cycle 63) — FLUENCY EXPERT C (byte GRU) + 3-way learned gate
- Routed text CE == C-alone (2.786/2.674; gate 99.6% C on text, 0% C
  on symbols); chatmix one-pass reasoning modk 1.0, pair .98/.99,
  dyck .985/.895, track .91/.95; best text model vs TF-NAPE d56
  (3.17/3.14) and GRU d64x2 (3.34/3.40) monoliths at ~matched params.
- L-MODALITY-BOUNDARY-IS-FREE. NEGATIVE: L-FLAT-GATE-BLURS-DEPTH —
  the 3-way gate regresses the length ratio to 1.0 (bars 3/4).
- chatmix reasoning metric is in-range (does not separate the GRU
  monolith); OOD bars for 304-vocab controls still owed.
- OPEN QUEUE (C63): 1. P26B hierarchical gate (modality gate over the
  certified 2-way gate) -> restore 4/4 + fluency; 2. OOD bars for the
  fluency controls; 3. 10-seed of the 3-expert system; 4. in-range
  parity; 5. co-adaptation.

## ARCH-VET P26B (cycle 63) — HIERARCHICAL GATE: end form reached at PoC scale
- Modality gate M (3.7k) over frozen certified gate G: symbolic
  dispatch identity 1.0; guard 4/4 both seeds; text CE == C-alone;
  chatmix reasoning modk 1.0 / pair .99 / dyck .90-.99.
- 97,950p single system = exact + fluent. L-HIERARCHICAL-GATE-
  PRESERVES-CERTIFICATION.
- OPEN QUEUE (C64): 1. 10-seed of the 3-expert system (C_s* for 8
  more seeds ~4 min each + M); 2. OOD bars for the 304-vocab
  monolith controls; 3. in-range parity; 4. co-adaptation;
  5. generation-quality probe (sampled text + exact answers in one
  transcript).

## ARCH-VET P27/P27B (cycle 64) — exact+fluent system 10-SEED CERTIFIED
- 3-expert hierarchical system: 4/4 bars on 10/10 seeds; dispatch
  identity 1.0; text CE 2.724+-.045 == C-alone; chatmix modk .996,
  pair .988, dyck .919+-.068; free-running generation exact .88-.91
  (modk 1.0, pair .93).
- Controls (304 vocab, OOD bars): TF-NAPE d56 0/0 bars; GRU d64x2
  2/1 bars (in-range generation .86-.91 = matches; OOD fails).
- L-EXACT-FLUENT-10-SEED; L-IN-RANGE-BLIND.
- OPEN QUEUE (C65): 1. in-range parity (256-hard CE 2.5-2.9 vs TF
  ~2.0 — the only axis where TF still leads); 2. end-to-end
  co-adaptation (unfreeze experts under the hierarchical gate);
  3. track family: certify or retire; 4. distribution-shift routing;
  5. write-up: results-per-compute table (all runs < 45 min on 2
  cores, whole 10-seed cert ~4 h).

## ARCH-VET P28 (cycle 65) — in-range parity DIAGNOSED and re-scoped
- 256-hard CE decomposed: det (answers/closes, 5.3% of positions) VET
  .98-1.31 vs TF-NoPE 1.73 / ALiBi 3.33; stoch (94.7%) VET 2.41-2.82
  vs NoPE 2.10 — the entire TF lead is filler/gap-end hazard
  extrapolation (2.66 vs 2.94-3.39) + count continuation.
- In range and at 1024 VET wins both halves (det .05/.06 vs .93/2.40).
- L-PARITY-GAP-IS-HAZARD. Axis closed as a reasoning concern; report
  det-CE and hazard-CE separately from now on.
- OPEN QUEUE (C65/66): 1. end-to-end co-adaptation under the
  hierarchical gate (experts unfrozen, small lr) — does it help det
  or damage certification?; 2. distribution-shift routing (gate on
  unseen mixture ratios / longer text turns); 3. depth-3 seam of the
  learned gate (s222 det 2.17); 4. results-per-compute write-up.

## ARCH-VET P29 (cycle 65) — co-adaptation NEGATIVE
- Unfreezing A/B under the frozen gate (lr 3e-4, 600 st, HARD or
  SOFT): det CE not improved (s111 worse 1.31->1.80), pair-OOD eroded
  (s222 .877->.698/.717), only joint dyck +.05. L-FROZEN-EXPERTS-
  ARE-THE-OPTIMUM. Closed.
- OPEN QUEUE (C66): 1. distribution-shift routing (unseen mixture
  ratios, longer text turns, L=1024 chatmix) for the hierarchical
  gate; 2. learned-gate depth-3 seam (s222 det 2.17): gate trained
  with a det-position-weighted loss; 3. results-per-compute write-up
  (all certified results < 45 min/run on 2 cores).

## ARCH-VET P30 (cycle 66) — distribution-shift routing ROBUST
- 7 unseen mixtures (text-heavy, reasoning-only, long turns, L=1024,
  rapid alternation, deep dyck in text, 10% byte noise): modality
  .948-.9985, symbolic identity .959-.9992 (large-n cells), text CE
  leak 0.0000 everywhere, in-stream modk/pair .90-1.0, dyck .83-1.0;
  gate entropy .05-.13. Weakest = boundary re-lock under dense
  alternation (S5). L-GRAMMAR-LOCAL-GATE-IS-SHIFT-ROBUST.
- OPEN QUEUE (C66/67): 1. boundary re-lock: 1-token lookahead-free
  fix = gate input includes a delimiter-reset (test whether M with
  h=8 or an explicit U/EOS reset closes S5 to >=.98); 2. results-per-
  compute write-up + REPRO table; 3. new capability axis (queue item
  from C61: associative/multi-content capacity probe — the register
  holds 1 key/value; test 2-4 simultaneous bindings).

## ARCH-VET P31 (cycle 66) — KEYED REGISTER BANK: MQAR capacity wall broken
- MQAR (n bindings, random-order queries, hard gap 24-48; n=6,8 OOD):
  VETDCC .99/.70/.63/.51/.39/.27 (1-register wall); VET-KRB 1.0 x6
  (s111), .955/1.0/1.0/1.0/1.0/.997 (s222); TF-ALiBi .88/.98/.95/
  .89/.78/.67 (s111) but .61->.27 (s222); TF-NoPE .05-.27.
- KRB = exact integer key->slot addressing + learned read gate (+33
  params). L-KEYED-REGISTER-BANK. Count-extrapolating recall.
- OPEN QUEUE (C67): 1. KRB collision/overflow (16 keys / 8 slots)
  and multi-token keys; 2. fold KRB into expert A, re-certify the
  unified row 10-seed with MQAR as a 5th bar; 3. S5 boundary re-lock;
  4. results-per-compute write-up.

## ARCH-VET P32 (cycle 67) — KRB stress: collision / overflow / composite keys
- Collision (16 keys/8 slots): KRB .93/.91/.82/.82 (n1-4), n8 .69 —
  graceful, > TF-ALiBi at every n. Overflow (8 keys/4 slots): KRB
  ~slots/n physics (.56 at n6), TF wins (.77) — attention's real
  advantage = unbounded state. Composite two-token keys: pair-hash KRB
  1.0/.93/.90/.90, n8 .68 vs TF .40. Two-table soft combine = no gain.
- Budget confound: 3000 steps (P31 = 4000) -> R1 .93 not 1.0.
- OPEN QUEUE (C67/68): 1. TAGGED KRB (key tag per slot, exact
  tag-match read) at 4000 steps on R1/R2; 2. fold KRB into expert A
  + unified 10-seed re-cert with MQAR bar; 3. S5 re-lock; 4. write-up.

## ARCH-VET P33 (cycle 68) — TAGGED KRB (cuckoo 2-choice + verified read)
- P32 DATA BUG: key/value tokens overlapped filler/ONE/MANS; all P32
  rows (incl. TF) are lower bounds only. Fixed; P33 is on clean sets.
- R2 collision 16k/8s: KRB-TAG 1/1/.99/1.0, n6 .946, n8 .865 = oracle
  ceiling; KRB .854/.764 ~ TF-ALiBi .829/.754 (2x params). Tagging
  closes the collision gap. R4 composite keys: .800/.638 (ceiling).
- R3 overflow 8k/4s: .728/.591 at n6/n8 — slots/n physics, unchanged;
  attention's unbounded state remains its real advantage.
- Single seed 111. OPEN QUEUE (C69): 1. fold KRB-TAG into expert A +
  unified 10-seed re-cert incl. MQAR bar; 2. S5 re-lock; 3. results-
  per-compute write-up; 4. RoPE/ALiBi TF control on the legacy axes.
