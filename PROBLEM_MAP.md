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
