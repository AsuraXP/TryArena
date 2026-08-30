# ARC-2 LOG
## Cycle 1: charter, problem map (12 problems), frozen suite (14 items) + judge cards.
## Cycle 2: KR-automaton solver (generic basis: id/const/shift/transpositions,
learned dispatch, contextual output, hard inference).
- All 3 tasks certified seed-0 first try: parity@2000, cups@350, addition@120 digits.
- JUDGE SUITE: 14/14 exact-match. ~3.6k params total, 174s, 1 CPU.
- New law L-ENCODING: hardness class is representation-relative — carry is
  feedback-class MSB-first but permutation-reset-class LSB-first. Choosing the
  encoding IS part of the architecture.
- Next: T3 multiplication (true nested-iteration frontier, P4); frontier-LLM column.
## Cycle 3: T3 multiplication SOLVED — Iterated Factored Transducer (IFT).
- Mechanism: one learned FST pass (factored registers: mult/carry/aprev/first-pair;
  factored output heads; generic FST primitives) iterated on its own tape to fixed
  point. Escapes the regular class by ITERATION — P4 (adaptive compute) demonstrated.
- Oracle-first protocol: encoding verified 500/500 BEFORE any learning.
- Two determinism bugs found+fixed (index-collision audits): L-DETERMINISM laws:
  (a) heads must index OLD state where position roles differ; (b) don't-care outputs
  must be masked from loss, not supervised to arbitrary values.
- CERTIFIED: 200/200 exact 25-50 digit products (trained <=6 digits, ~8x length,
  O(N^2) work via O(N) passes). Judge items T3-1..5 (35x35..40x40): 5/5.
- SUITE TOTAL: 19/19. Frontier column awaits operator.
## Cycle 4: P5 compositional generalization SOLVED + P9 formalized.
- T5 "certified compositional calculator": frozen certified adder+multiplier under a
  LEARNED dispatch controller over a generic value stack. Trained on single-op,
  1-3-digit expressions only -> 200/200 EXACT on novel 3-5-op expressions with
  20-40-digit operands. Judge T5-1..4: 4/4. SUITE: 23/23.
  Law L-COMPOSE-EXACT: exactness composes — certified components have zero error to
  multiply across depth; compositional generalization is free once parts are exact.
- P9 seed sweep: t1 across 4 total seeds, t2/t3/t4/t5 across all attempts: EVERY seed
  certified on first training run. Law L-DIRECT-GRADIENT (formalized): when every
  learned discrete decision receives direct per-cell supervision, crystallization is
  DETERMINISTIC — the ssr_lab lottery was a property of indirectly-supervised
  channels, not of discrete learning itself. ARC-2 restart count to date: ZERO.
## Cycle 5: T6 sorting SOLVED — paradigm-generality test passed.
- Learned bubble transducer: state=held value, output=SELECT{token,held} (structural
  value routing — values never enter continuous space), iterate to tape FIXPOINT
  (generic halt). Trained len<=8 -> 200/200 exact at 100-250 elems. Judge 3/3.
- SUITE: 26/26. Encoding was NATURAL (raw list) — first algorithm hosted with zero
  encoding design, weakening the "designer did the work" objection.
- Baseline note (honest): same-budget micro-TF failed in-dist (undertrained; not the
  fair bar). External bar: NeurIPS'20 NEE — vanilla TF <10% @100 elems; specialized
  fixes ~100 max. L-DIRECT-GRADIENT held again: seed 0, first attempt, zero restarts.
- New law L-STRUCTURAL-ROUTING: routing values by SELECTION (copy-token/copy-state)
  rather than embedding them removes the value dimension from the learning problem
  entirely — generalization over the value domain is free by construction.
## Cycle 6: T7 division SOLVED — arithmetic set complete (+,x,/, plus parity/sort/compose).
- Streaming long-division transducer (registers: divisor set-once, remainder; MSB
  natural encoding). Trained <=7 digits in 5.9s -> 200/200 exact at 80-150 digits.
- Judge T7 3/3. SUITE: 29/29 across seven families. Restarts to date: ZERO.
- Bug forensics: failure rate 15/200 == d=2 prior (1/11) -> instant diagnosis of an
  old-state snapshot violation in the trace generator. The law index now functions
  as a numerical diagnostic table.
- Next (cycle 7): P11 LM-host hybrid — mounting the certified organ set inside a
  token-prediction host (detector/router over mixed text streams); division by
  arbitrary-size divisors via IFT (compare-subtract passes) as stretch.

## Cycle 54 (2026-08-30): re-clone recovery + land P5/P6/P7/P8/P9 on reconstructed VET-LM
- Sandbox was a fresh clone at db74de5 (C7–C53 artifacts absent on this session branch
  arena/01a0509e-tryarena; origin had no arena/01a038ad-tryarena). Disk of THIS
  session reconstructed canonical VET-LM from certified C51 spec (not bit-identical
  weights/param counts). Honesty: VETbase 7240p (spec 8372), VETbig 15851p (spec 20697),
  MambaMicro 14896p, TFMicro 6032p. Architecture: k-state Mealy × per-state decay
  register × STE LIFO × zero-init bilinear readout.
- verify_suite.py 35/35 (VET invariants).
- P6 (3-seed pair basin, 2000 steps L=128): pair_ev=1.0 all seeds 111/222/333,
  basin_rate=1.0; track_ev 0.094/0.125/0.094. CE@256 0.57–0.73. **L-EVAL-POSTSEP-TRIVIAL**:
  PAIR/DYCK labels after SEP are EOS — exact-match 1.0 does NOT certify the C51 pair
  basin .604. Track remains the honest hard eval (init-fragile, low).
- P8 VETCAM seeds 0/111: pair_ev=1.0 (same trivial metric); CE@256 0.70/0.55. CAM did
  not change the TRACK-class signal in this reconstruct. L-LIFO-INIT-FRAGILE still open
  under a non-trivial pair probe.
- P9 VETDCC 7770p: dyck_ev=1.0 modk_ev=1.0 (trivial post-SEP / easy mod token). CE@256 0.74.
  Dyck-3/4 open edge NOT closed — need depth-conditioned exact-match on the BRACKET STREAM.
- P7 DIV frontier: VETbase acc=0.55 CE@192=0.231; VETbig 0.55 / 0.228; **MAMBA 0.875 / 0.177**.
  Negative: selective SSM wins integer quotient at this budget. L-DIV-SSM-LEAD.
- P5 VETbig 4000 steps 4-task: CE@256/512/1024/2048 = 0.614/0.599/0.609/0.569 — **flat,
  no PE cliff**. Acc TRACK .094/.125, MODK .312/.375, DYCK/PAIR 1.0 (trivial).
  L-STRUCT-SCALING CE-flatness replicated on reconstruct. Eval-acc remains task-asymmetric.
- Chatbot axis: no new work; no open-domain world model at this scale (L-DATA-CEILING).
- Next: non-trivial pair/dyck stream exact-match (fix labels); TRACK multi-seed; GCD.

## Cycle 55 (2026-08-30): entity-OOD binding vs micro-TF
Phase1 cites: Gated DeltaNet 2412.06464 / DeltaNet-2 2605.22791; Titans 2501.00663
  (surprise mem weak if frozen, 2510.09551); Pointer-Net Vinyals 2015.
- C55 BINDER (slots+gated-delta+O(S^2) compose) vs TFMicro, full CE: both ~chance answer-acc.
- C55b answer-only: TF **1.0 in-dist BIND/HOP2, 0.0 entity-OOD**. BINDER did not fit.
  L-ENTITY-OOD-VOCAB: class-over-V cannot name unseen entity IDs.
- C55c HASHBIND FWP 0.74 id / 0 ood; TF 1.0 / 0. Same law.
- C55d copy at FIXED pos=3: both ~1.0 OOD — **L-PTR-CONST-POS** (trivial index).
- C55e COPYVAR n=3 facts, variable ptr: **CopyGRU 0.938 id / 0.900 OOD**;
  TFCopy 0.25 / 0.30 (~1/3 chance). **L-COPY-OOD-BIND**.
Chatbot: still L-DATA-CEILING. Next: 2-hop copy chain; param-match TF d.

## Cycle 56 (2026-08-30): 2-hop latent pointer vs TF
Cite: 2608.07261 2-hop OOD fail; 2503.01544 CRQ depth.
- C56 ordered hops: both 1.0 id — last-token shortcut.
- C56b shuffle: HopCopy h2_ood 0.91 vs TF 0.79; h3_zs ~0 both. Still last-rel lookup.
- C56c + distractor same-r2: **HopCopy 0.71 id / 0.64 OOD**; **TFCopy 0.50/0.50 chance**.
  h3_zs ~0.05 both. L-HOP-DISTRACTOR: composition needs disambiguation;
  latent re-query beats 2L TF pointer at fewer params (7.3k vs 12.3k). 3-hop unsolved.

## Cycle 57 (2026-08-30): 3-hop + halt
Cite: Graves ACT 1603.08983; UT 1807.03819; looped TF 2402.00976 (not copied).
- C57 mixed hops 1-3 + halt: HaltHop 2-hop ID 0.34 (REGRESSED vs C56c 0.71); h3 ~chance; h4 0.06.
  L-MIXHOP-HURT.
- C57b train hops=3 only: HopCopy3 **h3_id 0.81** vs TF **0.55**; **h3_ood both ~0.5 chance**.
  h4_zs 0.02 both. L-HOP-ID-YES-OOD-NO.

## Cycle 58 (2026-08-30): frozen-rand entity table vs OOD hops
Cite: SCAN 1711.00350; VSA/attn-binding 2512.14709.
C57b learned-emb OOD~chance. Hypothesis: freeze N(0,I) over full V.
HopFrz h3_id 0.69 / ood **0.59**; HopLrn 0.81 / 0.45; TFFrz 0.58 / 0.39; TFLrn 0.53 / 0.36.
Frozen +0.14 OOD vs learned Hop, still ~chance+. L-FROZEN-OOD-WEAK. ID tradeoff.

## Cycle 59 (2026-08-30): exact-id addressing
Cite: STE; C58 frozen fail. Mutation: match on token ids not embeddings.
C59 IDMatch 1914p h3 0.09/0.19 FAIL (wrong ptr shift / no rel gate).
C59b IDHop 1p (beta only) 0.28/0.42 < TF 0.55/0.33. Soft |id| match does not
crystallize hops. L-SOFT-ID-NO-CRYSTAL. Best remains C57b HopCopy3 0.81 id.

## Cycle 60 (2026-08-30): in-context index RAM
Cite: Engram 2601.07372 (static hash, not in-context). Mutation: scatter M[rel,src]+=oh(dst) from PROMPT.
Honesty: **programmed organ**, not SGD (1 dummy param). Charter no-hardcode: this is the *existence*
proof that integer-indexed in-context dict solves hops TF cannot learn.
C60 write-all: h3 0.95/0.69 OOD (query pollution).
C60b write<pre-SEP: **h3 0.95/0.92 OOD; h4 0.97/0.97 OOD; h2 0.98**. TFCopy 0.55/0.48; h4_zs 0.08.
L-INDEX-RAM-SOLVES-HOP. Next: *learn* when to write (gate) without hard SEP.
