# NOVELTY.md — the map, the ledger, the gate
Read before every hypothesis. Update after every cycle. This file exists because the
agent's memory IS the standard path; left alone it will regenerate Mamba, RWKV, NTM,
Hopfield, MoE, sparse attention, and our own dead ends with new names.

## A. THE GATE (every hypothesis passes all five or is not run)
1. **Locate it on the map** (section B). Which cells does it occupy? Are they walked?
2. **Search to DISPROVE novelty** — arXiv / GitHub / blogs, 2+ queries aimed at finding it
   already done. Cite in the code header. If found: write the one-sentence DELTA
   ("mine differs because ___ and that is what removes limitation ___"). No delta → rejected.
3. **Check the negatives ledger** (section C) by FAILURE REASON, not by name. If the
   predicted way this could fail matches a banked reason → rejected or redesigned.
4. **Generate ≥4 candidates, kill the familiar.** Kill every one in a known family or
   banked reason. Run the survivor. If none survive, generate again; do not lower the bar.
5. **Pre-register** in log.md before running: "if right, SLOPE moves by ≥ X on task Y
   at sizes Z; if wrong, we learn ___." A result with no pre-registered prediction is
   exploration, not evidence.

## B. THE MAP — axes of a learning core
Mark: [F] walked by the field (cite + why it stalled) · [U] walked by us (log pointer) · [ ] unwalked.
Pick from [ ]. Combining two [F] cells is not novelty unless the combination removes a named limit.

| Axis | Cells |
|---|---|
| **Learning rule** | [F] backprop-through-time · [F] truncated BPTT · [F] Hebbian/fast-weights (Schlag, Schmidhuber: capacity-bound, no error signal) · [F] local/forward-forward (Hinton 2022: weak scaling) · [F] predictive coding · [ ] rule discovered by invariance (weights change only when a prediction is contradicted by a contradiction-type signal) · [ ] two-timescale rule (slow weights learn the *law*, fast state carries the *case*, each with its own error) · [ ] learning signal from *consistency across paraphrases/lengths*, not next token |
| **What the state is** | [F] dense vector (RNN/SSM: forgetting is geometric) · [F] KV cache (TF: cost ∝ N) · [F] external tape (NTM/DNC: unstable addressing) · [F] sparse slot delta memory (SDM 2026, Lattice 2025: write every token, soft superposition) · [U] exact hash/stack/counter organs (C40–C93: exact but *outside* the learner) · [U] **state = learner-created bindings (SBC, C95: exact, length-free x23, count-free, 5k params — but a bistable write-event basin; selectivity needs scarcity)** · [ ] state partitioned into "law" (slow, shared) and "situation" (fast, episodic) inside one network |
| **State update** | [F] gated linear (LSTM/GRU/Mamba) · [F] additive outer product (linear attention/DeltaNet) · [U] hard routed update, one organ per step (U3) · [ ] update that is *idempotent* for repeated facts and *overwriting* for contradicted ones, learned not wired · [ ] update whose arity is chosen by the input (0 / 1 / overwrite) via a learned, invariance-trained gate |
| **What the objective pays for** | [F] next-token CE (buys interpolation; our L-PREDICATE-NONIDENTIFIABLE-IN-RANGE) · [F] contrastive · [F] span/denoising · [U] CE + exact consumer (buys predicates only given a consumer, C85) · [U] *invariance loss* on a dense GRU state (C94: satisfied, changes nothing — L-BINDING-NEEDS-A-PLACE) · [ ] *counterfactual consistency*: swap a fact in context, require the swap to propagate to every dependent prediction · [ ] loss on *what the model declines to predict* (abstention is scored) |
| **Credit assignment** | [F] global gradient · [F] REINFORCE-style on discrete choices · [U] straight-through on routes (U3) · [ ] credit routed through the *binding* that was used (a prediction error charges only the fact it read) |
| **Discrete vs continuous** | [F] all-continuous · [F] VQ bottlenecks (codebook collapse) · [U] hard routes + exact organs · [ ] continuous *inside* a binding, discrete *between* bindings, with annealed hardness tied to invariance score |
| **The unit of binding (NEW AXIS, C96)** | [F] tokens/bytes given by a tokenizer · [F] HM-RNN learned boundaries feeding a hierarchy (ST estimators, no cardinality control) · [U] 2-byte window taps (P47: exact on synthetic, useless on text — L-BINDING-NEEDS-A-UNIT) · [U] controller state as key (kq=h: 0/2, agreement not discoverable) · [U] chunker reset = key (C97 .186/.184 boundaries; +surprise C98 .31/.18; prefix-key C99 boundary→0) · [U] fixed window key W=6 (C100, M32/M512 no change: context-polluted) · [U] content-gated window (C101: pass factors never cut; −10% vs GRU) — **L-UNIT-IS-NOT-LEARNED-FROM-SPARSE-NEED** · [U] **unit under dense consumer pressure** (C102: reset boundary becomes word-like .629/.057; unit formed, but chunk-summary values cannot replay unseen strings) |
| **Binding identity / key agreement (NEW AXIS, C102b)** | [U] full recurrent context as key (C95: write/read never agree) · [U] context-free prefix as key (C99/C102b: agrees but role collisions) · [U] prefix + 4D role bottleneck (C103: first/re-mention prefix state agrees exactly; role distance .2-.5, but allocation/consumer defects dominate) · [ ] identity as the *minimal counterfactual context*: retain only features whose swap changes the dependent prediction |
| **Allocation / preservation of learned bindings (NEW AXIS, C103)** | [F] NTM/DNC usage/LRU allocation (external memory, unstable joint addressing) · [U] soft emptiest-slot (C103: tied empties smear every value) · [U] nearest-initial-key hash (C103b: collision overwrite) · [U] soft match/LRU (C103c: match collapse) · [U] ST hard match/LRU + distance read (C103d/e: mechanically preserves .95 and retrieves .66-.72; diagnostic scaffold, not the deliverable) · [ ] learned state update whose prediction error charges and repairs only the binding actually read |
| **What slow weights are forbidden to memorize (NEW AXIS, C103d)** | [U] fixed entity inventory lets h/head bypass episodic state (C103d: exact target read still ignored) · [U] fresh case identities per dialogue force fast-state use (C103e: unseen CE 3.612→2.756) · [ ] make law/case separation structural: slow weights invariant to counterfactual identity permutations, fast state solely carries the case |
| **Read-to-prediction coupling (NEW AXIS, C103e)** | [U] arbitrary linear head over [h,r] learns to ignore r under any shortcut (C103d) · [U] even exact target embedding through that head gives CE 2.26 after random-name training (C103e) · [ ] prediction in the same learned geometry as the binding value (C103f tied copy-energy running) · [ ] consumer error routed back through only the used binding |
| **How the unit is credited (NEW AXIS, C106)** | [W] WALL: unit learned through the CE gradient that flows through the reads it enables (C97–C105: five wirings, all collapse; loop: reads cannot help until keys agree, keys cannot agree until b is right) · [F] C106: consumer-credited boundary (REINFORCE, reward = key recurrence x read gain) — CE 2.252 but the policy stopped cutting; C106b ablation with fixed random resets gives 2.296: credit was not the cause · [R] C106b: NO boundary — random hard truncation of the key state during training forces the recurrence to learn history-invariant keys (L-TRUNCATION-MAKES-INVARIANT-KEYS); unseen-name CE 2.30 vs GRU 3.61, below oracle-boundary 2.76; slope pending · [U] unit credited by invariance of the learner's own states across occurrences · [U] no unit decision at all: write every position, keep what is read |
| **Where time lives** | [F] position embeddings (TF: length-fragile) · [F] recurrence (RNN/SSM) · [ ] time as *event count per binding* (each fact knows how many contradictions it survived), not position |
| **Sharing across positions** | [F] full weight tying (TF/RNN) · [F] MoE by token · [U] route by context (U3) · [ ] sharing by *role* (the same weights serve "the thing being asked about" whatever its position or surface form) |
| **Self-knowledge** | [F] calibration post-hoc · [F] verbalized uncertainty (unreliable) · [ ] state carries an explicit "no binding exists" that the loss rewards for being right |

## C. NEGATIVES LEDGER — by failure reason (do not re-walk; full detail in log.md)
| Reason | Instances (log laws) |
|---|---|
| A learned unit (boundary/reset/gated window) does not bootstrap from sparse downstream need (~5 bytes/300); five mechanisms, −5…−10% vs GRU | C97–C101 |
| A learned unit can form under dense need but is not enough: a chunk-summary value cannot replay an unseen string | C102, L-VALUES-MUST-BE-REPLAYABLE |
| A binding identity needs chosen context: context-free prefixes collide by role; full-context states do not agree | C102b, L-IDENTITY-IS-UNIT-PLUS-CHOSEN-CONTEXT |
| Soft free-slot allocation cannot break tied empties and smears values; nearest-key hashing / soft match then overwrite under text-scale writes | C103–C103c, L-ADDRESS-MUST-BREAK-TIES |
| Fixed case inventories let slow weights memorize and the output head bypass a correct episodic read | C103d, L-COPY-NEEDS-UNMEMORIZABLE-TRAINING |
| Read/write key agreement must hold by construction (same input -> same key regardless of context); a history-mixing state as key never agrees | C95 kq=h 0/2 (symmetry already broken) |
| Unbroken slot symmetry: identical initial slots -> uniform addressing -> zero gradient to q/k | C95 L-SLOT-SYMMETRY (was misdiagnosed 5 times as a gate problem) |
| Joint writer/reader co-adaptation from scratch under CE is bistable: reader ignores noisy memory -> writer's gradient dies (or, with forced writes, writer never becomes selective because memory never overflows) | C85 joint 0/7, C95 L-WRITE-EVENT-BASIN (4/9 collapsed; warm-up alone -> writes everything) |
| A loss term cannot create structure the state has no place for | C94 INV==GRU |
| In-range CE does not identify a predicate / rule; model interpolates instead | L-PREDICATE-NONIDENTIFIABLE-IN-RANGE, L-JOINT-CONSUMER-BLINDNESS, L-PREDICATES-DONT-LEARN-THROUGH-EXACT-MATCH |
| Dense state forgets geometrically; no size fixes distance | L-STATE-BUDGET, L-TF-BINDING-COUNT-COLLAPSE, P45 C-alone 0/15 |
| Exact organ lifts the metric but the learned part is bypassed (would survive deleting the learner) | all organ wins C40–C93 — **not the deliverable** |
| Capacity/data overfit at micro scale, not architecture | P44 small-corpus d96 (train .62 / val .99) |
| Joint learning from scratch of predicate + consumer: 0/7 | C85 joint arm |
| Transformer ceiling is optimizer-invariant; retesting TF on old axes is waste | L-TF-CEILING-OPTIMIZER-INVARIANT |
| Hash/cuckoo load ceilings, indirection overheads | L-CUCKOO-LOAD-CEILING, L-INDIRECTION-OVERHEAD, L-SMART-KICK-SATURATES-B4 |

## D. WHAT WE KNOW THAT POINTS AT THE CORE (build from these)
- The learner generalized OOD only when the architecture made *rules* cheaper to learn than *cases* (routing + exact state). Principle, not trick.
- Per-token cost flat 256→16k when state is bounded and structured.
- Fair micro-TF loses on identical batches, in-range and OOD, and its ceiling is optimizer-invariant: the defect is in the design, not the training.
- Predicates become identifiable *only when a consumer needs them* — the loss must create a need for the structure.
