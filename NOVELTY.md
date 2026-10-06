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
| **What the state is** | [F] dense vector (RNN/SSM: forgetting is geometric) · [F] KV cache (TF: cost ∝ N) · [F] external tape (NTM/DNC: unstable addressing) · [U] exact hash/stack/counter organs (C40–C93: exact but *outside* the learner) · [ ] state = a set of discrete *bindings* the learner itself creates and the loss pays for · [ ] state partitioned into "law" (slow, shared) and "situation" (fast, episodic) inside one network |
| **State update** | [F] gated linear (LSTM/GRU/Mamba) · [F] additive outer product (linear attention/DeltaNet) · [U] hard routed update, one organ per step (U3) · [ ] update that is *idempotent* for repeated facts and *overwriting* for contradicted ones, learned not wired · [ ] update whose arity is chosen by the input (0 / 1 / overwrite) via a learned, invariance-trained gate |
| **What the objective pays for** | [F] next-token CE (buys interpolation; our L-PREDICATE-NONIDENTIFIABLE-IN-RANGE) · [F] contrastive · [F] span/denoising · [U] CE + exact consumer (buys predicates only given a consumer, C85) · [ ] *invariance loss*: same meaning under paraphrase/length/distractors must give same state; one changed fact must change exactly one slot · [ ] *counterfactual consistency*: swap a fact in context, require the swap to propagate to every dependent prediction · [ ] loss on *what the model declines to predict* (abstention is scored) |
| **Credit assignment** | [F] global gradient · [F] REINFORCE-style on discrete choices · [U] straight-through on routes (U3) · [ ] credit routed through the *binding* that was used (a prediction error charges only the fact it read) |
| **Discrete vs continuous** | [F] all-continuous · [F] VQ bottlenecks (codebook collapse) · [U] hard routes + exact organs · [ ] continuous *inside* a binding, discrete *between* bindings, with annealed hardness tied to invariance score |
| **Where time lives** | [F] position embeddings (TF: length-fragile) · [F] recurrence (RNN/SSM) · [ ] time as *event count per binding* (each fact knows how many contradictions it survived), not position |
| **Sharing across positions** | [F] full weight tying (TF/RNN) · [F] MoE by token · [U] route by context (U3) · [ ] sharing by *role* (the same weights serve "the thing being asked about" whatever its position or surface form) |
| **Self-knowledge** | [F] calibration post-hoc · [F] verbalized uncertainty (unreliable) · [ ] state carries an explicit "no binding exists" that the loss rewards for being right |

## C. NEGATIVES LEDGER — by failure reason (do not re-walk; full detail in log.md)
| Reason | Instances (log laws) |
|---|---|
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
