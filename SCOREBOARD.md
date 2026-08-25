# ARC-2 SCOREBOARD — sandbox-trained KR automata vs frozen judge suite

Machine total: **35/35** exact-match

| item | machine | frontier-LLM (operator to fill) |
|---|---|---|
| T2-1 | PASS | pro ✗ · flash ✗ |
| T2-2 | PASS | pro ✓ · flash ✗ |
| T2-3 | PASS | pro ✗ · flash ✓ |
| T2-4 | PASS | pro ✗ · flash ✗ |
| T4-1 | PASS | pro ? · flash ✓ |
| T4-2 | PASS | pro ? · flash ✓ |
| T4-3 | PASS | pro ? · flash ✓ |
| T4-4 | PASS | pro ? · flash ✗ |
| T1-1 | PASS | pro ✓ · flash ✗ |
| T1-2 | PASS | pro ✓ · flash ✗ |
| T1-3 | PASS | pro ✓ · flash ✓ |
| T1-4 | PASS | pro ✓ · flash ✗ |
| T1-5 | PASS | pro ✓ · flash ✓ |
| T1-6 | PASS | pro ✓ · flash ✓ |

| T3-1 | PASS | pro — · flash ✗ |

| T3-2 | PASS | pro — · flash ✗ |

| T3-3 | PASS | pro — · flash ✗ |

| T3-4 | PASS | pro — · flash ✗ |

| T3-5 | PASS | pro — · flash — |

| T5-1 | PASS | _pending_ |

| T5-2 | PASS | _pending_ |

| T5-3 | PASS | _pending_ |

| T5-4 | PASS | _pending_ |

| T6-1 | PASS | _pending_ |

| T6-2 | PASS | _pending_ |

| T6-3 | PASS | _pending_ |

| T7-1 | PASS | _pending_ |

| T7-2 | PASS | _pending_ |

| T7-3 | PASS | _pending_ |

| T9-1 | PASS | _pending_ |

| T9-2 | PASS | _pending_ |

| T9-3 | PASS | _pending_ |

| T8-1 | PASS | _pending_ |

| T8-2 | PASS | _pending_ |

| T8-3 | PASS | _pending_ |

Certification seeds used: {'t2': 0, 't4': 0, 't1': 0}
Total params (3 models): 3577
Wall: 174s · peak RAM 310MB · 1 CPU

## FRONTIER JUDGING — operator round 1 (2026-08-23)
Models judged by operator (paste of judge cards): **gemini-3.1-pro** (latest)
and **gemini-3.5-flash thinking-extended**. Items answered: T1 (both), T2
(both), T3-1..4 (flash only), T4 (flash; pro ambiguous). Flash's own item
labels were off-by-one in the paste; matched by content order.
- pro:  T1 6/6 · T2 1/4 · T4 ambiguous (trailing "0 0 0 0" / "4 2 1 3"; if
        T4-1..4 then 3/4, T4-4 wrong) — recorded as "?" until clarified.
- flash: T1 3/6 · T2 1/4 · T3 0/4 (all mid-digit corruption) · T4 3/4.
- INTER-TEST ERROR (T3-2/T3-3): pro's multiplication answers are not yet
  recorded; flash's differ from key in middle digits.
Machine column unchanged: 29/29 on these items.


## P4-DISC — open-ended iteration discovery (2026-08-23, certified)
Machine: crisp 14-token/16-state Mealy tape machine, c24k_crispfix.pt.
Counter protocol discovered by contract-decomposed search (2 edits from
identity); digit increment (+1 mod10 LSB-first, carry persistence) learned
by crisp-STE SGD from the discovered seed; 1204 steps. No TF arm (waived).
| test | result |
|---|---|
| in-dist k<=4, L<=12 | 500/500 |
| k=16, L=40 | 200/200 |
| k=64, L=40 (4x unseen) | 100/100 |
| joint k=64 x L=120 | 100/100 |
| pass count = k+1, all scales | exact |
| one-mark trace, all scales | ok |

## C25a — iterated subtraction on the P4-DISC loop (2026-08-23, certified)
Counter organ reused intact (0 rows changed); borrow digit organ learned.
| test | result |
|---|---|
| in-dist k<=4 | 500/500 |
| k=16 | 200/200 |
| k=64 unseen | 100/100 |
| joint k=64 x L=120 | 100/100 |
| pass count = k+1 | exact |

## C22-R — chatbot state machine repair (2026-08-24, cycle 38, CERTIFIED)
Machine v9c champion c22r8.pt (20,518p). Fixed an eval defect (probe oracle
never subtracted iid turn-choice entropy at U positions, H=1.667) and a
mechanism defect (organ emit off-by-one); plus decay-clamp, math-in-state-
family, long-window + overwrite-distance fine-tunes, organ gain x2.4.
| bar | threshold | result |
|---|---|---|
| D1 state @4096 (corrected) | <=0.01 | -0.0651 PASS |
| D2 overwrite @4096 | <=0.05 | 0.0389 PASS |
| D3 state @16384 | <=D1+0.05 | -0.0704 PASS |
| D4 math-plus / math-minus | <=0.02 / <=0.05 | -0.0004 / 0.0002 PASS |
| D5 chat @4096 | <=0.02 | 0.0001 PASS |
| D6 routing | 1.0 | 1.0 PASS |
| D7 dialogue exact | exact | dave/it/1-2/fine/6/4-2 PASS |
Robust: overwrite 0.0387-0.0392 over 6 seeds and at 8192 (2x train len);
state -0.056..-0.089 over seeds. verify_suite 35/35.

## C26-R — binding wall BROKEN: value-encoded transport (2026-08-25, cycle 40)
New machine class VET: control Mealy (5 states) x mechanism-owned value
register. Hand-derived existence proof; discoverability run = cycle 41.
| test | result | previous best |
|---|---|---|
| S1 in-dist nd<=4 | 500/500 | 326/500 (plateau) |
| S2 nd=16 | 200/200 | 9/200 |
| S3 nd=32 | 100/100 | 0/100 |
| S4 joint nd=64 | 100/100 | 0/100 |
| S5 passes=nd+1, one-mark trace | exact | drifted |
| stretch nd=128/256/512 | exact | n/a |
Depth-unlimited by construction. L-VALUE-CHANNEL banked.
Cycle 41: DISCOVERED from blank genome in 877 evals/20s; same genome
certifies all bars. C26 BINDING CERTIFIED (VET class). Laws:
L-VALUE-CHANNEL, L-DISCOVERABILITY-BY-CLASS.
