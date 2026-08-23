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
