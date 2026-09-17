# RESULTS — results-per-compute summary (through cycle 73)

Hardware envelope for EVERY number below: one sandbox, 2 CPU cores, ~4 GB RAM,
no GPU, PyTorch single-threaded (`OMP_NUM_THREADS=1`), two runs at a time.
Largest model ever trained: 116,858 parameters. Typical arm: 4,000 steps of
batch 8 x 256 tokens (~8.2M tokens, 40-60 min wall). Everything is reproducible
from `arch_vet_p*.py` + `p21_ckpt/` + `log.jsonl` (RESULT rows, one per run).

## 1. The system (one parameter set, 116,858p)

| expert | params | mechanism | trained on |
|---|---|---|---|
| A  VETDCC | 21,257 | value/exact-transition controller + counters + register | vanilla 4-task stream, 4000 st |
| B  STACKDCC2 D12 | 21,817 | A + explicit push/pop stack organ | deep-dyck mix, 2000 st |
| C  ByteGRU | 43,600 | byte-level GRU fluency expert | 1 MB text, 4000 st |
| K  KRB-SEEN | 21,324 | tagged keyed register bank, **learned** write predicate (hindsight self-supervised), cuckoo 2-choice placement, verified read | MQAR R2 stream, 4000 st |
| G / GM / GK | 1,410 / 3,713 / 3,737 | causal token-only GRU gates, hierarchical (certified 2-way -> +modality -> +memory) | composed-mixture CE, 1000-1500 st each, experts frozen |

No expert uses attention. Per-token cost is O(1) in sequence length; state is
a fixed set of registers/counters/stack/bank.

## 2. Certified bars (all OOD relative to training: longer streams, harder gaps, deeper nesting, more bindings)

| bar | definition | system (seeds) | best matched Transformer control |
|---|---|---|---|
| pair | key-value recall, hard gap, >= .717 | .86-.88, **10/10** seeds (P25/P27) | NAPE .87 on 1 of 2 seeds; ALiBi .02-.13 |
| modk | count-mod-k over long span, = 1.0 | 1.0, **10/10** | .10-.27 all TF arms |
| ratio | CE(L=1024)/CE(256 hard) <= .6 | .43-.60, **10/10** | ALiBi .49-.62, NoPE/NAPE .84-1.45 |
| dyck d12 | close-bracket acc at depth 12 (train d<=6) >= .85 | .95-.98, **10/10** | ALiBi .94 (1/2 seeds); TF-MoE .95 |
| text CE | bytes/char on held-out text | 2.67-2.79 == C-alone, 10/10 | TF-NAPE 85.9k: 3.14-3.17 |
| **MQAR n4 / n8** | multi-query recall, 16 keys / 8 slots, 4 (in-range) and 8 (2x OOD) bindings, >= .90 / .70 | .919 / .723, **10/10** (sd .008/.010) | TF-ALiBi 42.7k: .944/.754, .375/.274, .425/.247 (1/3 basin) |

Transformer controls (P23/P24/P27): 2L d48-56, NoPE / ALiBi / NAPE positional
schemes, matched or larger params, union corpus, combined step budget. Best any
single TF run achieved on the four legacy bars: **1 of 4**; a TF-of-experts with
the identical gate recipe: 1 of 4 (dyck only). The system: 4/4 on 10/10 seeds.

## 3. Compute ledger for the headline row

| item | wall (2 cores) | tokens |
|---|---|---|
| A + B + C + K per seed | ~3.5 h | ~30M |
| G + GM + GK per seed | ~35 min | ~6M |
| full 6-bar evaluation per seed | ~10 min | — |
| **one certified seed** | **~4.3 h** | ~36M |
| adding expert K to a certified 3-expert system | 12 min gate training + 0 retraining | 2M |

## 4. What was falsified along the way (banked negatives, all in log.md)

- Data-level fusion of families (one monolith) — fusion wall, closed (C58-C60).
- Soft/softmax slot memories (VETDCC-style) for MQAR: .35 at n4 (P31).
- Two-table soft combine over the register bank: inert (P32).
- Straight-through binary write gates through an exact tag match: collapse open
  or closed regardless of regulariser (P34, P34b).
- Learnable read gate in front of an exact bank: never bootstraps (P35b-d).
- Overflow beyond slot count: slots/n physics, attention's genuine advantage (P32/P33).
- P32 token-set overlap bug: all P32 rows are lower bounds only (disclosed C68).

## 5. Open (honest)

- Learned write predicate is .13 below the hand-wired ceiling at n8 (.73 vs .865).
- In-range CE parity with the TF is not reached (256-hard CE 2.2-2.5 vs 1.96).
- Fluency is corpus- and capacity-bound (1 MB, 43.6k params).
- ~60 empirical "laws", no theory. No inference-latency or robustness study.
- Third-party repro: run `python3 verify_suite.py` (35 exact-match checks) then
  any `arch_vet_p36.py --seeds 111` (needs `p21_ckpt/`).
