# TryArena — ARC-2 research program

**NEW AGENT? READ `HANDOVER_PROMPT.md` FIRST. It is the single
authoritative handover (written at cycle 53 close, 2026-08-30, branch
`arena/01a038ad-tryarena`). Do NOT follow any plan/queue in RESUME.md,
HANDOVER.md, or in old cycle blocks of log.md — those are C42-era
historical documents and their "next cycle" instructions are STALE
(and were superseded by cycles 43-53). After HANDOVER_PROMPT.md, read
the last 3 blocks of log.md and PROBLEM_MAP.md, then run
`python3 verify_suite.py` (must be 35/35) and resume at cycle 54 per
the CYCLE 54 PLAN section of HANDOVER_PROMPT.md.**

- Records: `log.md` (block per cycle), `log.jsonl` (RESULT lines),
  `PROBLEM_MAP.md` (problem -> status -> laws).
- Active axis (C51+): VET-LM architecture, code in `arch_vet_lm.py`
  + `arch_vet_p2..p9.py`.
- Closed axis (C1-C49): certified controller/machine line, artifacts
  `c43..c49_*.py/.pt` and the C22b fused module (`c22b_stage1.pt`).
- Suite: `verify_suite.py` (35/35 exact-match, must stay green).
