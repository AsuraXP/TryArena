#!/usr/bin/env bash
# Third-party reproduction of the certified unified row (ARCH-VET-LM-P25).
# Box: 2 CPU cores, 4 GB RAM, no GPU. torch CPU wheel from PyPI.
# Total wall ~6.5 h for 10 seeds (2-concurrent expert training); pass
# --seeds 111,222 for a ~1.5 h smoke.
set -euo pipefail
cd "$(dirname "$0")"
python3 -c "import torch" 2>/dev/null || pip install --break-system-packages torch numpy
python3 verify_suite.py | tail -n 1                    # 35/35 exact-match
python3 -u arch_vet_p25.py "$@"                         # experts -> gate -> 10-seed row + CIs
python3 - << 'PY'
import json
r=[json.loads(l) for l in open("log.jsonl") if '"ARCH-VET-LM-P25"' in l][-1]["summary"]
print("bars_passed_per_seed", r["bars_passed_per_seed"]); print("all4", r["all4"])
for k,v in r["metric_stats"].items(): print(f"{k:11s} mean {v['mean']:.4f} sd {v['sd']:.4f} [{v['min']}, {v['max']}]")
PY
