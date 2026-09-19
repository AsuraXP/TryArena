#!/bin/bash
# One-command reproduction of the certified headline (cycle 73/74).
# Levels:  ./reproduce.sh quick   -> verify suite + generator audit + re-score the 10-seed unified system from checkpoints (~15 min, CPU)
#          ./reproduce.sh seed S  -> retrain expert K (learned-predicate KRB) for seed S and re-fold it (~1.3 h)
#          ./reproduce.sh full    -> everything for seeds 111 222 333 (~4 h per seed at 2 lanes)
# Requirements: python3, `pip install --break-system-packages torch numpy`, this repo with p21_ckpt/ (A,B,C,G,GM,GK,P35_R2_SEEN per seed).
set -e; cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
python3 -c "import torch" 2>/dev/null || pip install -q --break-system-packages torch numpy
case "${1:-quick}" in
  quick)
    python3 verify_suite.py | tail -1
    python3 audit_generators.py | tail -1
    python3 - <<'PY'
import json,subprocess,sys
print("[repro] re-scoring unified 5-expert system from checkpoints, seeds 111..1010 (gate GK reused)")
PY
    python3 arch_vet_p36.py --K P37 --seeds 111,222,333,444,555,666,777,888,999,1010 --steps 0 | grep "^\[p36" ;;
  seed)
    S=${2:-111}; python3 arch_vet_p37.py --jobs R2 --arm CF85R --seed $S && python3 arch_vet_p36.py --K P37 --seeds $S ;;
  full)
    for S in 111 222 333; do python3 arch_vet_p37.py --jobs R2 --arm CF85R --seed $S; done; python3 arch_vet_p36.py --K P37 --seeds 111,222,333 ;;
esac
