#!/bin/bash
# C79 lane A: fresh-stream C for 3 seeds, then unified fold with K=SBK4 + C=F (waits for the SBK4 fold to finish first)
cd /home/user/TryArena
python3 arch_vet_p40.py --seeds 111,222,333 >> p40.log 2>&1
while pgrep -f "K SBK4 --seeds" >/dev/null; do sleep 60; done
python3 arch_vet_p36.py --K SBK4 --C F --seeds 111,222,333 >> p36ksbk4_cf.log 2>&1
