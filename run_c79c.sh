#!/bin/bash
# C79c lane A driver: wait for the SBK4 s1010 K and the CF seeds (lane B), then fold the 7 seeds
cd /home/user/TryArena
while pgrep -f "arm SBK4 --seed 1010" >/dev/null; do sleep 30; done
while pgrep -f "arch_vet_p40.py" >/dev/null; do sleep 30; done
python3 arch_vet_p36.py --K SBK4 --C F --seeds 444,555,666,777,888,999,1010 >> p36ksbk4_cf.log 2>&1
