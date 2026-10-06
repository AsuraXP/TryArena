#!/bin/bash
# C83 lane B: learned-predicate expert A (P42) after the lane-B folds
cd /home/user/TryArena
while pgrep -f "seeds 888,999,1010" >/dev/null; do sleep 60; done
for S in 111 222 333; do python3 arch_vet_p42.py --arms LP --seed $S >> p42.log 2>&1; done
python3 arch_vet_p42.py --arms HW --seed 111 >> p42.log 2>&1
