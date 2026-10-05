#!/bin/bash
cd /home/user/TryArena
while pgrep -f "arms HW --seed 222" >/dev/null; do sleep 30; done
python3 arch_vet_p42.py --arms LP --seed 222 --p0 0 --live --tau1 0.5 --l1 0.002 --tagsuffix C84 >> p42_c84b.log 2>&1
python3 arch_vet_p42.py --arms HW --seed 111 >> p42_hw111.log 2>&1
