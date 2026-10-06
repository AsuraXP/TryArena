#!/bin/bash
cd /home/user/TryArena
while pgrep -f "tagsuffix C8[5] " >/dev/null || pgrep -f "seed 222 --p0 0 --tau1 0.5 --freeze_ctrl" >/dev/null; do sleep 30; done
python3 arch_vet_p42.py --arms HW --seed 111 --regime wide --tagsuffix C87 >> p42_c87_hw_s111.log 2>&1
