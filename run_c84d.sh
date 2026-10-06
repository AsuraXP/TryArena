#!/bin/bash
cd /home/user/TryArena
python3 arch_vet_p42.py --arms LP --seed 222 --p0 0 --live --tau1 0.5 --tagsuffix C84b >> p42_c84b_s222.log 2>&1
python3 arch_vet_p42.py --arms HW --seed 111 >> p42_hw111.log 2>&1
