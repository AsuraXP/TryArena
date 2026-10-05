#!/bin/bash
cd /home/user/TryArena
python3 arch_vet_p42.py --arms LP --seed 111 --fixed_random --live --tagsuffix C90p1 > p42_c90p1_s111.log 2>&1
python3 arch_vet_p42.py --arms LP --seed 111 --p0 0 --tau1 0.5 --freeze_ctrl p21_ckpt/P42C90p1_LP_s111.pt --tagsuffix C90p2 > p42_c90p2_s111.log 2>&1
