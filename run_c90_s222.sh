#!/bin/bash
cd /home/user/TryArena
python3 arch_vet_p42.py --arms LP --seed 222 --fixed_random --live --tagsuffix C90p1 > p42_c90p1_s222.log 2>&1
python3 arch_vet_p42.py --arms LP --seed 222 --p0 0 --tau1 0.5 --freeze_ctrl p21_ckpt/P42C90p1_LP_s222.pt --tagsuffix C90p2 > p42_c90p2_s222.log 2>&1
