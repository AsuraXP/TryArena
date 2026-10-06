#!/bin/bash
cd /home/user/TryArena
python3 arch_vet_p42.py --arms LP --seed 111 --p0 0 --live --tau1 0.5 --l1 0.002 --tagsuffix C84 >> p42_c84.log 2>&1
python3 arch_vet_p42.py --arms LP --seed 333 --p0 0 --live --tau1 0.5 --l1 0.002 --tagsuffix C84 >> p42_c84.log 2>&1
