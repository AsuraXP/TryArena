#!/bin/bash
cd /home/user/TryArena
python3 arch_vet_p38.py --jobs R2 --arm SBK4 --seed 1010 >> p38_SBK4.log 2>&1
python3 arch_vet_p36.py --K SBK4 --C F --seeds 444,555,666,777,888,999,1010 >> p36ksbk4_cf.log 2>&1
