#!/bin/bash
# C79b lane A: 10-seed certification of the headline row (K=SBK4, C=F). Trains K then C for each seed, then folds all 7.
cd /home/user/TryArena
for S in 444 555 666 777 888 999 1010; do
  python3 arch_vet_p38.py --jobs R2 --arm SBK4 --seed $S >> p38_SBK4.log 2>&1
done
python3 arch_vet_p40.py --seeds 444,555,666,777,888,999,1010 >> p40.log 2>&1
python3 arch_vet_p36.py --K SBK4 --C F --seeds 444,555,666,777,888,999,1010 >> p36ksbk4_cf.log 2>&1
