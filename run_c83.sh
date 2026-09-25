#!/bin/bash
# C83 lane B: R6 k-hop — HOP (path-hindsight), TF-alibi, TF-alibi-big, HOP-BIGRAM ablation; seed 111 then HOP 222/333
cd /home/user/TryArena
for ARM in HOP TF-alibi TF-big HOP-BIGRAM; do python3 arch_vet_p42.py --arm $ARM --seed 111 >> p42.log 2>&1; done
for S in 222 333; do python3 arch_vet_p42.py --arm HOP --seed $S >> p42.log 2>&1; done
