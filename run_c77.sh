#!/bin/bash
# C77 lane: blocked cuckoo arms, seed 111 then 222/333 for the winner
cd /home/user/TryArena
ARM=$1; shift
for S in "$@"; do python3 arch_vet_p38.py --arm $ARM --seed $S >> p38_$ARM.log 2>&1; done
