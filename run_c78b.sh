#!/bin/bash
# C78b lane B continuation: after the unified SBK4 fold -> R5 (28 keys, S16) arms SBK8 then SBK4, seed 111 then 222,333
cd /home/user/TryArena
while pgrep -f "K SBK4" >/dev/null; do sleep 60; done
for ARM in SBK8 SBK4; do for S in 111 222 333; do python3 arch_vet_p38.py --jobs R5 --arm $ARM --seed $S >> p38_R5_$ARM.log 2>&1; done; done
