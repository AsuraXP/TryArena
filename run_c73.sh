#!/bin/bash
# lane script: train K experts for given seeds, then P36 on them
cd /home/user/TryArena
for s in "$@"; do python3 arch_vet_p35.py --jobs R2 --arm SEEN --seed $s >> p35_seen10.log 2>&1; done
python3 arch_vet_p36.py --seeds $(echo "$@" | tr ' ' ',') >> p36_10seed.log 2>&1
