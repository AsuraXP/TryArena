#!/bin/bash
cd /home/user/TryArena
for s in "$@"; do python3 arch_vet_p37.py --jobs R2 --arm CF85R --seed $s >> p37_10seed.log 2>&1; done
python3 arch_vet_p36.py --K P37 --seeds $(echo "$@" | tr ' ' ',') >> p36k37_10seed.log 2>&1
