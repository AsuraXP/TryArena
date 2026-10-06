#!/bin/bash
# C80b lane B: over-provisioned TF control on R5 (4L d96 ALiBi/NAPE, 308k p), 4000 st, seeds 111,222
cd /home/user/TryArena
for S in 111 222; do python3 arch_vet_p32.py --regimes R5 --arms TF-alibi-big,TF-nape-big --steps 4000 --seed $S >> p32_R5_tfbig.log 2>&1; done
