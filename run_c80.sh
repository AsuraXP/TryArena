#!/bin/bash
# C80 lane B: matched-budget Transformer control (ALiBi, 42.7k p = 2x the K) on R5, 4000 st, 3 seeds
cd /home/user/TryArena
for S in 111 222 333; do python3 arch_vet_p32.py --regimes R5 --arms TF-alibi --steps 4000 --seed $S >> p32_R5_tf.log 2>&1; done
