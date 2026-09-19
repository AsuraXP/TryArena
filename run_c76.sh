#!/bin/bash
# C76: TF control fairness — third seed at the original recipe + lr-tuned (1e-3, warmup 300, cosine) x 3 seeds, NAPE and ALiBi arms.
cd /home/user/TryArena
python3 arch_vet_p23.py --arms nape,alibi --seeds 333 >> p23_s333.log 2>&1
python3 arch_vet_p23.py --arms nape,alibi --seeds 111,222,333 --lr 1e-3 --warmup 300 >> p23_tuned.log 2>&1
