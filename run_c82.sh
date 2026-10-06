#!/bin/bash
# C82: inference-envelope sweep after both fold lanes finish (needs idle cores for honest timing)
cd /home/user/TryArena
while pgrep -f "arch_vet_p36.py" >/dev/null; do sleep 60; done
python3 arch_vet_p41.py --lens 256,1024,4096,16384 >> p41.log 2>&1
