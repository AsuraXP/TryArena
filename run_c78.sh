#!/bin/bash
# C78 lane queues. A: after SBK2 seeds -> P39 fluency probe. B: after SBK4 seeds -> unified fold with --K SBK4.
cd /home/user/TryArena
case "$1" in
  A) while pgrep -f "arm SBK2" >/dev/null; do sleep 60; done; python3 arch_vet_p39.py >> p39.log 2>&1 ;;
  B) while pgrep -f "arm SBK4" >/dev/null; do sleep 60; done; python3 arch_vet_p36.py --K SBK4 --seeds 111,222,333 >> p36ksbk4.log 2>&1 ;;
esac
