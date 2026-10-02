#!/bin/bash
# Evaluate the three round-2 dream controllers of the loop run on the 100 test tracks.
# Each controller reads its memory from the round-2 world model it was trained in.
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy
for i in 0 1 2; do
  echo "=== controller $i, started $(date +%H:%M:%S) ==="
  python -m ldr.evaluate --ckpt runs/loop/r02/proposer$i/best.pt --name loop_r02_p$i \
    --vae runs/loop/r02/vae/best.pt --rnn runs/loop/r02/mdnrnn/best.pt --workers 6
done
echo "=== done $(date +%H:%M:%S) ==="
