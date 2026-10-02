#!/bin/bash
# Waits for the on-road-head run (runs/loop_head) to end, then runs the control arm: the same loop from
# the same model, drivers, seeds and round-0 branches, but with no on-road head and no road penalty.
# Round 0's measurement (same model, drivers, validation seeds) and round 0's critic branches (the head
# is not used before the first fine-tune, so the head run's round-0 critic is exactly the control's) are
# reused, so the two arms fine-tune on identical data at round 0 and differ only in the head.
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy
while pgrep -f "ldr.loop --rounds 2 .*--out runs/loop_head" > /dev/null; do sleep 60; done
[ -f runs/loop_head/r00/critic.done ] || { echo "head run has no finished round-0 critic: not starting"; exit 1; }
mkdir -p runs/loop_ctrl/r00
python3 - <<'PY'
import json
e = json.load(open("runs/loop_road/r02/measure.json")); e["round"] = 0
json.dump({"hours": 0.0, "ledger": [e], "note": "round 0 measurement reused from runs/loop_road/r02"},
          open("runs/loop_ctrl/state.json", "w"), indent=1)
PY
for f in measure.json measure.done critic.json critic.done branches.pkl; do cp runs/loop_head/r00/$f runs/loop_ctrl/r00/$f; done
echo "=== control arm started $(date +%H:%M:%S) ==="
caffeinate -i python -m ldr.loop --rounds 2 --workers 6 --device cpu --price-eur-hour 0 \
  --select-by road --onset-frac 0.5 --start-from runs/loop_road/r02 \
  --prior-branches runs/loop/r00/branches.pkl runs/loop/r01/branches.pkl \
    runs/loop_road/r00/branches.pkl runs/loop_road/r01/branches.pkl \
  --round-offset 4 --out runs/loop_ctrl
