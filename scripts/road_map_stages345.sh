#!/usr/bin/env bash
# Road-map experiment, stages 3-5 (reports/road_map/preregistration.md). Waits for stages 1-2
# (scripts/road_map_stages.sh) to finish, then runs each stage only if the one before passed.
# Exit code 3: stopped at a gate (a result, not an error).
#
#   nohup bash scripts/road_map_stages345.sh > runs/road_map/stages345.log 2>&1 &
#   python scripts/progress_watch.py runs/road_map/stages.log runs/road_map/stages345.log
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONWARNINGS=ignore
trap 'rc=$?; echo "EXIT $(date +%s) $rc"' EXIT
R=runs/road_map
S5=$R/stage5
MAPS="map_s0 map_s1"
ALL5="map_s0 map_s1 ctrl_s0 ctrl_s1"

mark() { echo "PROGRESS $(date +%s) $2 $3 $1"; }          # mark TASK DONE TOTAL
task() { local t=$1; shift; mark "$t" 0 1; "$@"; mark "$t" 1 1; }
passed() { python -c "import json, sys; sys.exit(0 if json.load(open('$1'))['passed'] else 1)"; }
stop() { echo "RESULT $1"; exit 3; }

plan="wait_for_stages_1_2=1800 patch_labels=150 tests=40 smoke_training=150 smoke_paths=120"
for m in $MAPS; do plan+=" train_$m=7200"; done
for m in $MAPS; do plan+=" measure_$m=1500"; done
plan+=" measure_ctrl_s0=400 measure_ctrl_s1=400 stage4_gate=10"
for m in $ALL5; do
  for p in 0 1 2; do plan+=" s5_${m}_proposer$p=600"; done
  plan+=" s5_${m}_measure=800"
  for p in 0 1 2; do plan+=" s5_${m}_test_p$p=200"; done
done
plan+=" stage5_summary=10"
echo "PLAN $plan"
echo "START $(date +%s)"

# ---- wait for stages 1-2
mark wait_for_stages_1_2 0 1
while pgrep -f "road_map_stages\.sh 1" > /dev/null; do sleep 30; done
mark wait_for_stages_1_2 1 1
[[ -f reports/road_map/stage2.json ]] || stop "stages 1-2 did not produce a stage-2 result: see runs/road_map/stages.log"
passed reports/road_map/stage2.json || stop "STAGE 3-5 NOT RUN: stage 2 did not pass (reports/road_map/stage2.json)"

# ---- stage 3: labels and plumbing
LDR_PROGRESS_TASK=patch_labels python -m ldr.label_patch --workers 6
task tests python -m pytest tests -q -x
rm -rf $R/smoke_map
(export LDR_PROGRESS_TASK=smoke_training
 python -m ldr.train_mdnrnn --steps 200 --shared-mixture --ss-prob 0.3 --map-weight 1 --map-input \
   --eval-every 100 --out $R/smoke_map)
(cd diagnostics && task smoke_paths python road_map_smoke.py ../$R/smoke_map/best.pt)
python - <<'EOF'
import json
from pathlib import Path
lab = json.loads(Path("reports/road_map/stage3_labels.json").read_text())
ok = lab["wheel_agreement_with_game"] >= 0.98
out = {"labels": lab, "tests_passed": True, "smoke_passed": True,
       "criteria": {"patch wheels agree with the game >= 0.98": ok}, "passed": ok}
Path("reports/road_map/stage3.json").write_text(json.dumps(out, indent=1))
print(f"RESULT patch wheels agree with the game on {lab['wheel_agreement_with_game']:.4f} of steps; "
      f"tests and plumbing smoke run passed")
print(f"RESULT STAGE 3 {'PASSED' if ok else 'FAILED'}  (reports/road_map/stage3.json)")
EOF
passed reports/road_map/stage3.json || stop "stopped after stage 3"

# ---- stage 4: train the two road-map models (concurrently), measure, gate
for s in 0 1; do
  (LDR_PROGRESS_TASK=train_map_s$s python -m ldr.train_mdnrnn --steps 20000 --shared-mixture --ss-prob 0.3 \
     --map-weight 1 --map-input --seed $s --out $R/map_s$s > $R/map_s$s.log 2>&1; \
   echo "trained map_s$s: exit $?") &
done
# the training logs are separate; copy their progress lines into this log while they run
while pgrep -f "train_mdnrnn.*--out $R/map_s" > /dev/null; do
  for s in 0 1; do (grep "^PROGRESS" $R/map_s$s.log 2>/dev/null || true) | tail -1; done
  sleep 60
done
wait
for s in 0 1; do
  (grep "^PROGRESS" $R/map_s$s.log || true) | tail -1
  [[ -f $R/map_s$s/best.pt ]] || { echo "training map_s$s failed, last lines:"; tail -5 $R/map_s$s.log; exit 1; }
done
cd diagnostics
for m in $MAPS; do
  LDR_PROGRESS_TASK=measure_$m python map_horizon.py --model ../$R/$m/best.pt --name $m --sets dev val test --tau 1.15 0.05
done
for s in 0 1; do
  LDR_PROGRESS_TASK=measure_ctrl_s$s python map_horizon.py --model ../runs/pose_state/ctrl_s$s/best.pt \
    --name ctrl_s$s --sets dev val --tau 1.15 0.05
done
task stage4_gate python map_horizon.py --gate --tau 1.15 0.05
cd ..
passed reports/road_map/stage4_gate.json || stop "stopped after stage 4: no controllers trained"

# ---- stage 5: controllers in each model (and in each control), validation measures, test tracks
for m in $ALL5; do
  if [[ $m == map_* ]]; then dir=$R/$m; else dir=runs/pose_state/$m; fi
  (cd diagnostics && python road_map_stage5.py setup $m ../$dir)
  LDR_PROGRESS_PREFIX=s5_$m python -m ldr.loop --rounds 0 --start-from $S5/$m --out $S5/${m}_loop \
    --price-eur-hour 0 --budget-eur 1 > $S5/${m}_loop.log 2>&1 &
  loop_pid=$!
  # the proposers write to their own logs; copy their progress lines into this log while they run
  while kill -0 $loop_pid 2> /dev/null; do
    for p in 0 1 2; do (grep -h "^PROGRESS" $S5/${m}_loop/r00/proposer$p.log 2>/dev/null || true) | tail -1; done
    [[ -f $S5/${m}_loop/r00/proposers.done && ! -f $S5/${m}_loop/r00/measure.done ]] && mark s5_${m}_measure 0 1
    sleep 60
  done
  wait $loop_pid || { echo "loop for $m failed, last lines:"; tail -15 $S5/${m}_loop.log; exit 1; }
  for p in 0 1 2; do (grep -h "^PROGRESS" $S5/${m}_loop/r00/proposer$p.log || true) | tail -1; done
  mark s5_${m}_measure 1 1
  for p in 0 1 2; do
    LDR_PROGRESS_TASK=s5_${m}_test_p$p python -m ldr.evaluate --ckpt $S5/${m}_loop/r00/proposer$p/best.pt \
      --name roadmap_${m}_p$p --vae runs/vae/best.pt --rnn $dir/best.pt --workers 6
  done
done
(cd diagnostics && task stage5_summary python road_map_stage5.py summarise)
