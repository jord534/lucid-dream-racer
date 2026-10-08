#!/usr/bin/env bash
# Road-map experiment, stages 1 and 2 (reports/road_map/preregistration.md). Resumable: parts already
# written are skipped. Stage 2 runs only if stage 1 passed.
#
#   nohup bash scripts/road_map_stages.sh 1 2 > runs/road_map/stages.log 2>&1 &
#   python scripts/progress_watch.py runs/road_map/stages.log
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
export PYTHONWARNINGS=ignore
trap 'rc=$?; echo "EXIT $(date +%s) $rc"' EXIT
STAGES=" ${*:-1 2} "
DRIVERS="loop_head_r02_p0 loop_head_r02_p1 loop_head_r02_p2 loop_road_r02_p0 loop_road_r02_p1 loop_road_r02_p2"

# estimated seconds per task (measured on small samples), refined by the scripts as they run
plan=""
if [[ $STAGES == *" 1 "* ]]; then
  for d in $DRIVERS; do plan+=" devruns_$d=150"; done
  plan+=" label_data=240"
  for d in $DRIVERS; do plan+=" label_test_$d=20"; done
  for d in $DRIVERS; do plan+=" label_dev_$d=12"; done
fi
[[ $STAGES == *" 2 "* ]] && plan+=" train_motion_guesser=510 evaluate=100"
echo "PLAN$plan"
echo "START $(date +%s)"

if [[ $STAGES == *" 1 "* ]]; then
  # stage 1a: the six dream-trained drivers on validation tracks 900,300-900,349
  for d in $DRIVERS; do
    arm=${d%%_r02_*}; p=${d##*_p}
    (cd diagnostics && LDR_DIAG_MODEL="$ROOT/runs/$arm/r02" LDR_DIAG_CTRL="$ROOT/runs/$arm/r02/proposer$p/best.pt" \
       python road_map_devruns.py --tag "$d")
  done
  # stage 1b: exact geometry for every run, and the stage-1 check
  python -m ldr.label_geometry --workers 6
fi

if [[ $STAGES == *" 2 "* ]]; then
  python -c "import json, sys; sys.exit(0 if json.load(open('reports/road_map/stage1.json'))['passed'] else 3)" \
    || { echo "RESULT stage 2 not run: stage 1 did not pass"; exit 3; }
  (cd diagnostics && python road_map_feasibility.py)
fi
