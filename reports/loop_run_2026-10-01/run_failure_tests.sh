#!/bin/bash
# The two dream-failure tests on each round-2 controller of the loop run, each in the round-2 world
# model it was trained in. Per controller: collect its real rollouts on the 100 test tracks, then run
# dream_closed_loop and dream_actions_in_sim side by side. Results: reports/dream_failure_analysis/<tag>/
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy PYTHONPATH=.
LOGS=reports/loop_run_2026-10-01/failure_tests
mkdir -p $LOGS
for i in 0 1 2; do
  export LDR_DIAG_TAG=loop_r02_p$i LDR_DIAG_CTRL=runs/loop/r02/proposer$i/best.pt LDR_DIAG_MODEL=runs/loop/r02
  mkdir -p reports/dream_failure_analysis/$LDR_DIAG_TAG
  echo "=== controller $i: collecting real rollouts, $(date +%H:%M:%S) ==="
  python diagnostics/dream_fidelity.py --collect > $LOGS/p${i}_collect.log 2>&1
  tail -2 $LOGS/p${i}_collect.log
  echo "=== controller $i: closed-loop test and actions-in-sim test, $(date +%H:%M:%S) ==="
  python diagnostics/dream_closed_loop.py > $LOGS/p${i}_closed_loop.log 2>&1 &
  python diagnostics/dream_actions_in_sim.py > $LOGS/p${i}_actions_in_sim.log 2>&1 &
  wait
  echo "=== controller $i finished, $(date +%H:%M:%S) ==="
done
echo "=== all done $(date +%H:%M:%S) ==="
