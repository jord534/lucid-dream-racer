#!/bin/bash
# For each of the three drivers of the road-status run (round 2, trained in runs/loop_road/r02/mdnrnn):
# the 100-test-track evaluation, then the two dream-failure tests. Results: reports/eval_loop_road_r02_p*.json
# and reports/dream_failure_analysis/loop_road_r02_p*/
set -e
cd "$(dirname "$0")/../.."
source .venv/bin/activate
export SDL_VIDEODRIVER=dummy PYTHONPATH=.
LOGS=reports/loop_run_2026-10-02_road/post_tests
mkdir -p $LOGS
M=runs/loop_road/r02
for i in 0 1 2; do
  tag=loop_road_r02_p$i
  echo "=== driver $i: test-track evaluation, $(date +%H:%M:%S) ==="
  python -m ldr.evaluate --ckpt $M/proposer$i/best.pt --name $tag --vae $M/vae/best.pt \
    --rnn $M/mdnrnn/best.pt --workers 6 | tee $LOGS/p${i}_eval.log
  export LDR_DIAG_TAG=$tag LDR_DIAG_CTRL=$M/proposer$i/best.pt LDR_DIAG_MODEL=$M
  mkdir -p reports/dream_failure_analysis/$tag
  echo "=== driver $i: collecting real rollouts, $(date +%H:%M:%S) ==="
  python diagnostics/dream_fidelity.py --collect > $LOGS/p${i}_collect.log 2>&1
  tail -2 $LOGS/p${i}_collect.log
  echo "=== driver $i: closed-loop and actions-in-sim tests, $(date +%H:%M:%S) ==="
  python diagnostics/dream_closed_loop.py > $LOGS/p${i}_closed_loop.log 2>&1 &
  python diagnostics/dream_actions_in_sim.py > $LOGS/p${i}_actions_in_sim.log 2>&1 &
  wait
  echo "=== driver $i finished, $(date +%H:%M:%S) ==="
done
echo "=== all done $(date +%H:%M:%S) ==="
