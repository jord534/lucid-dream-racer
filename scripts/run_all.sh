#!/usr/bin/env bash
# Full pipeline, in order. Each line is resumable; re-run after any crash.
#
# Two rounds: the first round (world model + controllers trained on it) is what the
# note calls "before"; the on-policy data and predictor changes in technical_note.md
# section 3.4 (added data, --shared-mixture, --ss-prob) are then applied and everything
# downstream of the predictor is retrained. See Appendix C of the technical note for
# which run folder each reported figure and table comes from.
set -euo pipefail

# --- round 1: pursuit + Brownian data, first predictor, first controllers ---
python -m ldr.collect --episodes 1000 --pursuit-frac 0.5 --workers 8
python -m ldr.data --pack
python -m ldr.train_vae --steps 30000
python -m ldr.encode
python -m ldr.train_mdnrnn --steps 20000            # -> runs/mdnrnn
python -m ldr.check_dream
python -m ldr.directions
python -m ldr.train_controller --generations 300 --popsize 32 --rollouts 4 --workers 8
python -m ldr.train_dream --generations 500 --horizon 1000   # round 1 used the full,
                                                               # uncapped horizon (3.2)

# Preserve the first predictor under the name the diagnostics and the technical note
# refer to it by, before round 2 overwrites runs/mdnrnn.
mv runs/mdnrnn runs/mdnrnn_v1

# --- round 2: add 400 on-policy episodes, repaired predictor, repaired controllers ---
python -m ldr.collect --episodes 400 --pursuit-frac 0.0 --controller-frac 1.0 \
    --first-ep 1000 --workers 8
python -m ldr.data --pack
python -m ldr.encode
python -m ldr.train_mdnrnn --steps 20000 --shared-mixture --ss-prob 0.3   # -> runs/mdnrnn
python -m ldr.check_dream
python -m ldr.train_controller --generations 300 --popsize 32 --rollouts 4 --workers 8 \
    --race --out runs/controller_v2
python -m ldr.train_dream --generations 500 --out runs/dream_v3

# --- test-set and robustness evaluation (100 tracks each) ---
python -m ldr.evaluate --ckpt runs/controller_v2/best.pt --name wm_real_v2
python -m ldr.evaluate --ckpt runs/dream_v3/best.pt --name wm_dream_v3
python -m ldr.evaluate --ckpt runs/controller_v2/best.pt --name wm_real_v2_gas50 \
    --gas-cap 0.5
python -m ldr.evaluate --ckpt runs/controller_v2/best.pt --name wm_real_v2_noise10 \
    --obs-noise 10
python -m ldr.evaluate --ckpt runs/controller_v2/best.pt --name wm_real_v2_noise25 \
    --obs-noise 25
python -m ldr.evaluate --ckpt runs/controller_v2/best.pt --name wm_real_v2_noise50 \
    --obs-noise 50
python -m ldr.evaluate --ckpt runs/controller_v2/best.pt --name wm_real_v2_grip0.8 \
    --road-friction 0.8
python -m ldr.evaluate --ckpt runs/controller_v2/best.pt --name wm_real_v2_grip0.6 \
    --road-friction 0.6
python -m ldr.analysis

# --- rank-correlation diagnostic (Table 3.3, section 3.3, Appendix A.1) ---
python diagnostics/rank_check.py --label after
# The "before" figure needs the pre-fix predictor back in place temporarily:
#   mv runs/mdnrnn runs/mdnrnn_tmp && mv runs/mdnrnn_v1 runs/mdnrnn
#   python diagnostics/rank_check.py --label before
#   python -m ldr.evaluate --ckpt runs/dream_controller/best.pt --name wm_dream_v1 --workers 6
#   mv runs/mdnrnn runs/mdnrnn_v1 && mv runs/mdnrnn_tmp runs/mdnrnn
