#!/usr/bin/env bash
# Full pipeline, in order. Each line is resumable; re-run after any crash.
set -euo pipefail
python -m ldr.collect --episodes 1000 --pursuit-frac 0.5 --workers 8
python -m ldr.data --pack
python -m ldr.train_vae --steps 30000
python -m ldr.encode
python -m ldr.train_mdnrnn --steps 20000
python -m ldr.check_dream
python -m ldr.directions
python -m ldr.train_controller --generations 300 --popsize 32 --rollouts 4 --workers 8
python -m ldr.train_dream --generations 500
python -m ldr.evaluate --ckpt runs/controller/best.pt --name wm_real
python -m ldr.evaluate --ckpt runs/dream_controller/best.pt --name wm_dream
python -m ldr.analysis