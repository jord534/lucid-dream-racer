"""Phase 6: final evaluation on 100 unseen tracks, plus robustness audits.

    python -m ldr.evaluate --ckpt runs/controller/best.pt --name wm_real
    python -m ldr.evaluate --ckpt runs/controller/best.pt --name wm_real_noise20 --obs-noise 20
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from .agent import Perturb, WorldModelAgent
from .config import REPORTS, RUNS, SEED_TEST

_AGENT = None


def _init(vae_path, rnn_path):
    global _AGENT
    _AGENT = WorldModelAgent(vae_path, rnn_path)


def _run(job):
    theta, seed, pert = job
    r, n = _AGENT.rollout(theta, seed, p=Perturb(**pert), patience=0)  # no early stop at test
    return seed, r, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--vae", type=Path, default=RUNS / "vae" / "best.pt")
    p.add_argument("--rnn", type=Path, default=RUNS / "mdnrnn" / "best.pt",
                   help="world model the controller reads its memory from (must be the one it was trained with)")
    p.add_argument("--tracks", type=int, default=100)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--obs-noise", type=float, default=0.0)
    p.add_argument("--road-friction", type=float, default=1.0)
    p.add_argument("--gas-cap", type=float, default=1.0)
    a = p.parse_args()
    theta = torch.load(a.ckpt, weights_only=False)["theta"]
    pert = dict(obs_noise=a.obs_noise, road_friction=a.road_friction, gas_cap=a.gas_cap)
    jobs = [(theta, SEED_TEST + i, pert) for i in range(a.tracks)]
    with Pool(a.workers, initializer=_init, initargs=(a.vae, a.rnn)) as pool:
        res = pool.map(_run, jobs)
    returns = np.array([r for _, r, _ in res])
    out = dict(name=a.name, ckpt=str(a.ckpt), vae=str(a.vae), rnn=str(a.rnn), perturb=pert, seeds=[s for s, _, _ in res],
               returns=returns.tolist(), mean=float(returns.mean()), std=float(returns.std()))
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"eval_{a.name}.json").write_text(json.dumps(out, indent=1))
    print(f"{a.name}: {returns.mean():.1f} +/- {returns.std():.1f} over {len(returns)} tracks")


if __name__ == "__main__":
    main()