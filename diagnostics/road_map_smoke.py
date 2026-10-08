"""Stage-3 plumbing check for a road-map world model: every path that later stages use, run for a few
steps (reports/road_map/preregistration.md, stage 3).

    python diagnostics/road_map_smoke.py runs/road_map/smoke_map/best.pt
Checks: the checkpoint loads with the map; the dream (DreamSim) on CPU and on the training device; the
dream fitness of train_dream; the agent in the real game; the loop's real log and dream branch; one
chunk of the stage-4 measurement. Exits non-zero on any failure."""

import sys
from pathlib import Path

import numpy as np
import torch

import map_horizon as mh
from ldr import branch
from ldr.agent import WorldModelAgent
from ldr.check_dream import load_mdnrnn
from ldr.config import RUNS, SEED_VAL, C
from ldr.controller import N_PARAMS
from ldr.dream import DreamSim
from ldr.train_dream import dream_fitness
from ldr.utils import get_device


def main(path):
    rnn = load_mdnrnn(path)
    assert rnn.map_input and rnn.has_map, "checkpoint does not load as a trained road-map model"
    rng = np.random.default_rng(0)
    for dev in ("cpu", get_device(None)):
        sim = DreamSim(load_mdnrnn(path, dev), dev)
        z, h = sim.reset(8, rng, warm=40)
        for _ in range(5):
            z, h, r, alive = sim.step(torch.rand(8, 3, device=dev))
        assert torch.isfinite(z).all() and np.isfinite(sim.map.pose).all()
        fit = dream_fitness(sim, [np.zeros(N_PARAMS, np.float32)] * 2, 2, 5, rng, dev, warm=40)
        assert np.isfinite(fit).all()
        print(f"dream on {dev}: ok (pose moved {np.abs(sim.map.pose[:, :2]).sum():.1f})", flush=True)
    theta = (np.random.default_rng(1).standard_normal(N_PARAMS) * 0.05).astype(np.float32)
    ag = WorldModelAgent(RUNS / "vae" / "best.pt", path)
    ret, n = ag.rollout(theta, SEED_VAL + 999, max_steps=30, patience=0)
    print(f"agent in the real game: ok ({n} steps, return {ret:.1f})", flush=True)
    branch.init_worker(RUNS / "vae" / "best.pt", path)
    log = branch.rollout_log((theta, SEED_VAL + 999, 80))
    assert len(log["pose"]) == len(log["act"]) + 1 and len(log["tiles"]) > 50
    acts, rs, zs, _ = branch.dream_branch(rnn, theta, log, 50, 10, 2, C.tau)
    assert acts.shape == (10, 2, 3) and np.isfinite(rs).all()
    print("loop real log and dream branch: ok", flush=True)
    probe = torch.nn.Sequential(torch.nn.Linear(C.z_dim, 1))
    wins = mh.dev_windows()[f"{mh.DEV} | random start"][:3]
    res, truth, mask, tpose = mh.run_chunk(rnn, probe, wins, C.tau, torch.Generator().manual_seed(0))
    assert set(mh.MODES) <= set(res) | {"real"} and np.isfinite(res["own"][1]).all()
    cnt = mh.pose_counts(res["own"][1], tpose, mask)
    print(f"stage-4 measurement chunk: ok (position error at steps 20-40 {cnt[3]['pos_err'].sum() / max(mask[20:40].sum() * mh.B, 1):.2f})",
          flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
