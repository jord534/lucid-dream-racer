"""Phase 4B (stretch): evolve the controller entirely inside the dream, and
measure how the policy transfers back to the real track.

    python -m ldr.train_dream --generations 500 --popsize 64 --rollouts 16
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cma
import numpy as np
import torch
from tqdm import tqdm

from .check_dream import load_mdnrnn
from .config import C, RUNS, SEED_VAL
from .controller import N_PARAMS, act_batched
from .dream import DreamSim
from .train_controller import evaluate_population, make_pool
from .utils import CSVLogger, get_device, save_ckpt, save_pickle


@torch.no_grad()
def dream_fitness(sim, X, rollouts, horizon, rng, dev):
    P = len(X)
    theta = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=dev)
    theta = theta.repeat_interleave(rollouts, 0)
    z, h = sim.reset(P * rollouts, rng)
    ret = torch.zeros(P * rollouts, device=dev)
    for _ in range(horizon):
        z, h, r, alive = sim.step(act_batched(theta, z, h))
        ret += r
        if alive.sum() == 0:
            break
    return ret.view(P, rollouts).mean(1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=500)
    p.add_argument("--popsize", type=int, default=64)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--horizon", type=int, default=C.max_steps)
    p.add_argument("--tau", type=float, default=C.tau)
    p.add_argument("--sigma0", type=float, default=0.1)
    p.add_argument("--real-every", type=int, default=25)
    p.add_argument("--real-tracks", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=RUNS / "dream_controller")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    dev = get_device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    sim = DreamSim(load_mdnrnn(device=dev), dev, tau=a.tau)
    es_path = a.out / "es.pkl"
    if es_path.exists():                                  # resume
        es, start, best = pickle.loads(es_path.read_bytes())
    else:
        es = cma.CMAEvolutionStrategy(np.zeros(N_PARAMS), a.sigma0,
                                      {"popsize": a.popsize, "seed": a.seed + 1, "verbose": -9})
        start, best = 0, -1e9
    rng, log = np.random.default_rng(a.seed + start), CSVLogger(a.out / "log.csv")
    real_seeds = [SEED_VAL + i for i in range(a.real_tracks)]
    bar = tqdm(total=a.generations, initial=start, desc="dream CMA-ES", unit="gen",
               dynamic_ncols=True)
    with make_pool(a.workers) as pool:
        for gen in range(start + 1, a.generations + 1):
            X = es.ask()
            fit = dream_fitness(sim, X, a.rollouts, a.horizon, rng, dev)
            es.tell(X, (-fit).tolist())
            row = dict(gen=gen, dream_fit_mean=fit.mean(), dream_fit_max=fit.max(), real_return="")
            if gen % a.real_every == 0:                    # the transfer gap, measured
                theta = np.asarray(es.result.xfavorite, np.float32)
                real, _ = evaluate_population(pool, [theta], real_seeds)
                row["real_return"] = real[0]
                if real[0] > best:
                    best = real[0]
                    save_ckpt(a.out / "best.pt", theta=theta, gen=gen, val_return=best)
            save_pickle(es_path, (es, gen, best))             # state first, then the log line
            log.log(**row)
            bar.update(1)
            bar.set_postfix(dream_avg=f"{fit.mean():.0f}",
                            best_real=f"{best:.0f}" if best > -1e8 else "none")
            if row["real_return"] != "":
                tqdm.write(f"gen {gen}: real track {row['real_return']:.1f}  "
                           f"(best so far {best:.1f})")
    bar.close()


if __name__ == "__main__":
    main()