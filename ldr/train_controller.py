"""Phase 4A: evolve the linear controller with CMA-ES in the REAL environment,
using frozen V and M as feature extractors (the setup Ha & Schmidhuber used
for CarRacing).

    python -m ldr.train_controller --popsize 32 --rollouts 4 --workers 8 --generations 300
"""
from __future__ import annotations

import argparse
import os
import pickle
from multiprocessing import Pool
from pathlib import Path

import cma
import numpy as np
import torch
from tqdm import tqdm

from .agent import WorldModelAgent
from .config import C, RUNS, SEED_TRAIN, SEED_VAL
from .controller import N_PARAMS
from .utils import CSVLogger, save_ckpt, save_pickle

_AGENT: WorldModelAgent | None = None


def _init_worker():
    global _AGENT
    os.environ["OMP_NUM_THREADS"] = "1"
    _AGENT = WorldModelAgent()


def _eval(job):
    theta, seeds, max_steps = job
    out = [_AGENT.rollout(theta, s, max_steps) for s in seeds]
    return float(np.mean([r for r, _ in out])), int(sum(n for _, n in out))


def make_pool(workers: int) -> Pool:
    return Pool(workers, initializer=_init_worker)


def evaluate_population(pool, thetas, seeds, max_steps=C.max_steps):
    res = pool.map(_eval, [(np.asarray(t, np.float32), seeds, max_steps) for t in thetas])
    return np.array([r for r, _ in res]), sum(n for _, n in res)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--sigma0", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--val-tracks", type=int, default=16)
    p.add_argument("--max-steps", type=int, default=C.max_steps)
    p.add_argument("--out", type=Path, default=RUNS / "controller")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    es_path = a.out / "es.pkl"
    if es_path.exists():                                  # resume
        es, gen, env_steps, best = pickle.loads(es_path.read_bytes())
    else:
        es = cma.CMAEvolutionStrategy(np.zeros(N_PARAMS), a.sigma0,
                                      {"popsize": a.popsize, "seed": a.seed + 1, "verbose": -9})
        gen, env_steps, best = 0, 0, -1e9
    log = CSVLogger(a.out / "log.csv")
    rng = np.random.default_rng(a.seed + gen)
    val_seeds = [SEED_VAL + i for i in range(a.val_tracks)]
    bar = tqdm(total=a.generations, initial=gen, desc="CMA-ES", unit="gen", dynamic_ncols=True)
    with make_pool(a.workers) as pool:
        while gen < a.generations:
            # Common random numbers: every candidate drives the same tracks this generation.
            seeds = [int(s) for s in SEED_TRAIN + rng.integers(0, 800_000, a.rollouts)]
            X = es.ask()
            fit, n = evaluate_population(pool, X, seeds, a.max_steps)
            es.tell(X, (-fit).tolist())                   # CMA-ES minimises
            gen, env_steps = gen + 1, env_steps + n
            row = dict(gen=gen, env_steps=env_steps, fit_mean=fit.mean(), fit_max=fit.max(),
                       sigma=es.sigma, val_return="")
            if gen % a.val_every == 0:
                mean_theta = es.result.xfavorite
                val, n = evaluate_population(pool, [mean_theta], val_seeds, a.max_steps)
                env_steps += n
                row["val_return"] = val[0]
                if val[0] > best:
                    best = val[0]
                    save_ckpt(a.out / "best.pt", theta=np.asarray(mean_theta, np.float32),
                              gen=gen, val_return=best)
            save_pickle(es_path, (es, gen, env_steps, best))   # state first, then the log line
            log.log(**row)
            bar.update(1)
            bar.set_postfix(avg=f"{fit.mean():.0f}", top=f"{fit.max():.0f}",
                            best_val=f"{best:.0f}" if best > -1e8 else "none",
                            sigma=f"{es.sigma:.3f}")
            if row["val_return"] != "":
                tqdm.write(f"gen {gen}: validation {row['val_return']:.1f}  "
                           f"(best so far {best:.1f})")
    bar.close()


if __name__ == "__main__":
    main()