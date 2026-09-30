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
from tqdm import tqdm

from .agent import WorldModelAgent
from .config import RUNS, SEED_TRAIN, SEED_VAL, C
from .controller import N_PARAMS
from .utils import CSVLogger, save_ckpt, save_pickle

_AGENT: WorldModelAgent | None = None


def _init_worker():
    global _AGENT
    os.environ["OMP_NUM_THREADS"] = "1"
    _AGENT = WorldModelAgent()


def _eval(job):
    """One rollout per task. Small units keep fast cores fed: on a machine with
    performance and efficiency cores, a slow worker delays the generation by at most
    one rollout instead of a whole candidate's worth."""
    i, theta, seed, max_steps = job
    r, n = _AGENT.rollout(theta, seed, max_steps)
    return i, r, n


def make_pool(workers: int) -> Pool:
    return Pool(workers, initializer=_init_worker)


def evaluate_racing(pool, thetas, seeds, keep_frac=0.5, max_steps=C.max_steps):
    """Score every candidate on seeds[0], keep the best `keep_frac`, and give only those
    the remaining tracks. CMA-ES uses the ranking and weights only the better half, so
    precision spent on the losers is wasted. Eliminated candidates keep their single-track
    score, which is noisier but only has to place them in the bottom half.
    Returns (fitness, env steps, rollouts saved)."""
    first, steps = evaluate_population(pool, thetas, seeds[:1], max_steps)
    if len(seeds) == 1:
        return first, steps, 0
    keep = np.argsort(-first)[: max(2, int(round(len(thetas) * keep_frac)))]
    rest, n = evaluate_population(pool, [thetas[i] for i in keep], seeds[1:], max_steps)
    fit = first.copy()
    fit[keep] = (first[keep] + rest * (len(seeds) - 1)) / len(seeds)
    # A one-track score can beat a survivor's four-track mean by luck, which would hand
    # an eliminated candidate weight in the CMA update. Rank them below every survivor,
    # keeping their order among themselves: that is what round one decided.
    cut = np.setdiff1d(np.arange(len(thetas)), keep)
    cut = cut[np.argsort(-first[cut])]
    fit[cut] = fit[keep].min() - 1e-3 * (1 + np.arange(len(cut)))
    saved = len(cut) * (len(seeds) - 1)
    return fit, steps + n, saved


def evaluate_population(pool, thetas, seeds, max_steps=C.max_steps):
    jobs = [(i, np.asarray(t, np.float32), s, max_steps)
            for i, t in enumerate(thetas) for s in seeds]
    totals = np.zeros(len(thetas))
    steps = 0
    for i, r, n in pool.imap_unordered(_eval, jobs, chunksize=1):
        totals[i] += r
        steps += n
    return totals / len(seeds), steps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--popsize", type=int, default=32)
    p.add_argument("--rollouts", type=int, default=4)
    p.add_argument("--sigma0", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--race", action="store_true",
                   help="score all candidates on one track, then only the better half on the rest")
    p.add_argument("--keep-frac", type=float, default=0.5)
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
            if a.race:
                fit, n, saved = evaluate_racing(pool, X, seeds, a.keep_frac, a.max_steps)
            else:
                (fit, n), saved = evaluate_population(pool, X, seeds, a.max_steps), 0
            es.tell(X, (-fit).tolist())                   # CMA-ES minimises
            gen, env_steps = gen + 1, env_steps + n
            row = dict(gen=gen, env_steps=env_steps, fit_mean=fit.mean(), fit_max=fit.max(),
                       sigma=es.sigma, rollouts_saved=saved, val_return="")
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
