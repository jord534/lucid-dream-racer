"""Wall-clock for one generation-equivalent at different worker counts."""
import time
import numpy as np
from ldr.config import SEED_TRAIN
from ldr.controller import N_PARAMS
from ldr.train_controller import evaluate_population, make_pool

POP, ROLLOUTS, STEPS = 8, 2, 300      # raise to 32 / 4 / 1000 for the real thing
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = [rng.standard_normal(N_PARAMS).astype(np.float32) * 0.1 for _ in range(POP)]
    seeds = [SEED_TRAIN + i for i in range(ROLLOUTS)]
    for w in (2, 3, 4, 6, 8):
        with make_pool(w) as pool:
            evaluate_population(pool, X[:2], seeds, 50)        # warm the workers
            t0 = time.perf_counter()
            _, n = evaluate_population(pool, X, seeds, STEPS)
            dt = time.perf_counter() - t0
        print(f"workers {w}: {dt:6.1f} s   {n / dt:6.0f} steps/s")
