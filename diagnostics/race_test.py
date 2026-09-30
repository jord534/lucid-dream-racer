import numpy as np
from ldr.controller import N_PARAMS
from ldr.config import SEED_TRAIN
from ldr.train_controller import evaluate_population, evaluate_racing, make_pool
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = [rng.standard_normal(N_PARAMS).astype(np.float32) * 0.15 for _ in range(8)]
    seeds = [SEED_TRAIN + i for i in range(4)]
    with make_pool(2) as pool:
        f_all, n_all = evaluate_population(pool, X, seeds, 80)
        f_race, n_race, saved = evaluate_racing(pool, X, seeds, 0.5, 80)
    top_all, top_race = set(np.argsort(-f_all)[:4]), set(np.argsort(-f_race)[:4])
    print(f"steps full {n_all} | racing {n_race} | saved {100*(1-n_race/n_all):.0f}% "
          f"({saved} rollouts)")
    print("top-half agreement:", len(top_all & top_race), "of 4")
