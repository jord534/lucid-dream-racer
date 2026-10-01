"""Does the dream RANK drivers the way the real track does? Spearman correlation over a
spread of drivers, with the dream's own noise floor from repeated evaluation.
rho near +1: searching in the dream is worthwhile. Near 0 or negative: it is not.

Prints rho over all controllers AND over the subset excluding controllers whose real
return is below 0 (inactive/degenerate controllers that agree trivially near zero). That
exclusion rule is fixed in advance, on the real (ground-truth) return only, before the
dream scores are looked at, so it cannot be applied post hoc to flatter either condition.

    python diagnostics/rank_check.py --label after
Writes reports/rank_<label>.json."""
import argparse
import json

import numpy as np, pickle
from scipy.stats import spearmanr

from ldr.check_dream import load_mdnrnn
from ldr.config import REPORTS, SEED_VAL
from ldr.dream import DreamSim
from ldr.train_controller import make_pool, evaluate_population
from ldr.train_dream import dream_fitness
from ldr.utils import get_device

# Fixed across labels: the before/after comparison must score the same controllers under
# different predictors, not different controllers.
RUNS = ["runs/controller/es.pkl", "runs/dream_v2/es.pkl"]
TAU, ROLLOUTS, HORIZON, WARM, REPS = 1.15, 16, 200, 40, 5


def spread(es_path, rng):
    """CMA means per generation are not stored, so span the space by shrinking and
    perturbing the final mean: drivers from competent to hopeless."""
    es, *_ = pickle.loads(open(es_path, "rb").read())
    m = np.asarray(es.result.xfavorite, np.float32)
    return [m * s + rng.standard_normal(m.shape).astype(np.float32) * n
            for s, n in [(1, 0), (1, .05), (1, .15), (.5, 0), (.25, .05), (0, .1)]]


def rank_figures(F: np.ndarray, R: np.ndarray) -> dict:
    rho_all, p_all = spearmanr(F, R)
    keep = R >= 0
    rho_sub, p_sub = spearmanr(F[keep], R[keep])
    return dict(n_all=int(len(R)), rho_all=float(rho_all), p_all=float(p_all),
                n_subset=int(keep.sum()), rho_subset=float(rho_sub), p_subset=float(p_sub))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True, help="tag for the output file, e.g. before/after")
    a = p.parse_args()

    dev, rng = get_device(), np.random.default_rng(0)
    sim = DreamSim(load_mdnrnn(device=dev), dev, tau=TAU)
    T = [t for path in RUNS for t in spread(path, rng)]
    reps = np.stack([dream_fitness(sim, T, ROLLOUTS, HORIZON, np.random.default_rng(s),
                                   dev, WARM) for s in range(REPS)])
    F, se = reps.mean(0), reps.std(0) / np.sqrt(REPS)
    with make_pool(6) as pool:
        R, _ = evaluate_population(pool, T, [SEED_VAL + i for i in range(8)])
    R = np.asarray(R)

    for f, e, r in zip(F, se, R):
        print(f"dream {f:8.1f} +/- {e:4.1f}    real {r:8.1f}")
    result = rank_figures(F, R)
    print(f"all controllers   (n={result['n_all']:2d}): "
          f"rho = {result['rho_all']:+.3f}  (p = {result['p_all']:.3f})")
    print(f"real return >= 0  (n={result['n_subset']:2d}): "
          f"rho = {result['rho_subset']:+.3f}  (p = {result['p_subset']:.3f})")

    out = REPORTS / f"rank_{a.label}.json"
    out.write_text(json.dumps(dict(label=a.label, **result), indent=1))
    print(f"wrote {out}")
