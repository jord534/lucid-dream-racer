"""Does the dream RANK drivers the way the real track does? Spearman correlation over a
spread of drivers, with the dream's own noise floor from repeated evaluation.
rho near +1: searching in the dream is worthwhile. Near 0 or negative: it is not."""
import numpy as np, pickle
from scipy.stats import spearmanr
from ldr.check_dream import load_mdnrnn
from ldr.dream import DreamSim
from ldr.train_dream import dream_fitness
from ldr.train_controller import make_pool, evaluate_population
from ldr.config import SEED_VAL
from ldr.utils import get_device

RUNS = ["runs/controller/es.pkl", "runs/dream_v2/es.pkl"]   # point at the runs you have
TAU, ROLLOUTS, HORIZON, WARM, REPS = 1.15, 16, 200, 40, 5

def spread(es_path, rng):
    """CMA means per generation are not stored, so span the space by shrinking and
    perturbing the final mean: drivers from competent to hopeless."""
    es, *_ = pickle.loads(open(es_path, "rb").read())
    m = np.asarray(es.result.xfavorite, np.float32)
    return [m * s + rng.standard_normal(m.shape).astype(np.float32) * n
            for s, n in [(1, 0), (1, .05), (1, .15), (.5, 0), (.25, .05), (0, .1)]]

if __name__ == "__main__":
    dev, rng = get_device(), np.random.default_rng(0)
    sim = DreamSim(load_mdnrnn(device=dev), dev, tau=TAU)
    T = [t for p in RUNS for t in spread(p, rng)]
    reps = np.stack([dream_fitness(sim, T, ROLLOUTS, HORIZON, np.random.default_rng(s),
                                   dev, WARM) for s in range(REPS)])
    F, se = reps.mean(0), reps.std(0) / np.sqrt(REPS)
    with make_pool(6) as pool:
        R, _ = evaluate_population(pool, T, [SEED_VAL + i for i in range(8)])
    rho, p = spearmanr(F, R)
    for f, e, r in zip(F, se, R):
        print(f"dream {f:8.1f} +/- {e:4.1f}    real {r:8.1f}")
    print(f"Spearman rho = {rho:.3f}  (p = {p:.3f}, n = {len(T)})")
