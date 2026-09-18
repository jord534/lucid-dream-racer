"""Phase 1 contract: environment, preprocessing, privileged labels."""
import numpy as np

from ldr.envs import BrownianPolicy, PursuitPolicy, curvature_ahead, make_env
from ldr.utils import preprocess


def test_env_and_preprocess():
    env = make_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == (96, 96, 3) and obs.dtype == np.uint8
    for policy in (BrownianPolicy, PursuitPolicy):
        a = policy(np.random.default_rng(0))(env)
        assert a.shape == (3,) and -1 <= a[0] <= 1 and 0 <= a[1] <= 1 and 0 <= a[2] <= 1
        obs, *_ = env.step(a)
    assert preprocess(obs).shape == (64, 64, 3)
    assert -np.pi <= curvature_ahead(env) <= np.pi