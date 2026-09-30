"""Phase 5 contract: NumPy and batched controllers agree and emit valid actions."""
import numpy as np
import torch

from ldr.config import C
from ldr.controller import N_PARAMS, act_batched, act_np


def test_controller_agreement():
    rng = np.random.default_rng(0)
    theta = (rng.standard_normal(N_PARAMS) * 0.1).astype(np.float32)
    z = rng.standard_normal(C.z_dim).astype(np.float32)
    h = rng.standard_normal(C.h_dim).astype(np.float32)
    a1 = act_np(theta, z, h)
    a2 = act_batched(torch.from_numpy(theta)[None], torch.from_numpy(z)[None],
                     torch.from_numpy(h)[None])[0].numpy()
    assert N_PARAMS == 867 and np.allclose(a1, a2, atol=1e-5)
    assert -1 <= a1[0] <= 1 and 0 <= a1[1] <= 1 and 0 <= a1[2] <= 1
