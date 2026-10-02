import numpy as np

from ldr.branch import assemble
from ldr.config import C
from ldr.train_mdnrnn import SequenceSampler


def _episode(T, rng):
    return {
        "mu": rng.standard_normal((T + 1, C.z_dim)).astype(np.float32),
        "logvar": np.zeros((T + 1, C.z_dim), np.float32),
        "act": rng.random((T, 3)).astype(np.float32),
        "rew": np.zeros(T, np.float32),
        "done": np.zeros(T, bool),
        "seed": 0,
        "gap": 0.0,
    }


def test_assembled_branches_are_readable_by_the_sampler(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "extra.npz"
    assemble([_episode(80, rng) for _ in range(10)], path)
    s = SequenceSampler(path)
    z, a, z_next, _r, _d = s.sample(4, "train", rng, "cpu")
    assert z.shape == (4, C.seq_len, C.z_dim) and a.shape == (4, C.seq_len, 3)
    assert z_next.shape == z.shape and len(s.starts["val"]) > 0
