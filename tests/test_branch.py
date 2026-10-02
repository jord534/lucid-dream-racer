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


def test_onset_windows_contain_the_onset(tmp_path):
    rng = np.random.default_rng(0)
    eps = [_episode(120, rng) for _ in range(6)]
    for e in eps:
        e["onset"] = 70
    path = tmp_path / "extra.npz"
    assemble(eps, path)
    s = SequenceSampler(path)
    pairs = s.near_onset["train"]
    assert len(pairs) > 0
    first_act = {int(a) for a in s.starts["train"][:, 1]}
    # window start t (relative to its episode) must satisfy onset - (L-16) <= t <= onset - 16
    d = np.load(path)
    rel = [int(a - d["act_start"][e]) for a in pairs[:, 1] for e in range(len(d["ep_len"]))
           if d["act_start"][e] <= a < d["act_start"][e] + d["ep_len"][e]]
    assert all(70 - (C.seq_len - 16) <= t <= 70 - 16 for t in rel) and first_act
    z, *_ = s.sample(8, "train", rng, "cpu", onset_frac=0.5)
    assert z.shape[0] == 8


def test_episodes_without_onset_are_never_oversampled(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "extra.npz"
    assemble([_episode(80, rng) for _ in range(10)], path)  # no "onset" key: unknown
    s = SequenceSampler(path)
    assert len(s.near_onset["train"]) == 0
    assert s.sample(4, "train", rng, "cpu", onset_frac=0.5)[0].shape[0] == 4


def test_road_probe_learns_a_separable_road_signal():
    from ldr.branch import road_probe

    rng = np.random.default_rng(0)
    logs = []
    for _ in range(10):
        on = rng.random(300) > 0.4
        mu = rng.standard_normal((300, C.z_dim)).astype(np.float32)
        mu[:, 0] += 3 * (np.r_[True, on[:-1]] * 2 - 1)  # latent carries the label of obs_t
        logs.append({"mu": mu, "logvar": np.full((300, C.z_dim), -4.0, np.float32), "on": on})
    _probe, info = road_probe(logs, rng)
    assert info["ok"] and info["auc"] > 0.95
    assert road_probe([{**logs[0], "on": np.ones(300, bool)}], rng)[0] is None


def test_on_road_labels_travel_with_the_episodes_and_train_the_head(tmp_path):
    import torch

    from ldr.mdnrnn import MDNRNN
    from ldr.train_mdnrnn import loss_fn

    rng = np.random.default_rng(0)
    eps = [_episode(100, rng) for _ in range(6)]
    for e in eps:
        e["on"] = rng.random(100) > 0.5
    eps.append(_episode(100, rng))  # no flags: unknown (-1), never trained on
    path = tmp_path / "extra.npz"
    assemble(eps, path)
    s = SequenceSampler(path)
    assert set(np.unique(s.on)) == {-1, 0, 1} and (s.on[-100:] == -1).all()
    batch = s.sample(8, "train", rng, "cpu", with_on=True)
    assert len(batch) == 6 and batch[5].shape == (8, C.seq_len)
    model = MDNRNN()
    stats = {}
    loss_fn(model, batch, on_weight=1.0, stats=stats)
    loss_fn(model, batch, on_weight=1.0, ss_prob=0.3, stats=stats)  # scheduled-sampling path too
    assert 0 <= stats["on_acc"] <= 1 and np.isfinite(stats["on_bce"])
    with torch.no_grad():  # the 5-tuple callers still work and the head is optional
        assert len(model(batch[0], batch[1])) == 6 and len(model(batch[0], batch[1], None, True)) == 7


def test_old_checkpoints_load_without_a_trained_head(tmp_path):
    import torch

    from ldr.check_dream import load_mdnrnn
    from ldr.mdnrnn import MDNRNN

    m = MDNRNN()
    sd = {k: v for k, v in m.state_dict().items() if not k.startswith("on_head")}
    torch.save({"model": sd}, tmp_path / "old.pt")
    assert load_mdnrnn(tmp_path / "old.pt").has_on is False
    torch.save({"model": m.state_dict(), "on_trained": False}, tmp_path / "untrained.pt")
    assert load_mdnrnn(tmp_path / "untrained.pt").has_on is False
    torch.save({"model": m.state_dict(), "on_trained": True}, tmp_path / "trained.pt")
    assert load_mdnrnn(tmp_path / "trained.pt").has_on is True
