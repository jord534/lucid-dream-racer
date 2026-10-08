"""Pose state (ldr.label_pose, MDNRNN pose head/input): shapes, old checkpoints, the fed-back pose
in training and in the dream, and label alignment."""
import numpy as np
import torch

from ldr.check_dream import load_mdnrnn
from ldr.config import C
from ldr.dream import DreamSim
from ldr.label_pose import pose_by_obs, replay_pose
from ldr.mdnrnn import MDNRNN, POSE_DIM, pose_features, pose_lateral
from ldr.train_mdnrnn import SequenceSampler, loss_fn
from ldr.utils import save_ckpt


def fake_latents(path, lens=(70, 90), seed=0):
    """A latents.npz with two episodes (the second held out) plus pose sidecar files whose values
    encode their own index, so alignment can be read back."""
    rng = np.random.default_rng(seed)
    lens = np.array(lens)
    n_obs, n_act = int((lens + 1).sum()), int(lens.sum())
    obs_start, act_start = np.r_[0, np.cumsum(lens + 1)[:-1]], np.r_[0, np.cumsum(lens)[:-1]]
    np.savez(path, mu=rng.standard_normal((n_obs, C.z_dim)).astype(np.float32),
             logvar=np.full((n_obs, C.z_dim), -4, np.float32), ep_len=lens, obs_start=obs_start,
             act_start=act_start, is_val=np.array([False, True]),
             actions=rng.uniform(-1, 1, (n_act, 3)).astype(np.float32),
             rewards=rng.standard_normal(n_act).astype(np.float32), dones=np.zeros(n_act, bool))
    pose = np.zeros((n_act, POSE_DIM), np.float32)
    pose[:, 0] = np.arange(n_act) / n_act            # lateral = action index, scaled
    pose[:, 1], pose[:, 2], pose[:, 3] = 0.0, 1.0, np.arange(n_act) % 50
    pose0 = np.array([[-0.5, 0, 1, 0]] * len(lens), np.float32)
    np.save(str(path).replace(".npz", "_pose.npy"), pose)
    np.save(str(path).replace(".npz", "_pose0.npy"), pose0)
    return pose, pose0, obs_start, act_start, lens


def test_shapes_and_default_outputs():
    m = MDNRNN(shared=True, pose_dim=POSE_DIM, pose_input=True)
    z, a, q = torch.randn(2, 10, C.z_dim), torch.rand(2, 10, 3), torch.randn(2, 10, POSE_DIM)
    out = m(z, a, q=q, return_on=True, return_pose=True)
    assert len(out) == 8 and out[6].shape == (2, 10) and out[7].shape == (2, 10, POSE_DIM)
    assert len(m(z, a, q=q)) == 6                    # without return_pose: the usual outputs
    st = m.step(z[:, 0], a[:, 0], q=q[:, 0], return_pose=True)
    assert len(st) == 7 and st[-1].shape == (2, POSE_DIM)
    try:
        m(z, a)
        raise AssertionError("a pose-input model must refuse to run without q")
    except AssertionError as e:
        assert "pass q" in str(e)
    head_only = MDNRNN(shared=True, pose_dim=POSE_DIM)   # pose head as an auxiliary output only
    assert head_only.lstm.input_size == C.z_dim + 3
    assert head_only(z, a, return_pose=True)[-1].shape == (2, 10, POSE_DIM)


def test_default_model_unchanged():
    torch.manual_seed(0)
    m = MDNRNN(shared=True)
    assert m.pose_dim == 0 and not m.pose_input and not hasattr(m, "pose_head")
    assert {k.split(".")[0] for k in m.state_dict()} == {"lstm", "head", "on_head"}
    assert m.lstm.input_size == C.z_dim + 3
    torch.manual_seed(0)                              # same seed -> same init as before pose existed
    m2 = MDNRNN(shared=True, pose_dim=POSE_DIM, pose_input=False)
    assert torch.equal(m.head.weight, m2.head.weight) and torch.equal(m.on_head.weight, m2.on_head.weight)


def test_old_checkpoints_load(tmp_path):
    torch.manual_seed(1)
    old = MDNRNN(shared=True)
    sd = {k: v for k, v in old.state_dict().items() if not k.startswith("on_head")}  # before on_head
    save_ckpt(tmp_path / "old.pt", model=sd)
    save_ckpt(tmp_path / "on.pt", model=old.state_dict(), on_trained=True)
    for f, has_on in (("old.pt", False), ("on.pt", True)):
        m = load_mdnrnn(tmp_path / f)
        assert m.pose_dim == 0 and not m.pose_input and not m.has_pose and m.has_on == has_on
        z, a = torch.randn(3, 5, C.z_dim), torch.rand(3, 5, 3)
        assert torch.allclose(m(z, a)[1], old(z, a)[1])
    pm = MDNRNN(shared=True, pose_dim=POSE_DIM, pose_input=True)
    save_ckpt(tmp_path / "pose.pt", model=pm.state_dict(), pose_trained=True)
    m = load_mdnrnn(tmp_path / "pose.pt")
    assert m.pose_dim == POSE_DIM and m.pose_input and m.has_pose and m.shared
    save_ckpt(tmp_path / "head.pt", model=MDNRNN(shared=False, pose_dim=POSE_DIM).state_dict())
    m = load_mdnrnn(tmp_path / "head.pt")
    assert m.pose_dim == POSE_DIM and not m.pose_input and not m.has_pose and not m.shared


def test_pose_features():
    q = np.array([[0.5, 0.1, 0.99, 25.0], [-12.0, -1, 0, 100.0]], np.float32)
    f = pose_features(q)
    assert np.allclose(f[0], [0.25, 0.1, 0.99, 0.5]) and np.allclose(pose_lateral(f), [0.5, -5.0])
    assert torch.allclose(pose_features(torch.from_numpy(q)), torch.from_numpy(f))


def test_label_alignment(tmp_path):
    pose, pose0, obs_start, act_start, lens = fake_latents(tmp_path / "latents.npz")
    P = pose_by_obs(obs_start, act_start, lens, pose, pose0)
    for e in range(2):
        o, a = obs_start[e], act_start[e]
        assert np.array_equal(P[o], pose0[e])                         # observation 0: pose at reset
        assert np.array_equal(P[o + 1: o + lens[e] + 1], pose[a: a + lens[e]])  # obs t+1: after action t
    s = SequenceSampler(tmp_path / "latents.npz", L=16, pose=True)
    rng = np.random.default_rng(0)
    _, act, _, _, _, q_in, q_next = s.sample(8, "train", rng, "cpu", with_pose=True)
    assert q_in.shape == q_next.shape == (8, 16, POSE_DIM)
    assert torch.equal(q_in[:, 1:], q_next[:, :-1])                   # target of step t = input of t+1
    # the target of step t is the label recorded after action t: find it from the action itself
    acts = np.load(tmp_path / "latents.npz")["actions"]
    for b in range(8):
        i = int(np.flatnonzero((acts == act[b, 3].numpy()).all(1))[0])
        assert np.isclose(pose_lateral(q_next[b, 3]).item(), pose[i, 0])


def test_replay_labels_follow_the_car():
    acts = np.tile(np.array([[0.0, 0.8, 0.0]], np.float32), (40, 1))
    pose0, pose, rew, on = replay_pose(7, acts)
    assert pose.shape == (40, POSE_DIM) and on.all()
    assert abs(pose0[0]) < 0.05 and pose0[2] > 0.9 and pose0[3] == 0   # on the centre line, at rest
    assert pose[0, 3] > 0 and pose[-1, 3] > pose[0, 3]                 # label t is after action t
    assert np.allclose(pose[:, 1] ** 2 + pose[:, 2] ** 2, 1, atol=1e-5)
    _, poseb, rewb, _ = replay_pose(7, acts)                      # replay is deterministic
    assert np.array_equal(poseb, pose) and np.array_equal(rewb, rew)


def _manual_free_pose(m, z, a, q_in):
    """Teacher-forced latents, own predicted pose fed back from step 1 on."""
    state, q, preds = None, q_in[:, 0], []
    for t in range(z.shape[1]):
        *_, state, qh = m.step(z[:, t], a[:, t], state, q=q, return_pose=True)
        preds.append(qh)
        q = qh
    return torch.stack(preds, 1)


def test_pose_feedback_in_training(tmp_path):
    fake_latents(tmp_path / "latents.npz")
    s = SequenceSampler(tmp_path / "latents.npz", L=12, pose=True)
    batch = s.sample(4, "train", np.random.default_rng(0), "cpu", with_pose=True)
    torch.manual_seed(0)
    m = MDNRNN(shared=True, pose_dim=POSE_DIM, pose_input=True)
    z, a, q_in, q_next = batch[0], batch[1], batch[-2], batch[-1]
    # pose_ss_prob 1, ss_prob 0: every pose input after the first is the model's own prediction
    st = {}
    loss_fn(m, batch, ss_prob=0.0, pose_ss_prob=1.0, pose_weight=1.0, stats=st)
    with torch.no_grad():
        own = _manual_free_pose(m, z, a, q_in)
        tf = m(z, a, q=q_in, return_pose=True)[-1]
    assert np.isclose(st["pose_mse"], torch.nn.functional.mse_loss(own, q_next).item(), rtol=1e-5)
    st0 = {}
    loss_fn(m, batch, ss_prob=0.0, pose_ss_prob=0.0, pose_weight=1.0, stats=st0)   # teacher-forced
    assert np.isclose(st0["pose_mse"], torch.nn.functional.mse_loss(tf, q_next).item(), rtol=1e-5)
    assert not np.isclose(st0["pose_mse"], st["pose_mse"])
    total, *_ = loss_fn(m, batch, ss_prob=0.3, pose_weight=1.0)
    total.backward()
    assert m.pose_head.weight.grad is not None and torch.isfinite(total)


def test_pose_feedback_in_dream(tmp_path):
    fake_latents(tmp_path / "latents.npz")
    torch.manual_seed(0)
    m = MDNRNN(shared=True, pose_dim=POSE_DIM, pose_input=True).eval()
    sim = DreamSim(m, "cpu", latents=tmp_path / "latents.npz", start_from="all")
    rng = np.random.default_rng(3)
    sim.reset(2, rng, warm=5)
    # the warm-up and the first input pose are the true ones
    rng2 = np.random.default_rng(3)
    eps = rng2.choice(sim.train_eps, 2)
    t0 = np.array([rng2.integers(0, max(1, sim.ep_len[e] - 5)) for e in eps])
    o = torch.as_tensor(sim.obs_start[eps] + t0 + 5)
    assert torch.equal(sim.q, sim.pose[o])
    state = tuple(x.clone() for x in sim.state)
    z, q = sim.z.clone(), sim.q.clone()
    a = torch.zeros(2, 3)
    sim.step(a)
    with torch.no_grad():
        *_, st_ref, q_ref = m.step(z, a, state, q=q, return_pose=True)
    assert torch.allclose(sim.q, q_ref) and torch.allclose(sim.state[0], st_ref[0])
    q1 = sim.q.clone()
    sim.step(a)                                       # the next step reads the fed-back prediction
    assert not torch.equal(sim.q, q1)
