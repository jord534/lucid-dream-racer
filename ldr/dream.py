"""The dream: a batched, differentiable-free simulator built from the MDN-RNN."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .config import DATA, C
from .mdnrnn import MDNRNN, mdn_sample, pose_features


class DreamSim:
    """Runs B dreams in parallel on one device.
    reset() starts each dream from the first frame of a random real episode
    (matching how real episodes start) or, with warm=K, from a random mid-episode
    frame after teacher-forcing K real steps to build up h.
    A world model that reads the pose (rnn.pose_input) is given the true pose (ldr.label_pose) at
    the start and through the warm-up, then its own predicted pose, carried in self.q.
    A world model that reads a road patch (rnn.map_input; ldr.road_map) is given the true patch
    through the warm-up; then the car is placed at its true pose on the whole track of the episode it
    starts from (self.map) and moved by the model's own predicted motion, and the patch is read there."""

    def __init__(self, rnn: MDNRNN, device, tau: float = C.tau, latents=DATA / "latents.npz",
                 start_from: str = "good", road_penalty: float = 0.0):
        """start_from="good" draws warm-start states only from the better half of the
        recorded episodes (the pure-pursuit ones), so the controller is scored from
        situations a competent driver actually reaches, rather than from a spin on the
        grass left behind by the Brownian collector."""
        d = np.load(latents)
        self.rnn, self.dev, self.tau = rnn, device, tau
        # road_penalty: reward lost per step in proportion to the on-road head's P(off the road)
        assert not road_penalty or rnn.has_on, "road_penalty needs a world model with a trained on-road head"
        self.road_penalty = road_penalty
        self.mu = torch.from_numpy(d["mu"]).to(device)
        self.logvar = torch.from_numpy(d["logvar"]).to(device)
        self.actions = torch.from_numpy(d["actions"]).to(device)
        self.obs_start, self.act_start, self.ep_len = d["obs_start"], d["act_start"], d["ep_len"]
        self.q = None
        if rnn.pose_input:
            from .label_pose import load_pose_obs
            raw = load_pose_obs(latents, d)
            assert raw is not None, "this world model reads the pose: run python -m ldr.label_pose"
            self.pose = torch.from_numpy(pose_features(raw)).to(device)   # indexed like mu
        self.map = None
        if rnn.map_input:
            g = np.load(Path(latents).with_name("road_map") / "geom_data.npz")
            assert np.array_equal(g["obs_start"], self.obs_start), "geometry not aligned with the latents"
            self.geo_pose = g["pose"].astype(np.float64)
            self.geo_tiles = [g["tiles"][s: s + n].astype(np.float64) for s, n in zip(g["tile_start"], g["n_tiles"])]
            side = Path(latents).with_name(Path(latents).stem + "_patch.npy")
            self.patch = torch.from_numpy(np.load(side).astype(np.float32)).to(device)   # indexed like mu
        train = np.flatnonzero(~d["is_val"])
        if start_from == "good":
            ret = np.array([d["rewards"][s: s + T].sum()
                            for s, T in zip(d["act_start"], d["ep_len"])])
            train = train[ret[train] >= np.median(ret[train])]
        self.train_eps = train

    def _z(self, idx):
        return self.mu[idx] + torch.randn_like(self.mu[idx]) * (0.5 * self.logvar[idx]).exp()

    @torch.no_grad()
    def reset(self, batch: int, rng: np.random.Generator, warm: int = 0):
        eps = rng.choice(self.train_eps, batch)
        self.state = self.rnn.initial_state(batch, self.dev)
        pin, pm = self.rnn.pose_input, self.rnn.map_input
        if warm == 0:
            o = torch.as_tensor(self.obs_start[eps], device=self.dev)
            self.z = self._z(o)
        else:
            t0 = np.array([rng.integers(0, max(1, self.ep_len[e] - warm)) for e in eps])
            for k in range(warm):
                o = torch.as_tensor(self.obs_start[eps] + t0 + k, device=self.dev)
                a = self.actions[torch.as_tensor(self.act_start[eps] + t0 + k, device=self.dev)]
                q = self.pose[o] if pin else None
                *_, self.state = self.rnn.step(self._z(o), a, self.state, q=q, patch=self.patch[o] if pm else None)
            o = torch.as_tensor(self.obs_start[eps] + t0 + warm, device=self.dev)
            self.z = self._z(o)
        self.q = self.pose[o] if pin else None
        if pm:
            from .road_map import RoadMap
            self.map = RoadMap.stack([self.geo_tiles[e] for e in eps], self.geo_pose[o.cpu().numpy()])
        self.alive = torch.ones(batch, device=self.dev)
        return self.z, self.h

    @property
    def h(self):
        return self.state[0][0]

    @torch.no_grad()
    def step(self, a: torch.Tensor, tau: float | None = None, z_offset=None):
        z_in = self.z if z_offset is None else self.z + z_offset
        pin, pm = self.rnn.pose_input, self.rnn.map_input
        patch = torch.from_numpy(self.map.read()).to(self.dev) if pm else None
        logit, mu, logsig, r, d_logit, self.state, *extra = self.rnn.step(
            z_in, a, self.state, return_on=bool(self.road_penalty), q=self.q, return_pose=pin,
            patch=patch, return_motion=pm
        )
        on = extra[:1] if self.road_penalty else []
        if pin:
            self.q = extra[int(bool(self.road_penalty))]   # the model's own predicted pose is the next input
        if pm:
            from .road_map import MOTION_SCALE
            self.map.move(extra[-1].cpu().numpy().astype(np.float64) * MOTION_SCALE)   # moved by its own motion
        self.z = mdn_sample(logit, mu, logsig, self.tau if tau is None else tau)
        done = torch.bernoulli(torch.sigmoid(d_logit))
        if on:
            r = r - self.road_penalty * (1 - torch.sigmoid(on[0]))
        reward = r * self.alive
        self.alive = self.alive * (1 - done)
        return self.z, self.h, reward, self.alive
