"""The dream: a batched, differentiable-free simulator built from the MDN-RNN."""
from __future__ import annotations

import numpy as np
import torch

from .config import C, DATA
from .mdnrnn import MDNRNN, mdn_sample


class DreamSim:
    """Runs B dreams in parallel on one device.
    reset() starts each dream from the first frame of a random real episode
    (matching how real episodes start) or, with warm=K, from a random mid-episode
    frame after teacher-forcing K real steps to build up h."""

    def __init__(self, rnn: MDNRNN, device, tau: float = C.tau, latents=DATA / "latents.npz"):
        d = np.load(latents)
        self.rnn, self.dev, self.tau = rnn, device, tau
        self.mu = torch.from_numpy(d["mu"]).to(device)
        self.logvar = torch.from_numpy(d["logvar"]).to(device)
        self.actions = torch.from_numpy(d["actions"]).to(device)
        self.obs_start, self.act_start, self.ep_len = d["obs_start"], d["act_start"], d["ep_len"]
        self.train_eps = np.flatnonzero(~d["is_val"])

    def _z(self, idx):
        return self.mu[idx] + torch.randn_like(self.mu[idx]) * (0.5 * self.logvar[idx]).exp()

    @torch.no_grad()
    def reset(self, batch: int, rng: np.random.Generator, warm: int = 0):
        eps = rng.choice(self.train_eps, batch)
        self.state = self.rnn.initial_state(batch, self.dev)
        if warm == 0:
            self.z = self._z(torch.as_tensor(self.obs_start[eps], device=self.dev))
        else:
            t0 = np.array([rng.integers(0, max(1, self.ep_len[e] - warm)) for e in eps])
            for k in range(warm):
                o = torch.as_tensor(self.obs_start[eps] + t0 + k, device=self.dev)
                a = self.actions[torch.as_tensor(self.act_start[eps] + t0 + k, device=self.dev)]
                *_, self.state = self.rnn.step(self._z(o), a, self.state)
            self.z = self._z(torch.as_tensor(self.obs_start[eps] + t0 + warm, device=self.dev))
        self.alive = torch.ones(batch, device=self.dev)
        return self.z, self.h

    @property
    def h(self):
        return self.state[0][0]

    @torch.no_grad()
    def step(self, a: torch.Tensor, tau: float | None = None, z_offset=None):
        z_in = self.z if z_offset is None else self.z + z_offset
        logit, mu, logsig, r, d_logit, self.state = self.rnn.step(z_in, a, self.state)
        self.z = mdn_sample(logit, mu, logsig, self.tau if tau is None else tau)
        done = torch.bernoulli(torch.sigmoid(d_logit))
        reward = r * self.alive
        self.alive = self.alive * (1 - done)
        return self.z, self.h, reward, self.alive