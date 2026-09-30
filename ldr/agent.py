"""V+M+C agent acting in the real environment, with robustness perturbations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .check_dream import load_mdnrnn
from .config import RUNS, C
from .controller import act_np
from .encode import load_vae
from .envs import make_env, set_road_friction
from .utils import preprocess, to_tensor


@dataclass
class Perturb:
    obs_noise: float = 0.0     # std of Gaussian pixel noise (0-255 scale)
    road_friction: float = 1.0 # grip multiplier on road tiles
    gas_cap: float = 1.0       # engine weakening
    latent_noise: float = 0.0  # std added to z (dashboard "nightmare")


class WorldModelAgent:
    def __init__(self, vae_path: Path = RUNS / "vae" / "best.pt",
                 rnn_path: Path = RUNS / "mdnrnn" / "best.pt", device="cpu"):
        torch.set_num_threads(1)            # one thread per worker process
        self.dev = torch.device(device)
        self.vae = load_vae(vae_path, self.dev)
        self.rnn = load_mdnrnn(rnn_path, self.dev)
        self.reset()

    def reset(self):
        self.state = self.rnn.initial_state(1, self.dev)

    @torch.no_grad()
    def encode(self, frame96: np.ndarray) -> np.ndarray:
        x = to_tensor(preprocess(frame96, C.img)[None], self.dev)
        return self.vae.encode(x)[0][0].cpu().numpy()

    @property
    def h(self) -> np.ndarray:
        return self.state[0][0, 0].cpu().numpy()

    @torch.no_grad()
    def observe(self, z: np.ndarray, a: np.ndarray) -> None:
        zt = torch.from_numpy(z[None]).float().to(self.dev)
        at = torch.from_numpy(a[None]).float().to(self.dev)
        *_, self.state = self.rnn.step(zt, at, self.state)

    def act(self, theta: np.ndarray, frame96: np.ndarray, rng=None, p: Perturb = Perturb()):
        z = self.encode(frame96)
        if p.latent_noise and rng is not None:
            z = z + p.latent_noise * rng.standard_normal(z.shape).astype(np.float32)
        a = act_np(theta, z, self.h)
        a[1] = min(a[1], p.gas_cap)
        self.observe(z, a)                  # h_{t+1} = f(h_t, z_t, a_t)
        return a

    def rollout(self, theta, seed: int, max_steps=C.max_steps, p: Perturb = Perturb(),
                patience: int = 200, env=None) -> tuple[float, int]:
        """Return (episode return, env steps). Stops early if no positive reward for
        `patience` steps and books the -0.1/step the car would still have paid."""
        own = env is None
        env = env or make_env()
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        if p.road_friction != 1.0:
            set_road_friction(env, p.road_friction)
        self.reset()
        total, since_pos, t = 0.0, 0, 0
        for t in range(1, max_steps + 1):
            if p.obs_noise:
                obs = np.clip(obs + rng.normal(0, p.obs_noise, obs.shape), 0, 255).astype(np.uint8)
            obs, r, term, trunc, _ = env.step(self.act(theta, obs, rng, p))
            total += r
            since_pos = 0 if r > 0 else since_pos + 1
            if term or trunc:
                break
            if patience and since_pos >= patience:
                total -= 0.1 * (max_steps - t)
                break
        if own:
            env.close()
        return total, t


class ControllerPolicy:
    """Drop-in data-collection policy for iterative world-model training (Phase 9)."""

    def __init__(self, rng, theta_path=RUNS / "controller" / "best.pt", noise=0.1):
        self.agent = WorldModelAgent()
        self.theta = torch.load(theta_path, weights_only=False)["theta"]
        self.rng, self.noise = rng, noise

    def reset(self):
        self.agent.reset()

    def __call__(self, env):
        a = self.agent.act(self.theta, env.unwrapped.state)
        a += self.noise * self.rng.standard_normal(3).astype(np.float32)
        return np.clip(a, [-1, 0, 0], [1, 1, 1]).astype(np.float32)
