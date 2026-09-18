"""Environment construction, perturbations, privileged track geometry and
data-collection driving policies."""
from __future__ import annotations
import math
import gymnasium as gym
import numpy as np
from .config import C
def make_env(render_mode: str | None= None, **kwargs) -> gym.Env:
    return gym.make(C.env_id, render_mode=render_mode, continuous=True, **kwargs)

def set_road_friction(env: gym.Env, k: float) -> None:
    """Scale grip on every road tile (1.0 = default). Call after reset()."""
    for tile in env.unwrapped.road:
        tile.road_friction = k
def _nearest_tile(env: gym.Env) -> int:
    u = env.unwrapped
    track = np.asarray(u.track) # rows: (alpha, beta, x, y)
    pos = np.asarray(u.car.hull.position)
    return int(np.argmin(((track[:, 2:4] - pos) ** 2).sum(1)))

def curvature_ahead(env: gym.Env, lookahead: int = 10) -> float:
    """Signed heading change (rad) of the track `lookahead` tiles ahead.
    Privileged label used only to find a 'curvature' direction in latent space."""
    track = np.asarray(env.unwrapped.track)
    i = _nearest_tile(env)
    j = (i + lookahead) % len(track)
    d = track[j, 1] - track[i, 1]
    return float((d + math.pi) % (2 * math.pi) - math.pi)

def car_speed(env: gym.Env) -> float:
    v = env.unwrapped.car.hull.linearVelocity
    return float(math.hypot(v[0], v[1]))

class BrownianPolicy:
    """Temporally correlated random driving: OU-process steering and
    piecewise-constant gas/brake. Explores grass, spins and recoveries."""
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.reset()

    def reset(self):
        self.steer, self.hold, self.gas, self.brake = 0.0, 0, 0.5, 0.0

    def __call__(self, env: gym.Env) -> np.ndarray:
        self.steer += -0.15 * self.steer + 0.3 * self.rng.standard_normal()
        self.steer = float(np.clip(self.steer, -1, 1))
        if self.hold <= 0:
            self.hold = int(self.rng.integers(10, 40))
            self.gas = float(self.rng.uniform(0.0, 1.0))
            self.brake = float(self.rng.uniform(0, 0.6)) if self.rng.random() < 0.1 else 0.0
            self.hold -= 1
        return np.array([self.steer, self.gas, self.brake], dtype=np.float32)


class PursuitPolicy:
    """Noisy pure-pursuit driver using privileged track state. Produces on-road,
    at-speed data so the world model sees corners, not just grass.
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.reset()

    def reset(self):
        self.target_speed = float(self.rng.uniform(25, 70))
        self.lookahead = int(self.rng.integers(4, 9))
        self.noise = float(self.rng.uniform(0.0, 0.4))
        self.swerve = 0

    def __call__(self, env: gym.Env) -> np.ndarray:
        u = env.unwrapped
        track = np.asarray(u.track)
        i = _nearest_tile(env)
        tx, ty = track[(i + self.lookahead) % len(track), 2:4]

        px, py = u.car.hull.position
        ang = u.car.hull.angle
        fx, fy = -math.sin(ang), math.cos(ang)  # car forward vector

        dx, dy = tx - px, ty - py
        err = math.atan2(
            fx * dy - fy * dx,
            fx * dx + fy * dy,
        )

        steer = -2.0 * err + self.noise * self.rng.standard_normal()

        if self.swerve > 0:  # occasional deliberate mistakes
            steer, self.swerve = self.swerve_dir, self.swerve - 1
        elif self.rng.random() < 0.005:
            self.swerve, self.swerve_dir = (
                int(self.rng.integers(5, 25)),
                float(self.rng.choice([-1, 1])),
            )

        v = car_speed(env)
        gas = 0.6 if v < self.target_speed else 0.0
        brake = 0.3 if v > self.target_speed + 15 else 0.0

        return np.array(
            [np.clip(steer, -1, 1), gas, brake],
            dtype=np.float32,
        )


POLICIES = {
    "brownian": BrownianPolicy,
    "pursuit": PursuitPolicy,
}