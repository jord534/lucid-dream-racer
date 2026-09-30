"""Phase 7: model-free PPO baseline (Stable-Baselines3, rl-baselines3-zoo
CarRacing-v3 hyperparameters) for the sample-efficiency comparison.

    python -m ldr.baseline_ppo --timesteps 4000000 --seed 0
    python -m ldr.baseline_ppo --eval runs/ppo_s0/final.zip --name ppo_s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize
from torch import nn

from .config import REPORTS, RUNS, SEED_TEST, SEED_TRAIN, C

SKIP = 2


class FrameSkip(gym.Wrapper):
    """Repeat each action SKIP times and sum rewards (env steps = agent steps * SKIP)."""

    def __init__(self, env, skip=SKIP):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total = 0.0
        for _ in range(self.skip):
            obs, r, term, trunc, info = self.env.step(action)
            total += r
            if term or trunc:
                break
        return obs, total, term, trunc, info


def wrap(env):
    env = FrameSkip(env)
    env = gym.wrappers.ResizeObservation(env, (64, 64))
    return gym.wrappers.GrayscaleObservation(env, keep_dim=True)


def make(n_envs, seed, norm_reward=True):
    venv = make_vec_env(C.env_id, n_envs=n_envs, seed=seed, wrapper_class=wrap,
                        vec_env_cls=SubprocVecEnv if n_envs > 1 else None)
    venv = VecFrameStack(venv, 2)
    return VecNormalize(venv, norm_obs=False, norm_reward=norm_reward)


def linear(lr0):
    return lambda progress_remaining: progress_remaining * lr0


def train(a):
    out = RUNS / f"ppo_s{a.seed}"
    env = make(8, SEED_TRAIN + a.seed)
    eval_env = make(1, SEED_TRAIN + 50_000 + a.seed, norm_reward=False)
    model = PPO("CnnPolicy", env, seed=a.seed, verbose=1, batch_size=128, n_steps=512,
                gamma=0.99, gae_lambda=0.95, n_epochs=10, ent_coef=0.0, use_sde=True,
                sde_sample_freq=4, max_grad_norm=0.5, vf_coef=0.5, clip_range=0.2,
                learning_rate=linear(1e-4),
                policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.GELU,
                                   net_arch=dict(pi=[256], vf=[256])))
    cb = EvalCallback(eval_env, n_eval_episodes=8, eval_freq=25_000 // 8,
                      log_path=str(out), best_model_save_path=str(out), deterministic=True)
    model.learn(a.timesteps, callback=cb)
    model.save(out / "final.zip")
    env.save(str(out / "vecnormalize.pkl"))


def evaluate(a):
    model = PPO.load(a.eval)
    returns = []
    for i in range(a.tracks):
        env = make(1, SEED_TEST + i, norm_reward=False)
        obs, done, total = env.reset(), False, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, dones, infos = env.step(action)
            total += float(env.get_original_reward()[0]); done = bool(dones[0])
        returns.append(total)
        env.close()
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"eval_{a.name}.json").write_text(json.dumps(
        dict(name=a.name, returns=returns, mean=float(np.mean(returns)))))
    print(f"{a.name}: {np.mean(returns):.1f} +/- {np.std(returns):.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=4_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval", type=Path)
    p.add_argument("--name", default="ppo")
    p.add_argument("--tracks", type=int, default=100)
    a = p.parse_args()
    evaluate(a) if a.eval else train(a)
