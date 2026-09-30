"""Phase 1: collect driving rollouts into compressed per-episode shards.

    python -m ldr.collect --episodes 1000 --pursuit-frac 0.5 --workers 8
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import DATA, SEED_DATA, C
from .envs import POLICIES, curvature_ahead, make_env
from .utils import preprocess


def run_episode(job: tuple) -> tuple[int, int]:
    ep, policy_name, max_steps, out_dir = job
    out = Path(out_dir) / f"ep_{ep:05d}.npz"
    if out.exists():                       # resumable: skip finished episodes
        return ep, 0
    seed = SEED_DATA + ep
    rng = np.random.default_rng(seed)
    env = make_env()
    if policy_name == "controller":                    # Phase 9 iterative collection
        from .agent import ControllerPolicy
        policy = ControllerPolicy(rng)
    else:
        policy = POLICIES[policy_name](rng)
    obs, _ = env.reset(seed=seed)
    frames, curv = [preprocess(obs, C.img)], [curvature_ahead(env)]
    acts, rews, dones = [], [], []
    for _ in range(max_steps):
        a = policy(env)
        obs, r, terminated, truncated, _ = env.step(a)
        frames.append(preprocess(obs, C.img))
        curv.append(curvature_ahead(env))
        acts.append(a); rews.append(r); dones.append(terminated)
        if terminated or truncated:
            break
    env.close()
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(
        tmp,
        obs=np.stack(frames),                          # (T+1, 64, 64, 3) uint8
        actions=np.stack(acts).astype(np.float32),     # (T, 3)
        rewards=np.asarray(rews, np.float32),          # (T,)
        dones=np.asarray(dones, bool),                 # (T,) true terminations only
        curv=np.asarray(curv, np.float32),             # (T+1,) privileged label
        policy=policy_name,
    )
    tmp.rename(out)
    return ep, len(acts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--pursuit-frac", type=float, default=0.5)
    p.add_argument("--controller-frac", type=float, default=0.0)
    p.add_argument("--first-ep", type=int, default=0, help="append new episodes")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=C.max_steps)
    p.add_argument("--out", type=Path, default=DATA / "rollouts")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    n_ctrl = int(args.episodes * args.controller_frac)
    n_pursuit = int(args.episodes * args.pursuit_frac)
    def kind(i):
        return "controller" if i < n_ctrl else "pursuit" if i < n_ctrl + n_pursuit else "brownian"
    jobs = [(args.first_ep + i, kind(i), args.max_steps, str(args.out))
            for i in range(args.episodes)]
    total = 0
    with Pool(args.workers) as pool:
        for _, n in tqdm(pool.imap_unordered(run_episode, jobs), total=len(jobs)):
            total += n
    print(f"collected {total:,} new transitions into {args.out}")


if __name__ == "__main__":
    main()
