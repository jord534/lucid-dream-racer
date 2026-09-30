"""Phase 1.5: pack shards into one memory-mapped frame array + metadata,
and provide random-access samplers for training.

    python -m ldr.data --pack
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import DATA, C


def pack(rollouts: Path, out: Path, val_frac: float = 0.05) -> None:
    files = sorted(rollouts.glob("ep_[0-9]*.npz"))
    assert files, f"no shards in {rollouts}"
    T = np.array([len(np.load(f)["actions"]) for f in tqdm(files, desc="scan")])
    obs_start = np.concatenate([[0], np.cumsum(T + 1)[:-1]])
    act_start = np.concatenate([[0], np.cumsum(T)[:-1]])
    out.mkdir(parents=True, exist_ok=True)
    frames = np.lib.format.open_memmap(out / "frames.npy", mode="w+", dtype=np.uint8,
                                       shape=(int((T + 1).sum()), C.img, C.img, 3))
    keys = ["actions", "rewards", "dones", "curv"]
    meta = {k: [] for k in keys}
    for e, f in enumerate(tqdm(files, desc="pack")):
        d = np.load(f)
        frames[obs_start[e]: obs_start[e] + T[e] + 1] = d["obs"]
        for k in keys:
            meta[k].append(d[k])
    frames.flush()
    n_val = max(1, int(len(files) * val_frac))
    rng = np.random.default_rng(0)
    is_val = np.zeros(len(files), bool)
    is_val[rng.choice(len(files), n_val, replace=False)] = True
    np.savez(out / "meta.npz", ep_len=T, obs_start=obs_start, act_start=act_start,
             is_val=is_val, **{k: np.concatenate(v) for k, v in meta.items()})
    print(f"packed {len(files)} episodes, {frames.shape[0]:,} frames -> {out}")


class FrameSampler:
    """Uniform random minibatches of frames from the memmap, split by episode."""

    def __init__(self, root: Path = DATA / "packed"):
        self.frames = np.load(root / "frames.npy", mmap_mode="r")
        m = np.load(root / "meta.npz")
        idx = {True: [], False: []}
        for e, (s, t) in enumerate(zip(m["obs_start"], m["ep_len"])):
            idx[bool(m["is_val"][e])].append(np.arange(s, s + t + 1))
        self.idx = {"train": np.concatenate(idx[False]), "val": np.concatenate(idx[True])}

    def sample(self, batch: int, split: str, rng: np.random.Generator) -> np.ndarray:
        pool = self.idx[split]
        pick = np.sort(rng.choice(pool, batch, replace=batch > len(pool)))
        return self.frames[pick]   # sorted indices = fewer disk seeks


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pack", action="store_true")
    p.add_argument("--rollouts", type=Path, default=DATA / "rollouts")
    p.add_argument("--out", type=Path, default=DATA / "packed")
    a = p.parse_args()
    if a.pack:
        pack(a.rollouts, a.out)
