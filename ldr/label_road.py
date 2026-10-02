"""Exact on-road labels for the world model's original training data.

    python -m ldr.label_road --workers 3        (resumable; about 40 minutes with 6 workers)

The recorded actions and the track seed reproduce every collected episode exactly (checked against
the recorded rewards), so replaying the actions in the real simulator gives the true on-road flag
(some wheel touches a road tile) after every action. Writes one small file per episode to
data/onroad/ and, at the end, data/latents_onroad.npy: an int8 array aligned with the `actions`
array of data/latents.npz (1 on the road, 0 off it). train_mdnrnn --on-weight reads it.
"""

from __future__ import annotations

import argparse
import time
from multiprocessing import Pool

import numpy as np

from .config import DATA, SEED_DATA
from .envs import make_env

ROLLOUTS, OUT = DATA / "rollouts", DATA / "onroad"


def label_episode(ep: int):
    f = OUT / f"ep_{ep:05d}.npy"
    if f.exists():
        return ep, True
    sh = np.load(ROLLOUTS / f"ep_{ep:05d}.npz")
    env = make_env()
    env.reset(seed=SEED_DATA + ep)
    on, total = [], 0.0
    for a in sh["actions"]:
        total += env.step(a)[1]
        on.append(any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels))
    env.close()
    ok = abs(total - float(sh["rewards"].sum())) < 1e-2  # replay reproduced the recording
    tmp = f.with_suffix(".tmp.npy")
    np.save(tmp, np.array(on, np.int8))
    tmp.rename(f)
    return ep, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA / "latents.npz")
    n = len(d["ep_len"])
    # Episode e of latents.npz must be shard ep_{e}: same actions, checked on the first steps.
    for e in range(0, n, 97):
        sh = np.load(ROLLOUTS / f"ep_{e:05d}.npz")
        s = int(d["act_start"][e])
        assert d["ep_len"][e] == len(sh["actions"]) and np.array_equal(
            d["actions"][s : s + 20], sh["actions"][:20]
        ), f"episode {e} of latents.npz is not shard ep_{e:05d}"
    t0, bad = time.time(), []
    with Pool(a.workers) as pool:
        for i, (ep, ok) in enumerate(pool.imap_unordered(label_episode, range(n)), 1):
            if not ok:
                bad.append(ep)
            if i % 50 == 0 or i == n:
                el = time.time() - t0
                print(
                    f"labelled {i}/{n}  ({el / 60:.1f} min elapsed, about "
                    f"{el / i * (n - i) / 60:.1f} min left)",
                    flush=True,
                )
    on = np.concatenate([np.load(OUT / f"ep_{e:05d}.npy") for e in range(n)])
    assert len(on) == len(d["actions"]), (len(on), len(d["actions"]))
    np.save(DATA / "latents_onroad.npy", on)
    print(
        f"wrote {DATA / 'latents_onroad.npy'}: {len(on):,} steps, {1 - on.mean():.1%} off the road; "
        f"replays not matching the recorded rewards: {len(bad)}"
    )


if __name__ == "__main__":
    main()
