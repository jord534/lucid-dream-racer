"""Exact pose labels (the car relative to the road) for the world model's training data.

    python -m ldr.label_pose --workers 6          (a few minutes)
    python -m ldr.label_pose --rollouts data/dream_failure/rollouts_loop_head_r02_p0.npz ...

Modelled on ldr.label_road: every recorded episode is replayed from its track seed with its recorded
actions (checked against the recorded rewards and, where stored, the on-road flags). After every
action it records the pose q_t:
  0  lateral offset from the track centre line, signed (+ left of the driving direction), divided
     by the road half-width (TRACK_WIDTH): |offset| > 1 is past the edge of the tarmac
  1  sin of the heading error (car heading minus the track tangent at the nearest centre point)
  2  cos of the heading error
  3  speed (norm of the hull's linear velocity, simulator units)
The nearest centre point is tracked incrementally from the start of the lap (a local search around
the previous index), so where the track runs close to itself the label stays on the stretch the car
is driving; a global search takes over only when it is closer by more than a road half-width (the
car has left the stretch it was on, e.g. far out on the grass).

The frame render (97% of simulator time) is skipped: it does not touch the physics, and the reward
check confirms the replay is unchanged.

Writes data/latents_pose.npy, float32 (N, 4), aligned with the `actions` array of data/latents.npz
(row t = pose after action t, like data/latents_onroad.npy), and data/latents_pose0.npy, float32
(episodes, 4): the pose at reset, before the first action (the input for observation 0).
With --rollouts: <file>_pose.npy (aligned with its `act`) and <file>_pose0.npy next to each file.
"""

from __future__ import annotations

import argparse
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH

from .config import DATA, SEED_DATA
from .envs import make_env

ROLLOUTS = DATA / "rollouts"
POSE_DIM = 4
WINDOW = 8  # centre points searched either side of the previous nearest one


def pose_by_obs(obs_start, act_start, ep_len, pose, pose0) -> np.ndarray:
    """Raw pose for every observation, indexed like `mu`: observation t of episode e has the pose
    after action t-1 (pose0[e] for t = 0), so pose_obs[o + 1] is the label of the action at o."""
    n_obs = int((np.asarray(obs_start) + np.asarray(ep_len)).max() + 1)
    out = np.zeros((n_obs, POSE_DIM), np.float32)
    for e, (o, a, T) in enumerate(zip(obs_start, act_start, ep_len)):
        out[o] = pose0[e]
        out[o + 1 : o + T + 1] = pose[a : a + T]
    return out


def load_pose_obs(latents=DATA / "latents.npz", d=None) -> np.ndarray | None:
    """pose_by_obs for a latents npz from its sidecar files (None if it has not been labelled)."""
    latents = Path(latents)
    side, side0 = (latents.with_name(latents.stem + s) for s in ("_pose.npy", "_pose0.npy"))
    if not (side.exists() and side0.exists()):
        return None
    d = np.load(latents) if d is None else d
    return pose_by_obs(d["obs_start"], d["act_start"], d["ep_len"], np.load(side), np.load(side0))


class PoseTracker:
    """Pose of the car relative to the track centre line, with the nearest index followed step to step."""

    def __init__(self, env):
        u = env.unwrapped
        self.u = u
        self.xy = np.asarray(u.track, np.float64)[:, 2:4]
        self.n = len(self.xy)
        self.i = 0  # the car starts on centre point 0

    def _nearest(self, pos):
        idx = np.arange(self.i - WINDOW, self.i + WINDOW + 1) % self.n
        dl = ((self.xy[idx] - pos) ** 2).sum(1)
        loc = int(idx[np.argmin(dl)])
        dg = ((self.xy - pos) ** 2).sum(1)
        glo = int(np.argmin(dg))
        if math.sqrt(dg[glo]) + TRACK_WIDTH < math.sqrt(dl.min()):
            return glo
        return loc

    def __call__(self) -> np.ndarray:
        car = self.u.car
        pos = np.asarray(car.hull.position, np.float64)
        self.i = i = self._nearest(pos)
        best = None
        for a, b in (((i - 1) % self.n, i), (i, (i + 1) % self.n)):  # the two segments meeting at i
            seg = self.xy[b] - self.xy[a]
            L2 = float(seg @ seg) + 1e-12
            s = float(np.clip((pos - self.xy[a]) @ seg / L2, 0.0, 1.0))
            rel = pos - (self.xy[a] + s * seg)
            dist = float(np.hypot(*rel))
            if best is None or dist < best[0]:
                best = (dist, seg / math.sqrt(L2), rel)
        dist, d, rel = best
        lat = math.copysign(dist, d[0] * rel[1] - d[1] * rel[0]) / TRACK_WIDTH
        ang = car.hull.angle
        f = np.array([-math.sin(ang), math.cos(ang)])  # car forward vector
        err = math.atan2(d[0] * f[1] - d[1] * f[0], d @ f)  # car heading minus track tangent
        v = car.hull.linearVelocity
        return np.array([lat, math.sin(err), math.cos(err), math.hypot(v[0], v[1])], np.float32)


def replay_pose(seed: int, actions: np.ndarray):
    """Replay `actions` from track `seed`; returns (pose0 (4,), pose (T,4), rewards (T,), onroad (T,))."""
    env = make_env(disable_env_checker=True)
    env.unwrapped._render = lambda *a, **k: None  # physics only (see the module docstring)
    env.reset(seed=int(seed))
    tr = PoseTracker(env)
    pose0, pose, rew, on = tr(), [], [], []
    for a in actions:
        rew.append(env.step(np.asarray(a, np.float32))[1])
        pose.append(tr())
        on.append(any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels))
    env.close()
    return pose0, np.asarray(pose, np.float32).reshape(-1, POSE_DIM), np.asarray(rew), np.asarray(on, bool)


def label_episode(ep: int):
    sh = np.load(ROLLOUTS / f"ep_{ep:05d}.npz")
    pose0, pose, rew, on = replay_pose(SEED_DATA + ep, sh["actions"])
    ok = abs(rew.sum() - float(sh["rewards"].sum())) < 1e-2  # replay reproduced the recording
    return ep, ok, pose0, pose, on


def label_rollout_track(job):
    seed, act, rew_ref, on_ref = job
    pose0, pose, rew, on = replay_pose(seed, act)
    ok = abs(rew.sum() - float(rew_ref.sum())) < 1e-2 and bool((on == on_ref).all())
    return int(seed), ok, pose0, pose


def label_rollouts(paths, workers):
    for path in paths:
        z = np.load(path)
        jobs = [(s, z["act"][b : b + n], z["rew"][b : b + n], z["onroad"][b : b + n].astype(bool))
                for s, b, n in zip(z["seed"], z["start"], z["length"])]
        with Pool(workers) as pool:
            res = {r[0]: r for r in pool.imap_unordered(label_rollout_track, jobs)}
        res = [res[int(s)] for s in z["seed"]]
        bad = [r[0] for r in res if not r[1]]
        pose = np.concatenate([r[3] for r in res])
        assert len(pose) == len(z["act"])
        np.save(Path(path).with_name(Path(path).stem + "_pose.npy"), pose)
        np.save(Path(path).with_name(Path(path).stem + "_pose0.npy"), np.stack([r[2] for r in res]))
        print(f"{path}: {len(res)} tracks, {len(pose):,} steps; replays not matching: {len(bad)} {bad}",
              flush=True)
        assert not bad, "a replay did not reproduce its recording"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--rollouts", type=Path, nargs="*", default=None,
                   help="rollout files of diagnostics/dream_fidelity.py to label instead of the data")
    a = p.parse_args()
    if a.rollouts:
        return label_rollouts(a.rollouts, a.workers)
    d = np.load(DATA / "latents.npz")
    ep_len, act_start, actions = d["ep_len"], d["act_start"], d["actions"]
    n = len(ep_len)
    # Episode e of latents.npz must be shard ep_{e}: same actions, checked on the first steps.
    for e in range(0, n, 97):
        sh = np.load(ROLLOUTS / f"ep_{e:05d}.npz")
        s = int(act_start[e])
        assert ep_len[e] == len(sh["actions"]) and np.array_equal(
            actions[s : s + 20], sh["actions"][:20]
        ), f"episode {e} of latents.npz is not shard ep_{e:05d}"
    onroad_side = DATA / "latents_onroad.npy"
    on_ref = np.load(onroad_side) if onroad_side.exists() else None
    t0, bad, on_mismatch = time.time(), [], 0
    pose = np.zeros((len(actions), POSE_DIM), np.float32)
    pose0 = np.zeros((n, POSE_DIM), np.float32)
    with Pool(a.workers) as pool:
        for i, (ep, ok, p0, pq, on) in enumerate(pool.imap_unordered(label_episode, range(n)), 1):
            s = int(act_start[ep])
            assert len(pq) == ep_len[ep]
            pose[s : s + len(pq)], pose0[ep] = pq, p0
            if not ok:
                bad.append(ep)
            if on_ref is not None:
                on_mismatch += int((on_ref[s : s + len(pq)].astype(bool) != on).sum())
            if i % 100 == 0 or i == n:
                print(f"labelled {i}/{n}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    np.save(DATA / "latents_pose.npy", pose)
    np.save(DATA / "latents_pose0.npy", pose0)
    print(f"wrote {DATA / 'latents_pose.npy'}: {len(pose):,} steps; replays not matching the recorded "
          f"rewards: {len(bad)}; on-road flags differing from latents_onroad.npy: {on_mismatch}")


if __name__ == "__main__":
    main()
