"""Stage 1 of the road-map experiment: exact road geometry for every recorded run, and the check
that it reproduces the game's own road status (reports/road_map/preregistration.md).

    python -m ldr.label_geometry --workers 6        (a few minutes; sources already written are skipped)

Every run is replayed from its track seed with its recorded actions, the frame render skipped (it
does not touch the physics; the reward check confirms it). Sources:
  data  the 1,400 episodes of data/latents.npz (seed SEED_DATA + episode)
  test  the six test-track runs of diagnostics/dream_fidelity.py (data/dream_failure/rollouts_*_r02_p*.npz)
  dev   the validation-track development runs (data/road_map/devruns_*.npz)
Writes data/road_map/geom_<source>[_<tag>].npz with, per run: seed, ep_len, obs_start (T+1
observations each), act_start, tile_start, n_tiles, replay_ok; per observation: pose (x, y, angle)
and the four wheels' road status from the actual wheel shapes (wheel_exact) and from the pose alone
(wheel_geo, what the road map reads); per action: game_on, the game's on-road flag, and the
action; and the tiles.
Then compares both readouts with the game's flag, in both alignments, and writes
reports/road_map/stage1.json with the pass/fail of the preregistered stage-1 criteria."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .config import DATA, REPORTS, SEED_DATA
from .envs import make_env
from .progress import progress, result
from .road_geometry import env_pose, env_tiles, env_wheel_points, points_in_tiles, wheel_points

OUT = DATA / "road_map"
TEST_FILES = [DATA / "dream_failure" / f"rollouts_{t}_r02_p{i}.npz" for t in ("loop_head", "loop_road") for i in range(3)]


def replay(seed: int, actions: np.ndarray):
    env = make_env(disable_env_checker=True)
    env.unwrapped._render = lambda *a, **k: None   # physics only
    env.reset(seed=int(seed))
    tiles = env_tiles(env)
    pose, wex, rew, on = [env_pose(env)], [env_wheel_points(env)], [], []
    for a in actions:
        rew.append(env.step(np.asarray(a, np.float32))[1])
        on.append(any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels))
        pose.append(env_pose(env))
        wex.append(env_wheel_points(env))
    env.close()
    pose, wex = np.array(pose), np.array(wex)
    exact = points_in_tiles(wex.reshape(-1, 20, 2), tiles[None]).reshape(-1, 4, 5).any(-1)
    geo = points_in_tiles(wheel_points(pose).reshape(-1, 20, 2), tiles[None]).reshape(-1, 4, 5).any(-1)
    return tiles, pose, exact, geo, np.array(rew), np.array(on, bool)


def job_data(ep):
    sh = np.load(DATA / "rollouts" / f"ep_{ep:05d}.npz")
    tiles, pose, exact, geo, rew, on = replay(SEED_DATA + ep, sh["actions"])
    ok = abs(rew.sum() - float(sh["rewards"].sum())) < 1e-2
    return ep, SEED_DATA + ep, ok, tiles, pose, exact, geo, on, sh["actions"]


def job_log(args):
    i, seed, act, rew_ref, on_ref = args
    tiles, pose, exact, geo, rew, on = replay(seed, act)
    ok = abs(rew.sum() - float(rew_ref.sum())) < 1e-2 and bool((on == on_ref).all())
    return i, seed, ok, tiles, pose, exact, geo, on, act


def write(path: Path, rows):
    rows = sorted(rows, key=lambda r: r[0])
    T = np.array([len(r[7]) for r in rows])
    nt = np.array([len(r[3]) for r in rows])
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, seed=np.array([r[1] for r in rows]), replay_ok=np.array([r[2] for r in rows]), ep_len=T,
        obs_start=np.r_[0, np.cumsum(T + 1)[:-1]], act_start=np.r_[0, np.cumsum(T)[:-1]],
        tile_start=np.r_[0, np.cumsum(nt)[:-1]], n_tiles=nt,
        tiles=np.concatenate([r[3] for r in rows]).astype(np.float32),
        pose=np.concatenate([r[4] for r in rows]).astype(np.float32),
        wheel_exact=np.concatenate([r[5] for r in rows]), wheel_geo=np.concatenate([r[6] for r in rows]),
        game_on=np.concatenate([r[7] for r in rows]),
        actions=np.concatenate([r[8] for r in rows]).astype(np.float32))
    tmp.rename(path)


def run_source(name, fn, jobs, workers, out=OUT):
    path = out / f"geom_{name}.npz"
    task = f"label {name}"
    if path.exists():
        progress(task, len(jobs), len(jobs))
        return path
    rows = []
    progress(task, 0, len(jobs))
    with Pool(workers) as pool:
        for r in pool.imap_unordered(fn, jobs, chunksize=4):
            rows.append(r)
            progress(task, len(rows), len(jobs))
    write(path, rows)
    return path


def agreement(path):
    g = np.load(path)
    s = {"runs": int(len(g["seed"])), "replays_not_matching": int((~g["replay_ok"]).sum()), "steps": int(len(g["game_on"]))}
    for key in ("wheel_exact", "wheel_geo"):
        any_on = g[key].any(1)
        before, after = [], []
        for o, a, T in zip(g["obs_start"], g["act_start"], g["ep_len"]):
            before.append(any_on[o : o + T] == g["game_on"][a : a + T])        # wheels at obs t vs flag after action t
            after.append(any_on[o + 1 : o + T + 1] == g["game_on"][a : a + T])  # wheels at obs t+1
        s[key] = {"agree_before_action": float(np.concatenate(before).mean()),
                  "agree_after_action": float(np.concatenate(after).mean())}
    return s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=None, help="first runs of each source only (smoke test)")
    p.add_argument("--out", type=Path, default=OUT, help="where the geometry files go")
    p.add_argument("--report", type=Path, default=REPORTS / "road_map" / "stage1.json")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    d = np.load(DATA / "latents.npz")
    paths = {"data": run_source("data", job_data, list(range(len(d["ep_len"])))[: a.limit], a.workers, a.out)}
    devruns = sorted(f for f in OUT.glob("devruns_*.npz")   # smoke-test runs only in a smoke test
                     if ("smoke" not in f.name or a.limit) and ".tmp" not in f.name)
    for f in TEST_FILES + devruns:
        z = np.load(f)
        kind = "test" if f in TEST_FILES else "dev"
        tag = f.stem.replace("rollouts_", "").replace("devruns_", "")
        jobs = [(i, int(s), z["act"][b : b + n], z["rew"][b : b + n], z["onroad"][b : b + n].astype(bool))
                for i, (s, b, n) in enumerate(zip(z["seed"], z["start"], z["length"]))][: a.limit]
        paths[f"{kind}_{tag}"] = run_source(f"{kind}_{tag}", job_log, jobs, a.workers, a.out)
    # the game's flag in the data must also match data/latents_onroad.npy (labelled independently)
    g = np.load(paths["data"])
    side = np.load(DATA / "latents_onroad.npy").astype(bool)
    stats = {"sources": {k: agreement(v) for k, v in paths.items()},
             "data_flags_equal_latents_onroad": bool(np.array_equal(g["game_on"], side[: len(g["game_on"])]))}
    sd = stats["sources"]["data"]
    align = max(("before_action", "after_action"), key=lambda k: sd["wheel_exact"][f"agree_{k}"])
    crit = {
        "1 every replay reproduces its recording": all(v["replays_not_matching"] == 0 for v in stats["sources"].values())
        and stats["data_flags_equal_latents_onroad"],
        "2 actual wheel shapes agree with the game >= 0.995": sd["wheel_exact"][f"agree_{align}"] >= 0.995,
        "3 wheels from the pose agree with the game >= 0.98": sd["wheel_geo"][f"agree_{align}"] >= 0.98,
    }
    stats.update(alignment=align, criteria=crit, passed=all(crit.values()))
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(stats, indent=1))
    result(f"alignment: the game's flag after action t describes the wheels {align.replace('_', ' ')}")
    result(f"agreement on the original data: actual wheels {sd['wheel_exact'][f'agree_{align}']:.4f}, "
           f"wheels from the pose {sd['wheel_geo'][f'agree_{align}']:.4f}")
    for k, v in crit.items():
        result(f"{'PASS' if v else 'FAIL'}  {k}")
    result(f"STAGE 1 {'PASSED' if stats['passed'] else 'FAILED'}  ({a.report})")


if __name__ == "__main__":
    main()
