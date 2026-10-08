"""Stage 3 of the road-map experiment: road-patch and motion labels for the world model's training
data (reports/road_map/preregistration.md).

    python -m ldr.label_patch --workers 6          (needs data/road_map/geom_data.npz from ldr.label_geometry)

For every observation of data/latents.npz: the road patch at the car's true pose (ldr.road_map; the
whole track, which at these points equals what the camera shows) and the car's exact motion to the next
observation (zero after the last action). Writes data/latents_patch.npy (uint8, N_obs x MAP_DIM) and
data/latents_motion.npy (float32, N_obs x 3), indexed like `mu`, and reports/road_map/stage3_labels.json:
agreement of the patch's wheel values with the game's on-road flag (the flag after action t describes
the wheels at observation t, stage 1), and of the nearest-tile search with a search over every tile."""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool

import numpy as np

from .config import DATA, REPORTS
from .progress import progress
from .road_geometry import motion_from_poses, points_in_tiles
from .road_map import MAP_DIM, PATCH_UV, patch_points, read_patch, wheels_from_patch

GEOM = DATA / "road_map" / "geom_data.npz"
_G = None


def _init():
    global _G
    _G = np.load(GEOM)


def label(e):
    g = _G
    o, T, ts, nt = (int(g[k][e]) for k in ("obs_start", "ep_len", "tile_start", "n_tiles"))
    pose = g["pose"][o : o + T + 1].astype(np.float64)
    tiles = g["tiles"][ts : ts + nt].astype(np.float64)
    patch = read_patch(pose, np.broadcast_to(tiles, (T + 1, *tiles.shape)))
    sample = np.arange(0, T + 1, 25)                                   # check the nearest-tile search
    full = _all_tiles(pose[sample], tiles)
    motion = np.zeros((T + 1, 3), np.float32)
    motion[:T] = motion_from_poses(pose)
    return e, patch.astype(np.uint8), motion, float((full == patch[sample]).mean())


def _all_tiles(pose, tiles):
    pts = patch_points(pose)
    road = points_in_tiles(pts, tiles[None])
    P = len(PATCH_UV)
    return np.concatenate([road[:, :P], road[:, P:].reshape(len(pose), 4, 5).any(-1)], 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    g = np.load(GEOM)
    d = np.load(DATA / "latents.npz")
    assert np.array_equal(g["obs_start"], d["obs_start"]) and np.array_equal(g["ep_len"], d["ep_len"]), \
        "geometry and latents are not aligned"
    n_obs, n = len(d["mu"]), len(g["ep_len"])
    patch = np.zeros((n_obs, MAP_DIM), np.uint8)
    motion = np.zeros((n_obs, 3), np.float32)
    near_ok = []
    progress("patch labels", 0, n)
    with Pool(a.workers, initializer=_init) as pool:
        for i, (e, pe, me, ok) in enumerate(pool.imap_unordered(label, range(n), chunksize=8), 1):
            o, T = int(g["obs_start"][e]), int(g["ep_len"][e])
            patch[o : o + T + 1], motion[o : o + T + 1] = pe, me
            near_ok.append(ok)
            progress("patch labels", i, n)
    np.save(DATA / "latents_patch.npy", patch)
    np.save(DATA / "latents_motion.npy", motion)
    wheels = wheels_from_patch(patch).any(1)
    agree = np.concatenate([wheels[o : o + T] == g["game_on"][a_ : a_ + T]
                            for o, a_, T in zip(g["obs_start"], g["act_start"], g["ep_len"])]).mean()
    out = {"wheel_agreement_with_game": float(agree), "nearest_tile_search_agreement": float(np.mean(near_ok)),
           "patch_road_share": float(patch[:, :-4].mean()), "observations": int(n_obs)}
    (REPORTS / "road_map").mkdir(parents=True, exist_ok=True)
    (REPORTS / "road_map" / "stage3_labels.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
