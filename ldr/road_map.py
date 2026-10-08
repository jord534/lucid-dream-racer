"""A road map the car moves across (technical note 5.1, candidate 2; reports/road_map/).

The world model reads a road patch: road status at PATCH_UV, fixed points around the car (car frame,
u right, v forward; all inside the camera's view), plus the four wheels (some point of the footprint
on a road tile): MAP_DIM values in {0, 1}. In imagination, RoadMap holds the road (the tiles of the
track) and the car's pose; the pose moves by the model's own predicted motion and the patch is read at
it. With seen frames, the map holds only the road those camera frames showed; elsewhere it is unknown
and reads as road.
"""

from __future__ import annotations

import numpy as np

from .road_geometry import axes, in_view, points_in_tiles, to_world, wheel_points

PATCH_UV = np.array([(u, v) for v in (-4, 0, 4, 8, 12, 16) for u in (-12, -8, -4, 0, 4, 8, 12)], np.float64)
MAP_DIM = len(PATCH_UV) + 4                         # 42 points + 4 wheels
MOTION_SCALE = np.array([0.17, 0.38, 0.08])         # std of (du, dv, da) per step in the original data
NEAR = 24                                           # tiles nearest the car searched per read


def patch_points(pose):
    """World points of the patch and of the wheel footprints: (..., 42 + 20, 2)."""
    sh = pose.shape[:-1]
    grid = to_world(pose, np.broadcast_to(PATCH_UV, (*sh, *PATCH_UV.shape)))
    return np.concatenate([grid, wheel_points(pose).reshape(*sh, 20, 2)], -2)


def read_patch(pose, tiles, valid=None, seen_pose=None, seen_rect=None):
    """Road patch (N, MAP_DIM) float32 at car poses (N,3) on maps given by tiles (N,M,4,2) (+ valid
    (N,M)); with seen_pose (N,K,3) and seen_rect (N,K,4) only the road those frames showed is known."""
    N, M = tiles.shape[:2]
    pts = patch_points(pose)                                          # (N, 62, 2)
    if valid is None:
        valid = np.ones((N, M), bool)
    k = min(NEAR, M)
    if k < M:
        cent = tiles.mean(2)
        d = ((cent - pose[:, None, :2]) ** 2).sum(-1)
        d[~valid] = np.inf
        near = np.argpartition(d, k, axis=1)[:, :k]
        tiles = np.take_along_axis(tiles, near[..., None, None], 1)
        valid = np.take_along_axis(valid, near, 1)
    road = points_in_tiles(pts, tiles, valid)
    if seen_pose is not None:
        road |= ~in_view(seen_pose, seen_rect, pts)                   # unknown reads as road
    P = len(PATCH_UV)
    return np.concatenate([road[:, :P], road[:, P:].reshape(N, 4, 5).any(-1)], 1).astype(np.float32)


def wheels_from_patch(patch):
    """The four wheel values of a patch (..., MAP_DIM) -> (..., 4)."""
    return patch[..., len(PATCH_UV):]


def step_pose(pose, motion):
    """Move car poses (N,3) by one step's motion (N,3) (du, dv, da in the car frame)."""
    r, f = axes(pose[:, 2])
    return np.concatenate([pose[:, :2] + motion[:, :1] * r + motion[:, 1:2] * f, pose[:, 2:] + motion[:, 2:]], 1)


class RoadMap:
    """B maps with a car on each: tiles (B,M,4,2) padded with valid (B,M); pose (B,3);
    optional seen frames (B,K,3) and rects (B,K,4)."""

    def __init__(self, tiles, valid, pose, seen_pose=None, seen_rect=None):
        self.tiles, self.valid = np.asarray(tiles, np.float64), np.asarray(valid, bool)
        self.pose = np.asarray(pose, np.float64).copy()
        self.seen_pose, self.seen_rect = seen_pose, seen_rect

    def read(self):
        return read_patch(self.pose, self.tiles, self.valid, self.seen_pose, self.seen_rect)

    def move(self, motion):
        self.pose = step_pose(self.pose, np.asarray(motion, np.float64))

    @classmethod
    def stack(cls, tiles_list, pose, seen_pose=None, seen_rect=None):
        """From a list of per-sample tile arrays of different lengths."""
        M = max(len(t) for t in tiles_list)
        tiles = np.zeros((len(tiles_list), M, 4, 2))
        valid = np.zeros((len(tiles_list), M), bool)
        for i, t in enumerate(tiles_list):
            tiles[i, : len(t)], valid[i, : len(t)] = t, True
        return cls(tiles, valid, pose, seen_pose, seen_rect)
