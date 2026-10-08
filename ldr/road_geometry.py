"""Exact road geometry around the car, for a road map the car moves across (technical note 5.1,
candidate 2; reports/road_map/).

Conventions. A car pose is (x, y, angle) in world units, angle as Box2D's hull angle. The car's
forward vector is f = (-sin a, cos a), its right vector r = (cos a, sin a) (screen right when the
camera shows the car facing up). Car-frame coordinates of a world point p: u = (p - c).r (right),
v = (p - c).f (forward). Road tiles are convex quadrilaterals (world coordinates, the game's own
fixtures). A wheel is on the road when its footprint touches a tile; we test the footprint's four
corners and centre.

Motion of one step, in the car frame at the start of the step: (du, dv, da). integrate() and
motion_from_poses() convert between motion and poses.

All functions are numpy and broadcast over leading dimensions.
"""

from __future__ import annotations

import numpy as np
from gymnasium.envs.box2d.car_dynamics import SIZE, WHEEL_R, WHEEL_W, WHEELPOS
from gymnasium.envs.box2d.car_racing import FPS, SCALE, WINDOW_H, WINDOW_W, ZOOM

# wheel centres in the hull frame (u right, v forward) and the footprint's half extents
WHEEL_UV = np.array([(wx * SIZE, wy * SIZE) for wx, wy in WHEELPOS], np.float64)       # (4, 2)
WHEEL_HALF = np.array([WHEEL_W * SIZE, WHEEL_R * SIZE])                                 # (u, v)
_FOOT = np.array([(0, 0), (-1, -1), (1, -1), (1, 1), (-1, 1)], np.float64) * WHEEL_HALF  # (5, 2)
WHEEL_POINTS_UV = WHEEL_UV[:, None, :] + _FOOT[None]                                     # (4, 5, 2)


def axes(a):
    """Right and forward unit vectors of heading a: (..., 2) each."""
    c, s = np.cos(a), np.sin(a)
    return np.stack([c, s], -1), np.stack([-s, c], -1)


def to_world(pose, uv):
    """Car-frame points uv (..., P, 2) of car pose (..., 3) -> world points (..., P, 2)."""
    r, f = axes(pose[..., 2])
    return pose[..., None, :2] + uv[..., :1] * r[..., None, :] + uv[..., 1:2] * f[..., None, :]


def to_car(pose, p):
    """World points p (..., P, 2) -> car-frame (u, v) of car pose (..., 3)."""
    r, f = axes(pose[..., 2])
    d = p - pose[..., None, :2]
    return np.stack([(d * r[..., None, :]).sum(-1), (d * f[..., None, :]).sum(-1)], -1)


def wheel_points(pose):
    """World points of the four wheel footprints (corners and centre): (..., 4, 5, 2)."""
    sh = pose.shape[:-1]
    return to_world(pose, np.broadcast_to(WHEEL_POINTS_UV.reshape(20, 2), (*sh, 20, 2))).reshape(*sh, 4, 5, 2)


def points_in_tiles(p, tiles, valid=None):
    """p (..., P, 2) world points; tiles (..., M, 4, 2) convex quads (either winding); valid (..., M)
    mask for padded tiles. Returns (..., P) bool: the point lies in some tile."""
    e = np.roll(tiles, -1, -2) - tiles                                    # edges (..., M, 4, 2)
    d = p[..., :, None, None, :] - tiles[..., None, :, :, :]             # (..., P, M, 4, 2)
    cr = e[..., None, :, :, 0] * d[..., 1] - e[..., None, :, :, 1] * d[..., 0]
    inside = (cr >= 0).all(-1) | (cr <= 0).all(-1)                        # (..., P, M)
    if valid is not None:
        inside &= valid[..., None, :]
    return inside.any(-1)


def zoom_at(k):
    """The game's camera zoom (pixels per world unit) at observation k (k actions after reset)."""
    t = (np.asarray(k, np.float64) + 1) / FPS            # reset() takes one step without an action
    return 0.1 * SCALE * np.maximum(1 - t, 0) + ZOOM * SCALE * np.minimum(t, 1)


def visible_rect(k):
    """Car-frame rectangle the camera shows at observation k: (u_min, u_max, v_min, v_max).
    The car sits at the horizontal centre, a quarter of the window height above the bottom edge;
    the bottom eighth is covered by the dashboard."""
    z = zoom_at(k)
    behind = (WINDOW_H / 4 - WINDOW_H * 5 / 40) / z
    return np.stack([-WINDOW_W / 2 / z, WINDOW_W / 2 / z, -behind, WINDOW_H * 3 / 4 / z], -1)


def in_view(pose_k, rect_k, p):
    """p (..., P, 2) world points; pose_k (..., K, 3) and rect_k (..., K, 4) the camera frames that
    built the map. Returns (..., P) bool: seen in at least one frame."""
    K = pose_k.shape[-2]
    pts = np.broadcast_to(p[..., None, :, :], (*p.shape[:-2], K, *p.shape[-2:]))
    uv = to_car(pose_k, pts)                                              # (..., K, P, 2)
    r = rect_k[..., :, None, :]
    ok = (uv[..., 0] >= r[..., 0]) & (uv[..., 0] <= r[..., 1]) & (uv[..., 1] >= r[..., 2]) & (uv[..., 1] <= r[..., 3])
    return ok.any(-2)


def motion_from_poses(pose):
    """Poses (..., T+1, 3) -> per-step motion (..., T, 3): (du, dv, da) in the car frame at the start."""
    r, f = axes(pose[..., :-1, 2])
    d = pose[..., 1:, :2] - pose[..., :-1, :2]
    da = pose[..., 1:, 2] - pose[..., :-1, 2]
    return np.stack([(d * r).sum(-1), (d * f).sum(-1), da], -1)


def integrate(pose0, motion):
    """Start pose (..., 3) and motion (..., T, 3) -> poses (..., T+1, 3), the start included."""
    out = [np.asarray(pose0, np.float64)]
    for t in range(motion.shape[-2]):
        p = out[-1]
        r, f = axes(p[..., 2])
        m = motion[..., t, :]
        out.append(np.concatenate([p[..., :2] + m[..., :1] * r + m[..., 1:2] * f, p[..., 2:] + m[..., 2:]], -1))
    return np.stack(out, -2)


def env_tiles(env) -> np.ndarray:
    """Road tiles of the current track: (M, 4, 2) world coordinates."""
    return np.array([[tuple(v) for v in t.fixtures[0].shape.vertices] for t in env.unwrapped.road], np.float64)


def env_pose(env) -> np.ndarray:
    h = env.unwrapped.car.hull
    return np.array([h.position[0], h.position[1], h.angle], np.float64)


def env_wheel_points(env) -> np.ndarray:
    """The actual wheel footprints (front wheels steered) as corners and centre: (4, 5, 2)."""
    out = []
    for w in env.unwrapped.car.wheels:
        vs = [tuple(w.GetWorldPoint(v)) for v in w.fixtures[0].shape.vertices]
        out.append([tuple(w.position)] + vs)
    return np.array(out, np.float64)
