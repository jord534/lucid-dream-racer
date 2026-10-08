"""Road-map geometry (ldr/road_geometry.py): point tests, motion integration, the camera's view, and
agreement with the game's own wheel contacts; and the progress-line protocol."""
import importlib.util
from pathlib import Path

import numpy as np

from ldr.envs import make_env
from ldr.road_geometry import (env_pose, env_tiles, env_wheel_points, in_view, integrate, motion_from_poses,
                               points_in_tiles, to_car, to_world, visible_rect, wheel_points)


def test_points_in_tiles_either_winding_and_mask():
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    tiles = np.stack([sq, sq[::-1] + 2])                  # one counter-clockwise, one clockwise
    p = np.array([[0.5, 0.5], [2.5, 2.5], [1.5, 1.5], [1.0, 0.5]])
    assert points_in_tiles(p, tiles).tolist() == [True, True, False, True]   # edges count as inside
    assert points_in_tiles(p, tiles, np.array([True, False])).tolist() == [True, False, False, True]


def test_frames_and_motion_roundtrip():
    rng = np.random.default_rng(0)
    pose = np.array([3.0, -2.0, 0.7])
    uv = rng.standard_normal((5, 2))
    assert np.allclose(to_car(pose, to_world(pose, uv)), uv)
    assert np.allclose(to_car(np.array([0, 0, 0.0]), np.array([[0, 1.0]])), [[0, 1]])  # angle 0 faces +y
    poses = np.cumsum(rng.standard_normal((30, 3)) * [0.5, 0.5, 0.05], 0)
    assert np.allclose(integrate(poses[0], motion_from_poses(poses)), poses)


def test_camera_view():
    u0, u1, v0, v1 = visible_rect(np.array(100))
    assert np.allclose([u0, u1, v0, v1], [-1000 / 32.4, 1000 / 32.4, -100 / 16.2, 600 / 16.2])
    assert visible_rect(np.array(10))[3] > v1              # zooming in during the first second
    pose = np.array([[5.0, 5.0, 1.0]])
    pts = to_world(pose[0], np.array([[0, 10.0], [0, 45.0], [20, 0], [0, -10.0]]))
    assert in_view(pose, visible_rect(np.array([100])), pts).tolist() == [True, False, True, False]


def test_geometry_matches_the_game():
    env = make_env(disable_env_checker=True)
    env.unwrapped._render = lambda *a, **k: None
    env.reset(seed=11)
    tiles = env_tiles(env)
    rng = np.random.default_rng(1)
    poses, exact, flags = [env_pose(env)], [env_wheel_points(env)], []
    for t in range(150):                                   # drive, then swerve off the road
        a = np.array([0.0 if t < 60 else 0.8, 0.5, 0.0], np.float32) + rng.normal(0, 0.05, 3).astype(np.float32)
        env.step(np.clip(a, [-1, 0, 0], [1, 1, 1]))
        flags.append(any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels))
        poses.append(env_pose(env))
        exact.append(env_wheel_points(env))
    env.close()
    poses, exact, flags = np.array(poses), np.array(exact), np.array(flags)
    assert np.abs(wheel_points(poses)[:, :, 0] - exact[:, :, 0]).max() < 1e-3   # wheel centres from the pose
    geo = points_in_tiles(wheel_points(poses).reshape(-1, 20, 2), tiles[None]).reshape(-1, 4, 5).any(-1).any(1)
    assert (~flags).sum() > 10                             # the car did leave the road
    # the game's flag after action t describes the wheels before that action (contact is a step late)
    assert (geo[:-1] == flags).mean() >= 0.99 and (geo[:-1] == flags).mean() > (geo[1:] == flags).mean()


def test_progress_lines_parse(capsys):
    from ldr.progress import plan, progress, result
    plan({"label data": 10})
    progress("label data", 0, 4)
    progress("label data", 4, 4)
    result("ok")
    spec = importlib.util.spec_from_file_location("pw", Path(__file__).parents[1] / "scripts" / "progress_watch.py")
    pw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pw)
    pl, prog, res, _ = pw.parse(capsys.readouterr().out)
    assert pl == {"label_data": 10.0} and prog["label_data"]["done"] == 4 and res == ["ok"]
