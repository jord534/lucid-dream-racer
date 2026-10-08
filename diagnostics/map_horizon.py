"""Stage 4 of the road-map experiment: does a road map keep the world model on the road?
(reports/road_map/preregistration.md)

The same windows, independent road probe and readout as diagnostics/pose_horizon.py (and so as
head_horizon.py and technical note 3.8): real actions replayed for up to 100 steps after up to 40 real
steps, 16 samples per window, P(on road) read from each imagined latent. What drives a road-map model:
  real            real frames, and the true road patch at the car's true pose
  own             own latent samples, the car moved by the model's own motion across the map of the
                  road the camera showed during the history (unknown reads as road)   <- primary
  own, whole-track map   the same on a map of the whole track
  oracle          own latent samples, the true patch at the true pose
Models without a map: real and own only. Also, for road-map models: road status read off the map at the
wheels, and the error of the car's position and heading against horizon.
Sets: the test-track runs (as pose_horizon; evaluated once per final checkpoint), the held-out
validation episodes of the original data, and the development runs of stage 1 (data/road_map).

    python diagnostics/map_horizon.py --model runs/road_map/map_s0/best.pt --name map_s0 --sets dev val test --tau 1.15 0.05
    python diagnostics/map_horizon.py --gate --map map_s0 map_s1 --control ctrl_s0 ctrl_s1
Writes reports/road_map/horizon/<name>_tau<tau>.json/.npz, and with --gate reports/road_map/stage4_gate*.json."""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from head_horizon import BINS, H, LEAD, WARM, B, first_off_run, rollout_episodes
from ldr.check_dream import load_mdnrnn
from ldr.config import DATA, REPORTS, C
from ldr.mdnrnn import mdn_sample
from ldr.progress import progress, result
from ldr.road_geometry import visible_rect
from ldr.road_map import MOTION_SCALE, RoadMap, read_patch, wheels_from_patch
from pose_horizon import KINDS, SETS, summarise_counts, window_counts, windows_like_head_horizon

torch.set_num_threads(4)
OUT = REPORTS / "road_map" / "horizon"
POSE_OUT = REPORTS / "pose_state" / "horizon"
GEOM = DATA / "road_map"
CHUNK = 64
DEV = "development runs (validation tracks, six dream-trained drivers)"
SET_NAMES = {**SETS, "dev": DEV}
PRIMARY = f"{SETS['test']} | 30 steps before leaving the road"
MODES = {  # name: (latent fed, car pose / patch source)
    "real": ("real", "true"),
    "own": ("own", "seen"),
    "own, whole-track map": ("own", "full"),
    "oracle (own latent, true pose)": ("own", "true"),
}


# ---------------------------------------------------------------- windows and geometry

_GEOM = {}


def geom_for(name):
    """Geometry (true poses, obs-aligned, and tiles) of a window's run, from its name."""
    src, _, seed = name.partition(":")
    if src.startswith("base"):
        path, key = GEOM / "geom_data.npz", int(src[4:])
    else:
        tag = src.replace("rollouts_", "test_").replace("devruns_", "dev_")
        path, key = GEOM / f"geom_{tag}.npz", int(seed)
    if path not in _GEOM:
        g = np.load(path)
        _GEOM[path] = ({k: g[k] for k in ("pose", "tiles", "obs_start", "tile_start", "n_tiles", "ep_len")},
                       {int(s): i for i, s in enumerate(g["seed"])})
    g, by_seed = _GEOM[path]
    i = key if src.startswith("base") else by_seed[key]
    o, T, ts, nt = (int(g[k][i]) for k in ("obs_start", "ep_len", "tile_start", "n_tiles"))
    return g["pose"][o : o + T + 1].astype(np.float64), g["tiles"][ts : ts + nt].astype(np.float64)


def dev_windows():
    rng = np.random.default_rng(1)
    eps = [ep for f in sorted(GEOM.glob("devruns_*.npz")) if "smoke" not in f.name for ep in rollout_episodes(f)]
    out = {}
    for kind in KINDS:
        wins = []
        for ep in eps:
            T = len(ep["act"])
            if kind == "random start":
                t0s = [int(rng.integers(WARM, max(WARM + 1, T - H - 1))) for _ in range(3)] if T > WARM + H + 2 else []
            else:
                o = first_off_run(ep["on"])
                t0s = [] if o is None else [o + 1 - LEAD]
            for t0 in t0s:
                W = min(H, T - 1 - t0)
                if t0 >= 1 and W >= 20:
                    wins.append((ep, t0, W))
        out[f"{DEV} | {kind}"] = wins
    return out


# ---------------------------------------------------------------- one chunk of windows

@torch.no_grad()
def run_chunk(rnn, probe, wins, tau, gen):
    """Returns {readout: (P(on) (H,N,B), extra)} where extra holds, for map readouts, the car's poses
    (H,N,B,3); plus truth (H,N), mask (H,N) and the true poses (H,N,3)."""
    N, k = len(wins), min(WARM, wins[0][1])
    R, L = N * B, k + H + 1
    mu = np.zeros((N, L, C.z_dim), np.float32)
    lv = np.full((N, L, C.z_dim), -20, np.float32)
    act = np.zeros((N, L, 3), np.float32)
    truth, mask = np.ones((N, H), bool), np.zeros((N, H), bool)
    pmap = rnn.map_input
    if pmap:
        tpose = np.zeros((N, L, 3))
        tiles = []
    for n, (ep, t0, W) in enumerate(wins):
        s = t0 - k
        mu[n, : k + W + 1], lv[n, : k + W + 1] = ep["mu"][s : t0 + W + 1], ep["lv"][s : t0 + W + 1]
        act[n, : k + W] = ep["act"][s : t0 + W]
        truth[n, :W], mask[n, :W] = ep["on"][t0 : t0 + W], True
        if pmap:
            pose, til = geom_for(ep["name"])
            seg = pose[s : t0 + W + 1]
            tpose[n] = np.concatenate([seg, np.repeat(seg[-1:], L - len(seg), 0)])
            tiles.append(til)
    rep = lambda x: torch.from_numpy(x).repeat_interleave(B, 0)
    mu_t, sd_t, act_t = rep(mu), rep(np.exp(0.5 * lv)), rep(act)
    samp = lambda i: mu_t[:, i] + torch.randn(R, C.z_dim, generator=gen) * sd_t[:, i]
    if pmap:
        base = RoadMap.stack(tiles, tpose[:, 0])
        true_patch = np.stack([read_patch(tpose[n], np.broadcast_to(base.tiles[n], (L, *base.tiles.shape[1:])),
                                          np.broadcast_to(base.valid[n], (L, base.valid.shape[1]))) for n in range(N)])
        true_patch_t = rep(true_patch)
        frames = np.arange(-k, 1)                       # history frames, relative to the start
        seen_pose = np.repeat(tpose[:, : k + 1], B, 0)
        seen_rect = np.repeat(visible_rect(np.array([w[1] for w in wins])[:, None] + frames[None]), B, 0)
    pin = lambda i: true_patch_t[:, i] if pmap else None
    st0 = rnn.initial_state(R, "cpu")
    for i in range(k):
        *_, st0 = rnn.step(samp(i), act_t[:, i], st0, patch=pin(i))
    modes = MODES if pmap else {m: v for m, v in MODES.items() if m in ("real", "own")}
    res = {}
    for mode, (zs, ps) in modes.items():
        st, z = tuple(s.clone() for s in st0), samp(k)
        rmap = None
        if pmap and ps in ("seen", "full"):
            rmap = RoadMap(np.repeat(base.tiles, B, 0), np.repeat(base.valid, B, 0), np.repeat(tpose[:, k], B, 0),
                           *((seen_pose, seen_rect) if ps == "seen" else ()))
        p_own, p_real, poses, wheel_on = [], [], [], []
        for j in range(H):
            if rmap is not None:
                patch = rmap.read()
                poses.append(rmap.pose.copy())
                wheel_on.append(wheels_from_patch(patch).any(1))
                patch = torch.from_numpy(patch)
            else:
                patch = pin(k + j)
            o = rnn.step(z, act_t[:, k + j], st, patch=patch, return_motion=pmap)
            st = o[5]
            own = mdn_sample(o[0], o[1], o[2], tau)
            real = samp(k + j + 1)
            p_own.append(torch.sigmoid(probe(own)[:, 0]))
            if mode == "real":
                p_real.append(torch.sigmoid(probe(real)[:, 0]))
            if rmap is not None:
                rmap.move(o[-1].numpy().astype(np.float64) * MOTION_SCALE)
            z = real if zs == "real" else own
        sh = lambda v: np.stack(v).reshape(H, N, B, *np.asarray(v[0]).shape[1:])
        if mode == "real":
            res["real: probe on real frame"] = (sh([x.numpy() for x in p_real]), None)
            res["real: probe on 1-step prediction"] = (sh([x.numpy() for x in p_own]), None)
        else:
            res[mode] = (sh([x.numpy() for x in p_own]), sh(poses) if rmap is not None else None)
            if rmap is not None:
                res[f"{mode}: road read off the map"] = (sh(wheel_on).astype(np.float32), None)
    return res, truth.T, mask.T, (tpose[:, k : k + H].transpose(1, 0, 2) if pmap else None)


def pose_counts(poses, true_pose, mask):
    """Per window and bin: summed position error (units) and heading error (degrees) of the car."""
    err = np.linalg.norm(poses[..., :2] - true_pose[:, :, None, :2], axis=-1)
    dh = np.abs((poses[..., 2] - true_pose[:, :, None, 2] + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
    m = mask[..., None]
    out = {}
    for bi, (a, b) in enumerate(BINS):
        out[bi] = {"pos_err": np.where(m, err, 0)[a:b].sum((0, 2)), "hdg_err": np.where(m, dh, 0)[a:b].sum((0, 2))}
    return out


def evaluate(rnn, probe, wins, tau, seed, task, done0, total):
    gen = torch.Generator().manual_seed(seed)
    order = sorted(range(len(wins)), key=lambda i: min(WARM, wins[i][1]))
    counts, idx, i, done = {}, [], 0, done0
    while i < len(order):
        k = min(WARM, wins[order[i]][1])
        grp = [j for j in order[i:] if min(WARM, wins[j][1]) == k][:CHUNK]
        res, truth, mask, tpose = run_chunk(rnn, probe, [wins[j] for j in grp], tau, gen)
        for name, (p, poses) in res.items():
            cnt = window_counts(p, None, truth, None, mask)
            if poses is not None:
                for bi, c in pose_counts(poses, tpose, mask).items():
                    cnt[bi].update(c)
            for bi, c in cnt.items():
                for key, v in c.items():
                    counts.setdefault(name, {}).setdefault(bi, {}).setdefault(key, []).append(v)
        idx += grp
        i += len(grp)
        done += len(grp)
        progress(task, done, total)
    inv = np.argsort(idx)
    return {n: {bi: {k: np.concatenate(v)[inv] for k, v in c.items()} for bi, c in byb.items()}
            for n, byb in counts.items()}, done


def summarise(c):
    r = summarise_counts(c)
    if "pos_err" in c:
        n = max(c["n"].sum(), 1)
        r.update(pos_err=float(c["pos_err"].sum() / n), hdg_err_deg=float(c["hdg_err"].sum() / n))
    return r


def measure(a):
    rnn = load_mdnrnn(a.model)
    print(f"{a.model}: road map {rnn.map_dim} (trained {rnn.has_map})", flush=True)
    probe, wins = windows_like_head_horizon({s for s in a.sets if s != "dev"} or {"val"})
    wins = {k: v for k, v in wins.items() if any(k.startswith(SET_NAMES[s]) for s in a.sets)}
    if "dev" in a.sets:
        wins.update(dev_windows())
    task = os.environ.get("LDR_PROGRESS_TASK", f"measure {a.name}")
    total, done = sum(len(w) for w in wins.values()) * len(a.tau), 0
    progress(task, 0, total)
    OUT.mkdir(parents=True, exist_ok=True)
    for tau in a.tau:
        t0 = time.time()
        out = {"model": str(a.model), "name": a.name, "tau": tau, "sets": a.sets, "results": {}}
        arrays = {}
        for key, w in wins.items():
            cnt, done = evaluate(rnn, probe, w, tau, a.seed, task, done, total)
            out["results"][key] = {"windows": len(w), "readouts": {
                n: {f"{x}-{y}": summarise(c) for (x, y), c in zip(BINS, byb.values())} for n, byb in cnt.items()}}
            for n, byb in cnt.items():
                for bi, c in byb.items():
                    for k2, v in c.items():
                        arrays[f"{key}||{n}||{bi}||{k2}"] = v
        out["minutes"] = (time.time() - t0) / 60
        stem = OUT / f"{a.name}_tau{tau}"
        Path(f"{stem}.json").write_text(json.dumps(out, indent=1))
        np.savez_compressed(f"{stem}.npz", **arrays)
        for key, r in out["results"].items():
            print(f"\n{key} ({r['windows']} windows), tau {tau}: accuracy / off-road recall by steps since the start")
            for n, byb in r["readouts"].items():
                f = lambda v: "-" if v is None else f"{v[0]:.2f}"
                print(f"  {n:42s} " + "  ".join(f"{b}: {f(s['acc'])}/{f(s['off_recall'])}" for b, s in byb.items()))


def counts_for(name, tau, readout="own"):
    """Per-window counts of M (steps 20-40 of the test departure windows) for a model's run: road-map
    runs from reports/road_map/horizon, controls from reports/pose_state/horizon (same windows)."""
    f = OUT / f"{name}_tau{tau}.npz"
    z = np.load(f if f.exists() else POSE_OUT / f"{name}_tau{tau}.npz")
    k = f"{PRIMARY}||{readout}||3"
    return z[f"{k}||off_correct"].astype(float), z[f"{k}||n_off"].astype(float)


def gate(a):
    for tau in a.tau:
        rng = np.random.default_rng(0)
        c = {n: counts_for(n, tau) for n in a.map + a.control}
        n_w = len(next(iter(c.values()))[0])
        boot = rng.integers(n_w, size=(2000, n_w))
        m = {n: float(x.sum() / d.sum()) for n, (x, d) in c.items()}
        ci = {n: np.percentile(x[boot].sum(1) / d[boot].sum(1), [2.5, 97.5]).tolist() for n, (x, d) in c.items()}
        pairs = {}
        for mn, cn in zip(a.map, a.control):
            (xm, dm), (xc, dc) = c[mn], c[cn]
            diff = xm[boot].sum(1) / dm[boot].sum(1) - xc[boot].sum(1) / dc[boot].sum(1)
            pairs[f"{mn} - {cn}"] = {"diff": m[mn] - m[cn], "ci95": np.percentile(diff, [2.5, 97.5]).tolist()}
        mm, mc = np.mean([m[n] for n in a.map]), np.mean([m[n] for n in a.control])
        crit = {"1 every map seed M >= 0.80": all(m[n] >= 0.80 for n in a.map),
                "2a mean map - mean control >= 0.10": bool(mm - mc >= 0.10),
                "2b lowest map seed above highest control": min(m[n] for n in a.map) > max(m[n] for n in a.control),
                "2c every paired 95% interval above zero": all(v["ci95"][0] > 0 for v in pairs.values())}
        out = {"tau": tau, "windows": n_w, "M": m, "M_ci95": ci, "mean_map": mm, "mean_control": mc,
               "paired": pairs, "criteria": crit, "passed": all(crit.values())}
        name = "stage4_gate.json" if tau == 1.15 else f"stage4_gate_tau{tau}.json"
        (REPORTS / "road_map" / name).write_text(json.dumps(out, indent=1))
        if tau != 1.15:
            continue
        for n in m:
            result(f"M {n}: {m[n]:.3f} [{ci[n][0]:.3f}, {ci[n][1]:.3f}]")
        for k, v in crit.items():
            result(f"{'PASS' if v else 'FAIL'}  {k}")
        result(f"STAGE 4 {'PASSED' if out['passed'] else 'FAILED'}  (reports/road_map/{name})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path)
    p.add_argument("--name")
    p.add_argument("--sets", nargs="+", default=["dev", "val"], choices=["dev", "val", "test"])
    p.add_argument("--tau", type=float, nargs="+", default=[C.tau])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gate", action="store_true")
    p.add_argument("--map", nargs="+", default=["map_s0", "map_s1"])
    p.add_argument("--control", nargs="+", default=["ctrl_s0", "ctrl_s1"])
    a = p.parse_args()
    gate(a) if a.gate else measure(a)


if __name__ == "__main__":
    main()
