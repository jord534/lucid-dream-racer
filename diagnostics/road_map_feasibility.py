"""Stage 2 of the road-map experiment: does a map stay right for 20-40 steps? No world model involved.
(reports/road_map/preregistration.md)

For each window (as in head_horizon.py: 30 steps before the car first leaves the road, or a random
start; up to 100 steps; 40 real steps of history), the map holds the road tiles the camera showed
during the history (each frame's exact visible rectangle); elsewhere it is unknown, read as road.
The car is moved across it by true motion, held motion (the last real step repeated) or guessed
motion (a small GRU, trained here on the training episodes of the original data, fed the actions,
its own previous motion and which wheels the map shows on the road). Road status at each step: some
wheel touches a road tile of the map. The truth is the game's own flag, whose one-step delay
(stage 1) is respected: the flag after action t describes the wheels at observation t.

    python diagnostics/road_map_feasibility.py          (trains the guesser if runs/road_map/motion_gru.pt
                                                          is missing, then evaluates; ~15 min)
Writes reports/road_map/stage2.json, stage2_counts.npz and prints the preregistered gate."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from head_horizon import BINS, H, LEAD, WARM, first_off_run
from ldr.config import DATA, REPORTS, RUNS
from ldr.progress import plan, progress, result
from ldr.road_geometry import (axes, in_view, motion_from_poses, points_in_tiles, visible_rect,
                               wheel_points)

torch.set_num_threads(4)
GEOM = DATA / "road_map"
OUT = REPORTS / "road_map"
GRU_PATH = RUNS / "road_map" / "motion_gru.pt"
NEAR = 12          # tiles nearest the car tested per step (the wheels lie within 2 units of the hull)
CHUNK = 128
MOTIONS = ("true", "held", "guessed")
MAPS = ("seen", "full")


# ---------------------------------------------------------------- data

def load_runs(path):
    g = np.load(path)
    runs = []
    for i, (o, a, T, ts, nt) in enumerate(zip(g["obs_start"], g["act_start"], g["ep_len"], g["tile_start"], g["n_tiles"])):
        pose = g["pose"][o : o + T + 1].astype(np.float64)
        runs.append(dict(name=f"{path.stem}:{int(g['seed'][i])}", pose=pose, motion=motion_from_poses(pose),
                         geo=g["wheel_geo"][o : o + T + 1], on=g["game_on"][a : a + T],
                         act=g["actions"][a : a + T], tiles=g["tiles"][ts : ts + nt].astype(np.float64)))
    return runs


def windows(runs, rng):
    """head_horizon's two kinds of start; returns {kind: [(run, t0, W)]}."""
    out = {"random start": [], "30 steps before leaving the road": []}
    for r in runs:
        T = len(r["act"])
        if T > WARM + H + 2:
            for _ in range(3):
                out["random start"].append((r, int(rng.integers(WARM, max(WARM + 1, T - H - 1)))))
        o = first_off_run(r["on"])
        if o is not None:
            out["30 steps before leaving the road"].append((r, o + 1 - LEAD))
    return {k: [(r, t0, min(H, len(r["act"]) - 1 - t0)) for r, t0 in v if t0 >= 1 and min(H, len(r["act"]) - 1 - t0) >= 20]
            for k, v in out.items()}


# ---------------------------------------------------------------- motion guesser

class MotionGRU(nn.Module):
    """[action, previous motion (scaled), wheels on the road] -> this step's motion (scaled)."""

    def __init__(self, hidden=128):
        super().__init__()
        self.cell = nn.GRUCell(3 + 3 + 4, hidden)
        self.out = nn.Linear(hidden, 3)
        self.register_buffer("scale", torch.ones(3))

    def forward(self, a, m_prev, geo, h):
        h = self.cell(torch.cat([a, m_prev / self.scale, geo], -1), h)
        return self.out(h) * self.scale, h


def train_gru(runs, steps, L=120, teach=8, batch=128, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    runs = [r for r in runs if len(r["act"]) > L + 1]
    M = np.concatenate([r["motion"] for r in runs])
    model = MotionGRU()
    model.scale.copy_(torch.tensor(M.std(0), dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    progress("train motion guesser", 0, steps)
    for s in range(steps):
        idx = rng.integers(len(runs), size=batch)
        t0 = [int(rng.integers(1, len(runs[i]["act"]) - L)) for i in idx]
        g = lambda k, off=0: torch.tensor(np.stack([runs[i][k][t + off : t + off + L] for i, t in zip(idx, t0)]),
                                          dtype=torch.float32)
        a, m, mp, geo = g("act"), g("motion"), g("motion", -1), g("geo").float()
        h = torch.zeros(batch, model.cell.hidden_size)
        prev, preds = mp[:, 0], []
        for t in range(L):
            pred, h = model(a[:, t], prev, geo[:, t], h)
            preds.append(pred)
            prev = m[:, t] if t + 1 < teach else pred          # own guesses after the first steps
        loss = (((torch.stack(preds, 1) - m) / model.scale) ** 2)[:, teach:].mean()
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        progress("train motion guesser", s + 1, steps)
    return model.eval()


# ---------------------------------------------------------------- map readout

def map_wheels(pose, win, use_view):
    """Road status of the four wheels at car poses (N,3) on each window's map: (N,4) bool; and
    whether all wheel points lie on the seen part of the map (N,) bool."""
    pts = wheel_points(pose).reshape(len(pose), 20, 2)
    d = ((win["cent"] - pose[:, None, :2]) ** 2).sum(-1)
    d[~win["valid"]] = np.inf
    near = np.argpartition(d, NEAR, axis=1)[:, :NEAR]
    tiles = np.take_along_axis(win["tiles"], near[..., None, None], 1)
    road = points_in_tiles(pts, tiles, np.take_along_axis(win["valid"], near, 1))
    seen = in_view(win["hist_pose"], win["hist_rect"], pts)
    if use_view:
        road |= ~seen                                                  # unknown reads as road
    return road.reshape(-1, 4, 5).any(-1), seen.all(1)


def pack(wins):
    """Arrays for a chunk of windows: history padded at the front to WARM steps."""
    N, M = len(wins), max(len(r["tiles"]) for r, _, _ in wins)
    w = dict(tiles=np.zeros((N, M, 4, 2)), valid=np.zeros((N, M), bool), hist_pose=np.zeros((N, WARM + 1, 3)),
             hist_rect=np.zeros((N, WARM + 1, 4)), hist_mask=np.zeros((N, WARM), bool),
             h_act=np.zeros((N, WARM, 3)), h_mprev=np.zeros((N, WARM, 3)), h_geo=np.zeros((N, WARM, 4)),
             pose=np.zeros((N, H + 1, 3)), act=np.zeros((N, H, 3)), motion=np.zeros((N, H, 3)),
             truth=np.ones((N, H), bool), mask=np.zeros((N, H), bool), last=np.zeros((N, 3)))
    for n, (r, t0, W) in enumerate(wins):
        k = min(WARM, t0)
        nt = len(r["tiles"])
        w["tiles"][n, :nt], w["valid"][n, :nt] = r["tiles"], True
        frames = np.arange(t0 - k, t0 + 1)
        frames = np.r_[np.full(WARM - k, frames[0]), frames]           # pad: repeat the first frame
        w["hist_pose"][n], w["hist_rect"][n] = r["pose"][frames], visible_rect(frames)
        s = np.arange(t0 - k, t0)
        w["hist_mask"][n, WARM - k :] = True
        w["h_act"][n, WARM - k :], w["h_geo"][n, WARM - k :] = r["act"][s], r["geo"][s]
        w["h_mprev"][n, WARM - k :] = np.where((s > 0)[:, None], r["motion"][np.maximum(s - 1, 0)], 0)
        w["pose"][n, : W + 1], w["act"][n, :W], w["motion"][n, :W] = r["pose"][t0 : t0 + W + 1], r["act"][t0 : t0 + W], r["motion"][t0 : t0 + W]
        w["truth"][n, :W], w["mask"][n, :W] = r["on"][t0 : t0 + W], True
        w["last"][n] = r["motion"][t0 - 1]
    w["cent"] = w["tiles"].mean(2)
    return w


def step_pose(p, m):
    r, f = axes(p[:, 2])
    return np.concatenate([p[:, :2] + m[:, :1] * r + m[:, 1:2] * f, p[:, 2:] + m[:, 2:]], 1)


@torch.no_grad()
def run_chunk(win, motion, map_kind, gru):
    """Predicted road status (H,N), poses (H,N,3) and coverage (H,N) for one motion source and map."""
    use_view = map_kind == "seen"
    N = len(win["truth"])
    p = win["pose"][:, 0].copy()
    if motion == "guessed":
        T = lambda x: torch.tensor(x, dtype=torch.float32)
        h = torch.zeros(N, gru.cell.hidden_size)
        for i in range(WARM):                                  # the real history, teacher-forced
            _, h_new = gru(T(win["h_act"][:, i]), T(win["h_mprev"][:, i]), T(win["h_geo"][:, i]), h)
            h = torch.where(T(win["hist_mask"][:, i])[:, None] > 0, h_new, h)
        prev = win["last"].copy()
    on, poses, cov = [], [], []
    for j in range(H):
        wh, seen = map_wheels(p, win, use_view)
        on.append(wh.any(1))
        poses.append(p.copy())
        cov.append(seen)
        if motion == "true":
            m = win["motion"][:, j]
        elif motion == "held":
            m = win["last"]
        else:
            mt, h = gru(T(win["act"][:, j]), T(prev), T(wh.astype(np.float32)), h)
            m = prev = mt.numpy().astype(np.float64)
        p = step_pose(p, m)
    return np.stack(on), np.stack(poses), np.stack(cov)


def counts(on, poses, win):
    """Per window and bin: counts for accuracy, off-road recall, coverage and pose error."""
    truth, mask = win["truth"].T, win["mask"].T                 # (H,N)
    off = mask & ~truth
    err = np.hypot(*(poses[..., :2] - win["pose"][:, :H, :2].transpose(1, 0, 2)).transpose(2, 0, 1))
    dh = np.abs((poses[..., 2] - win["pose"][:, :H, 2].T + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
    out = {}
    for bi, (a, b) in enumerate(BINS):
        sl = slice(a, b)
        out[bi] = {"n": mask[sl].sum(0), "correct": (mask & (on == truth))[sl].sum(0),
                   "n_off": off[sl].sum(0), "off_correct": (off & ~on)[sl].sum(0),
                   "pos_err": np.where(mask, err, 0)[sl].sum(0), "hdg_err": np.where(mask, dh, 0)[sl].sum(0)}
    return out


def summarise(c, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    boot = rng.integers(len(c["n"]), size=(reps, len(c["n"])))
    def ratio(num, den):
        if c[den].sum() == 0:
            return None
        s = c[num][boot].sum(1) / np.maximum(c[den][boot].sum(1), 1)
        return [float(c[num].sum() / c[den].sum()), *np.percentile(s, [2.5, 97.5]).tolist()]
    return {"windows": int((c["n"] > 0).sum()), "steps": int(c["n"].sum()), "off_share": float(c["n_off"].sum() / max(c["n"].sum(), 1)),
            "acc": ratio("correct", "n"), "off_recall": ratio("off_correct", "n_off"),
            "pos_err": ratio("pos_err", "n"), "hdg_err_deg": ratio("hdg_err", "n"),
            **({"coverage": ratio("covered", "n")} if "covered" in c else {})}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gru-steps", type=int, default=6000)
    p.add_argument("--limit", type=int, default=None, help="windows per set (smoke test)")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--gru", type=Path, default=GRU_PATH)
    p.add_argument("--geom", type=Path, default=GEOM, help="where the stage-1 geometry files are")
    a = p.parse_args()
    t_start = time.time()
    is_val = np.load(DATA / "latents.npz")["is_val"][: len(np.load(a.geom / "geom_data.npz")["seed"])]
    data = load_runs(a.geom / "geom_data.npz")
    dev = [r for f in sorted(a.geom.glob("geom_dev_*.npz")) for r in load_runs(f)]
    assert dev, "no development runs: run stage 1 first"
    sets = {"development runs (validation tracks, six dream-trained drivers)": dev,
            "original data, held-out validation episodes": [r for r, v in zip(data, is_val) if v]}
    rng = np.random.default_rng(0)
    wins = {f"{s} | {k}": w[: a.limit] for s, runs in sets.items() for k, w in windows(runs, rng).items()}
    n_chunks = sum(-(-len(w) // CHUNK) for w in wins.values())
    if not a.gru.exists():
        plan({"train motion guesser": a.gru_steps * 0.085, "evaluate": n_chunks * len(MOTIONS) * len(MAPS) * 1.2})
        gru = train_gru([r for r, v in zip(data, is_val) if not v], a.gru_steps)
        a.gru.parent.mkdir(parents=True, exist_ok=True)
        torch.save(gru.state_dict(), a.gru)
    else:
        plan({"evaluate": n_chunks * len(MOTIONS) * len(MAPS) * 1.2})
        progress("train motion guesser", 1, 1)
        gru = MotionGRU()
        gru.load_state_dict(torch.load(a.gru, weights_only=True))
        gru.eval()
    res, arrays, done = {}, {}, 0
    total = n_chunks * len(MOTIONS) * len(MAPS)
    progress("evaluate", 0, total)
    for key, w in wins.items():
        acc = {}
        for c0 in range(0, len(w), CHUNK):
            win = pack(w[c0 : c0 + CHUNK])
            for mo in MOTIONS:
                for mp in MAPS:
                    on, poses, cov = run_chunk(win, mo, mp, gru)
                    cnt = counts(on, poses, win)
                    if mp == "seen":
                        for bi, (lo, hi) in enumerate(BINS):
                            cnt[bi]["covered"] = np.where(win["mask"].T, cov, 0)[lo:hi].sum(0)
                    for bi, cb in cnt.items():
                        for k2, v in cb.items():
                            acc.setdefault((mo, mp, bi, k2), []).append(v)
                    done += 1
                    progress("evaluate", done, total)
        res[key] = {"windows": len(w), "readouts": {}}
        for mo in MOTIONS:
            for mp in MAPS:
                byb = {}
                for bi, (lo, hi) in enumerate(BINS):
                    c = {k2: np.concatenate(acc[(mo, mp, bi, k2)]) for k2 in
                         ("n", "correct", "n_off", "off_correct", "pos_err", "hdg_err") + (("covered",) if mp == "seen" else ())}
                    for k2, v in c.items():
                        arrays[f"{key}||{mo}||{mp}||{bi}||{k2}"] = v
                    byb[f"{lo}-{hi}"] = summarise(c)
                res[key]["readouts"][f"{mo} motion, {mp} map"] = byb
    dev_on = "development runs (validation tracks, six dream-trained drivers) | 30 steps before leaving the road"
    dev_rand = "development runs (validation tracks, six dream-trained drivers) | random start"
    prim = res[dev_on]["readouts"]["guessed motion, seen map"]["20-40"]["off_recall"]
    racc = res[dev_rand]["readouts"]["guessed motion, seen map"]["20-40"]["acc"]
    crit = {"1 off-road steps recognised at steps 20-40 (departure windows) >= 0.80": prim[0] >= 0.80,
            "2 accuracy at steps 20-40 (random starts) >= 0.90": racc[0] >= 0.90}
    out = {"results": res, "criteria": crit, "passed": all(crit.values()), "gru": str(a.gru),
           "minutes": (time.time() - t_start) / 60, "smoke_limit": a.limit}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "stage2.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(a.out / "stage2_counts.npz", **arrays)
    for key, r in res.items():
        print(f"\n{key} ({r['windows']} windows): accuracy / off-road recall [/ coverage], by steps since the start")
        for name, byb in r["readouts"].items():
            f = lambda v: "-" if v is None else f"{v[0]:.2f}"
            print(f"  {name:28s} " + "  ".join(f"{b}: {f(s['acc'])}/{f(s['off_recall'])}"
                                              + (f"/{f(s['coverage'])}" if "coverage" in s else "") for b, s in byb.items()))
    tm = res[dev_on]["readouts"]["true motion, seen map"]["20-40"]["off_recall"]
    result(f"departure windows, steps 20-40, off-road steps recognised: true motion {tm[0]:.2f}, "
           f"guessed motion {prim[0]:.2f} [{prim[1]:.2f}, {prim[2]:.2f}]")
    result(f"random starts, steps 20-40, accuracy with guessed motion: {racc[0]:.2f} [{racc[1]:.2f}, {racc[2]:.2f}]")
    for k, v in crit.items():
        result(f"{'PASS' if v else 'FAIL'}  {k}")
    result(f"STAGE 2 {'PASSED' if out['passed'] else 'FAILED'}  ({a.out / 'stage2.json'})")


if __name__ == "__main__":
    main()
