"""Where does a controller leave the real track, can it recover, and is it a sharp bend?

For each of the 100 test tracks, drive the controller exactly as ldr.evaluate does and log
  - the first sustained excursion: >= 20 consecutive steps with no wheel on the road;
  - whether the car then recovers: back on the road for >= 20 consecutive steps and at
    least 5 further tiles collected;
  - how sharp the track is where the excursion began, as a percentile of that track's own
    bend sharpness (0.5 = no sharper than a random point on the track).

    python diagnostics/bend_check.py
Writes reports/dream_failure_analysis/bend_check.json. One progress line per finished track."""
import json
import time
from multiprocessing import Pool

import numpy as np
import torch
from scipy.stats import binomtest

from ldr.agent import WorldModelAgent
from ldr.config import REPORTS, RUNS, SEED_TEST
from ldr.envs import make_env

CKPTS = {"dream_v3": (RUNS / "dream_v3" / "best.pt", "wm_dream_v3"),
         "controller_v2": (RUNS / "controller_v2" / "best.pt", "wm_real_v2")}
OUT = REPORTS / "dream_failure_analysis"
TRACKS, WORKERS = 100, 6
RUN, REGAIN = 20, 5                         # steps for a sustained run; tiles to count as recovered
LOOKAHEAD = 10                              # tiles, as ldr.envs.curvature_ahead

_AGENT = None


def _init():
    global _AGENT
    _AGENT = WorldModelAgent()


def _sharpness(beta):
    d = np.roll(beta, -LOOKAHEAD) - beta
    return np.abs((d + np.pi) % (2 * np.pi) - np.pi)


def _first_run(flags, start=0):
    """Index where `flags` first holds for RUN consecutive steps at or after `start`."""
    n = 0
    for i in range(start, len(flags)):
        n = n + 1 if flags[i] else 0
        if n == RUN:
            return i - RUN + 1
    return None


def _run(job):
    name, theta, seed = job
    env = make_env()
    obs, _ = env.reset(seed=seed)
    _AGENT.reset()
    rng = np.random.default_rng(seed)
    track = np.asarray(env.unwrapped.track)
    total, onroad, gained, pos, term = 0.0, [], [], [], False
    for t in range(1, 1001):
        obs, r, term, trunc, _ = env.step(_AGENT.act(theta, obs, rng))
        total += r
        gained.append(r > 0)
        onroad.append(any(len(w.tiles) > 0 for w in env.unwrapped.car.wheels))
        pos.append(np.asarray(env.unwrapped.car.hull.position).copy())
        if term or trunc:
            break
    visited = env.unwrapped.tile_visited_count / len(track)
    env.close()

    off = [not o for o in onroad]
    onset = _first_run(off)
    out = dict(name=name, seed=seed, ret=total, steps=t, visited=visited,
               excursion=onset is not None, onset_step=None, onset_bend_pct=None,
               recovered=None)
    if onset is not None:
        sharp = _sharpness(track[:, 1])
        i = int(np.argmin(((track[:, 2:4] - pos[onset]) ** 2).sum(1)))
        out["onset_step"] = onset + 1
        out["onset_bend_pct"] = float(((sharp < sharp[i]).sum() + 0.5 * (sharp == sharp[i]).sum())
                                      / len(sharp))
        back = _first_run(onroad, onset + RUN)
        out["recovered"] = bool(back is not None and sum(gained[back:]) >= REGAIN)
    return out


def _summary(name, rows, ref):
    rows = sorted(rows, key=lambda r: r["seed"])
    ret = np.array([r["ret"] for r in rows])
    exc = [r for r in rows if r["excursion"]]
    out = dict(name=name, n=len(rows), mean_return=float(ret.mean()),
               matches_stored_eval=f"{int(np.sum(np.abs(ret - np.asarray(ref)) < 1e-3))}/{len(rows)}",
               median_tiles_visited=float(np.median([r["visited"] for r in rows])),
               tracks_with_excursion=len(exc),
               excursions_not_recovered=sum(not r["recovered"] for r in exc))
    if exc:
        pct = np.array([r["onset_bend_pct"] for r in exc])
        k = int((pct >= 0.75).sum())
        out.update(onset_bend_pct_mean=float(pct.mean()), onsets_in_top_quartile_of_bends=k,
                   top_quartile_binom_p=float(binomtest(k, len(exc), 0.25).pvalue),
                   median_onset_step=float(np.median([r["onset_step"] for r in exc])))
    return out


if __name__ == "__main__":
    jobs, refs = [], {}
    for name, (path, evalname) in CKPTS.items():
        theta = torch.load(path, weights_only=False)["theta"]
        refs[name] = json.loads((REPORTS / f"eval_{evalname}.json").read_text())["returns"]
        jobs += [(name, theta, SEED_TEST + i) for i in range(TRACKS)]
    done, rows, t0 = 0, {n: [] for n in CKPTS}, time.time()
    print(f"{len(jobs)} rollouts, {WORKERS} workers", flush=True)
    with Pool(WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(_run, jobs, chunksize=1):
            rows[r["name"]].append(r)
            done += 1
            eta = (time.time() - t0) / done * (len(jobs) - done)
            exc = (f"left road at step {r['onset_step']:4d}, bend pct {r['onset_bend_pct']:.2f}, "
                   f"{'recovered' if r['recovered'] else 'NOT recovered'}"
                   if r["excursion"] else "stayed on road")
            print(f"[{done:3d}/{len(jobs)}] {r['name']:13s} seed {r['seed']}  "
                  f"return {r['ret']:7.1f}  tiles {r['visited']:4.0%}  {exc}   "
                  f"eta {eta/60:4.1f} min", flush=True)
    summ = [_summary(n, rows[n], refs[n]) for n in CKPTS]
    print("\n=== summary ===")
    for s in summ:
        print(json.dumps(s, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bend_check.json").write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    print(f"wrote {OUT / 'bend_check.json'}")
