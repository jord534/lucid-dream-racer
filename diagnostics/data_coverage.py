"""How much of the world model's training data shows the car off the road, spun round, or
recovering back onto the road?

The dream shows the car staying on the road where the real car leaves it (dream_closed_loop.py,
dream_actions_in_sim.py). One candidate cause is that the training data rarely shows what happens
once the car leaves the road. This counts it, rather than guessing.

Replay a random sample of the recorded training episodes in the real simulator (the recorded
actions from the track seed reproduce each episode exactly; checked against the recorded rewards)
and read true labels from the simulator:
  off the road  - no wheel touches the road
  spun          - the car points more than 90 degrees away from the track direction
  excursion     - 20 or more consecutive off-road steps
  recovery      - an excursion followed by 20 or more consecutive on-road steps
Reported per recording policy (pursuit driver, random driver, on-policy controller) and as a
frame-weighted estimate for the whole dataset, with bootstrap 95% intervals over episodes, next to
the same numbers for the dream-trained controller's own test rollouts (the states the dream is
exploited in). Descriptive: no hypothesis test, no threshold.

    python diagnostics/data_coverage.py          (about 5 minutes, 6 workers)
Writes reports/dream_failure_analysis/data_coverage.json."""
import itertools
import json
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from dream_fidelity import OUT, RUN, STORE
from ldr.config import DATA, SEED_DATA
from ldr.envs import make_env

ROLLOUTS = DATA / "rollouts"
PER_TYPE, WORKERS = 60, 6


def _replay(ep):
    sh = np.load(ROLLOUTS / f"ep_{ep:05d}.npz")
    env = make_env()
    env.reset(seed=SEED_DATA + ep)
    xy = np.asarray(env.unwrapped.track)[:, 2:4]
    on, hdg, total = [], [], 0.0
    for a in sh["actions"]:
        total += env.step(a)[1]
        car = env.unwrapped.car
        on.append(any(len(w.tiles) > 0 for w in car.wheels))
        pos = np.asarray(car.hull.position)
        i = int(np.argmin(((xy - pos) ** 2).sum(1)))
        d = xy[(i + 1) % len(xy)] - xy[i]
        d = d / (np.linalg.norm(d) + 1e-9)
        f = np.array([-np.sin(car.hull.angle), np.cos(car.hull.angle)])
        hdg.append(np.arctan2(f[0] * d[1] - f[1] * d[0], f @ d))
    env.close()
    return dict(ep=ep, policy=str(sh["policy"]), on=np.array(on, bool), hdg=np.array(hdg, np.float32),
                reward_matches=bool(abs(total - float(sh["rewards"].sum())) < 1e-2))


def _episode(on, hdg):
    runs, i = [], 0
    for k, g in itertools.groupby(on):
        n = len(list(g))
        runs.append((bool(k), i, n))
        i += n
    exc = [j for j, (k, _, n) in enumerate(runs) if not k and n >= RUN]
    rec = [j for j in exc if j + 1 < len(runs) and runs[j + 1][0] and runs[j + 1][2] >= RUN]
    return dict(frames=len(on), off=int((~on).sum()), spun=int((np.abs(hdg) > np.pi / 2).sum()),
                excursions=len(exc), recoveries=len(rec))


def _ratio(eps, num, den="frames", scale=1.0, reps=2000):
    n = np.array([e[num] for e in eps], float)
    d = np.array([e[den] for e in eps], float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(eps), (reps, len(eps)))
    boot = scale * n[idx].sum(1) / d[idx].sum(1)
    return dict(value=float(scale * n.sum() / d.sum()), ci95=[float(x) for x in np.percentile(boot, [2.5, 97.5])])


def _summary(eps):
    return dict(episodes=len(eps), frames=int(sum(e["frames"] for e in eps)),
                off_road_share=_ratio(eps, "off"), spun_share=_ratio(eps, "spun"),
                share_of_episodes_with_an_excursion=float(np.mean([e["excursions"] > 0 for e in eps])),
                excursions_per_1000_frames=_ratio(eps, "excursions", scale=1000),
                recoveries_per_1000_frames=_ratio(eps, "recoveries", scale=1000))


if __name__ == "__main__":
    groups = defaultdict(list)
    for ep in range(1400):
        groups[str(np.load(ROLLOUTS / f"ep_{ep:05d}.npz")["policy"])].append(ep)
    lat = np.load(DATA / "latents.npz")
    ep_len = lat["ep_len"]
    rng = np.random.default_rng(0)
    pick = {p: sorted(rng.choice(v, PER_TYPE, replace=False).tolist()) for p, v in groups.items()}
    jobs = [e for v in pick.values() for e in v]
    print(f"episodes per policy in the dataset: { {p: len(v) for p, v in groups.items()} }", flush=True)
    print(f"replaying {len(jobs)} episodes ({PER_TYPE} per policy), {WORKERS} workers", flush=True)
    res, t0 = [], time.time()
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(_replay, jobs, chunksize=1):
            res.append(r)
            eta = (time.time() - t0) / len(res) * (len(jobs) - len(res))
            print(f"[{len(res):3d}/{len(jobs)}] episode {r['ep']:4d} ({r['policy']:10s}) "
                  f"off-road {1 - r['on'].mean():4.0%}   eta {eta / 60:4.1f} min", flush=True)
    eps = {p: [dict(**_episode(r["on"], r["hdg"])) for r in res if r["policy"] == p] for p in pick}
    out = dict(replays_matching_recorded_rewards=f"{sum(r['reward_matches'] for r in res)}/{len(res)}",
               episodes_per_policy_in_dataset={p: len(v) for p, v in groups.items()},
               frames_per_policy_in_dataset={p: int(sum(ep_len[e] for e in v)) for p, v in groups.items()},
               per_policy={p: _summary(e) for p, e in eps.items()})
    w = {p: out["frames_per_policy_in_dataset"][p] for p in eps}
    tot = sum(w.values())
    out["whole_dataset_estimate"] = {k: float(sum(w[p] / tot * out["per_policy"][p][k]["value"] for p in eps))
                                     for k in ("off_road_share", "spun_share", "recoveries_per_1000_frames")}
    out["whole_dataset_estimate"]["recoveries_in_dataset"] = float(sum(
        out["per_policy"][p]["recoveries_per_1000_frames"]["value"] * w[p] / 1000 for p in eps))
    d = np.load(STORE)
    test = [_episode(d["onroad"][int(s):int(s + n)], d["hdg"][int(s):int(s + n)])
            for s, n in zip(d["start"], d["length"])]
    out["dream_trained_controller_test_rollouts"] = _summary(test)
    print("\n=== summary ===\n" + json.dumps(out, indent=1))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data_coverage.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / 'data_coverage.json'}")
