"""The gate of reports/pose_state/preregistration.md, computed from pose_horizon.py's per-window counts.

M = share of real off-road steps recognised at imagined steps 20-40 (latent probe, model fed its own
predictions), on the 275 test-track windows started 30 steps before the car leaves the road.
Pass only if, at temperature 1.15: (1) M >= 0.80 for every pose seed; (2a) mean pose M minus mean
control M >= 0.10; (2b) lowest pose seed above highest control seed; (2c) for each seed, the 95%
bootstrap interval (windows resampled jointly) of pose M minus control M excludes zero.

    python diagnostics/pose_gate.py --pose pose_s0 pose_s1 --control ctrl_s0 ctrl_s1
Writes reports/pose_state/gate.json."""

import argparse
import json

import numpy as np

from pose_horizon import OUT

SET = "test-track rollouts of dream-trained drivers | 30 steps before leaving the road"
BIN = 3          # 20-40
REPS = 2000


def counts(name, tau, readout="own"):
    z = np.load(OUT / f"{name}_tau{tau}.npz")
    k = f"{SET}||{readout}||{BIN}"
    return z[f"{k}||off_correct"].astype(float), z[f"{k}||n_off"].astype(float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose", nargs="+", required=True)
    p.add_argument("--control", nargs="+", required=True)
    p.add_argument("--tau", default="1.15")
    a = p.parse_args()
    rng = np.random.default_rng(0)
    c = {n: counts(n, a.tau) for n in a.pose + a.control}
    n_w = len(next(iter(c.values()))[0])
    boot = rng.integers(n_w, size=(REPS, n_w))
    m = {n: float(x.sum() / d.sum()) for n, (x, d) in c.items()}
    ci = {n: np.percentile(x[boot].sum(1) / d[boot].sum(1), [2.5, 97.5]).tolist() for n, (x, d) in c.items()}
    pairs = {}
    for pn, cn in zip(a.pose, a.control):
        (xp, dp), (xc, dc) = c[pn], c[cn]
        diff = xp[boot].sum(1) / dp[boot].sum(1) - xc[boot].sum(1) / dc[boot].sum(1)
        pairs[f"{pn} - {cn}"] = {"diff": m[pn] - m[cn], "ci95": np.percentile(diff, [2.5, 97.5]).tolist()}
    mp, mc = np.mean([m[n] for n in a.pose]), np.mean([m[n] for n in a.control])
    crit = {
        "1 every pose seed M >= 0.80": all(m[n] >= 0.80 for n in a.pose),
        "2a mean pose - mean control >= 0.10": bool(mp - mc >= 0.10),
        "2b lowest pose above highest control": min(m[n] for n in a.pose) > max(m[n] for n in a.control),
        "2c every paired 95% interval excludes zero": all(v["ci95"][0] > 0 for v in pairs.values()),
    }
    out = {"tau": a.tau, "windows": n_w, "M": m, "M_ci95": ci, "mean_pose": mp, "mean_control": mc,
           "paired": pairs, "criteria": crit, "gate_passed": all(crit.values())}
    (OUT.parent / f"gate{'' if a.tau == '1.15' else '_tau' + a.tau}.json").write_text(json.dumps(out, indent=1))
    for n in m:
        print(f"{n:10s} M = {m[n]:.3f} [{ci[n][0]:.3f}, {ci[n][1]:.3f}]")
    for k, v in pairs.items():
        print(f"{k}: {v['diff']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
    for k, v in crit.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print("GATE", "PASSED" if out["gate_passed"] else "FAILED")


if __name__ == "__main__":
    main()
