"""Does a controller's per-track test return depend on how bendy the track is?
Spearman correlation of return against three track measures, for each stored evaluation.
The three evaluations share the same 100 test seeds, so the track measures are computed once.

    python diagnostics/bendiness_check.py
Writes reports/dream_failure_analysis/bendiness_corr.json."""
import json

import numpy as np
from scipy.stats import spearmanr

from ldr.config import REPORTS
from ldr.envs import make_env

OUT = REPORTS / "dream_failure_analysis"
EVALS = ["wm_dream_v3", "wm_dream_v1", "wm_real_v2"]


def track_measures(seed):
    env = make_env()
    env.reset(seed=seed)
    beta = np.array([t[1] for t in env.unwrapped.track])
    env.close()
    d = np.angle(np.exp(1j * np.diff(np.r_[beta, beta[0]])))      # wrapped heading change per tile
    return dict(total_turning=float(np.abs(d).sum()), sharpest_tile_turn=float(np.abs(d).max()),
                length_tiles=len(beta))


if __name__ == "__main__":
    ev = {n: json.loads((REPORTS / f"eval_{n}.json").read_text()) for n in EVALS}
    seeds = ev[EVALS[0]]["seeds"]
    assert all(ev[n]["seeds"] == seeds for n in EVALS), "evaluations use different tracks"
    tm = [track_measures(s) for s in seeds]
    out = {}
    for n in EVALS:
        r = np.asarray(ev[n]["returns"], float)
        out[n] = dict(n=len(r), min=float(r.min()), q25=float(np.percentile(r, 25)),
                      median=float(np.median(r)), q75=float(np.percentile(r, 75)),
                      max=float(r.max()))
        for k in tm[0]:
            rho, p = spearmanr([m[k] for m in tm], r)
            out[n][f"rho_vs_{k}"], out[n][f"p_vs_{k}"] = float(rho), float(p)
        print(n, json.dumps({k: round(v, 3) for k, v in out[n].items()}))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bendiness_corr.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT / 'bendiness_corr.json'}")
