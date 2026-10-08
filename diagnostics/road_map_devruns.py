"""Development runs for the road-map experiment (reports/road_map/preregistration.md).

Drives one dream-trained driver on validation tracks exactly as diagnostics/dream_fidelity.py
--collect drives it on the test tracks, and logs the same per-step record, so that development can
use the same kind of driving as the test-track runs without touching them. The driver is chosen
with dream_fidelity's variables:

    LDR_DIAG_CTRL=runs/loop_head/r02/proposer0/best.pt LDR_DIAG_MODEL=runs/loop_head/r02 \
        python diagnostics/road_map_devruns.py --tag loop_head_r02_p0
Writes data/road_map/devruns_<tag>.npz (skipped if it exists). Prints PROGRESS lines."""

import argparse
import time
from multiprocessing import Pool

import numpy as np
import torch

import dream_fidelity as df
from ldr.config import DATA, SEED_VAL
from ldr.progress import progress

FIRST_SEED = SEED_VAL + 300      # 900,300 onward: not used anywhere else


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--tracks", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    out = DATA / "road_map" / f"devruns_{a.tag}.npz"
    task = f"devruns {a.tag}"
    if out.exists():
        progress(task, a.tracks, a.tracks)
        return
    theta = torch.load(df.CTRL, weights_only=False)["theta"]
    jobs = [(theta, FIRST_SEED + i) for i in range(a.tracks)]
    res, t0 = [], time.time()
    progress(task, 0, len(jobs))
    with Pool(a.workers, initializer=df._init) as pool:
        for r in pool.imap_unordered(df._collect_one, jobs, chunksize=1):
            res.append(r)
            progress(task, len(res), len(jobs))
    res.sort(key=lambda r: r["seed"])
    length = np.array([len(r["act"]) for r in res])
    cat = lambda k: np.concatenate([r[k] for r in res])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, seed=np.array([r["seed"] for r in res]), length=length,
                        start=np.r_[0, np.cumsum(length)[:-1]], ctrl=str(df.CTRL), model=str(df.RNN_PATH),
                        **{k: cat(k) for k in ("mu", "logvar", "act", "rew", "onroad", "lat", "hdg")})
    tmp.rename(out)
    rets = [float(r["rew"].sum()) for r in res]
    print(f"{a.tag}: {len(res)} tracks in {(time.time() - t0) / 60:.1f} min, mean return {np.mean(rets):.1f}, "
          f"runs leaving the road: {sum(df._first_run(~r['onroad']) is not None for r in res)}", flush=True)


if __name__ == "__main__":
    main()
