"""Phase 9: statistics and figures for the write-up.

    python -m ldr.analysis
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baseline_ppo import SKIP
from .config import REPORTS, RUNS


def iqm(x: np.ndarray) -> float:
    """Interquartile mean: mean of the middle 50% of scores."""
    lo, hi = np.percentile(x, [25, 75])
    return float(x[(x >= lo) & (x <= hi)].mean())


def bootstrap_ci(x: np.ndarray, stat, reps: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval, resampling tracks with replacement."""
    rng = np.random.default_rng(seed)
    boots = [stat(rng.choice(x, len(x), replace=True)) for _ in range(reps)]
    return tuple(np.percentile(boots, [2.5, 97.5]))


def interval_table():
    rows = []
    for f in sorted(REPORTS.glob("eval_*.json")):
        d = json.loads(f.read_text())
        x = np.asarray(d["returns"], dtype=float)
        (ilo, ihi), (mlo, mhi) = bootstrap_ci(x, iqm), bootstrap_ci(x, np.mean)
        rows.append(dict(method=d["name"], tracks=len(x), iqm=iqm(x), iqm_lo=ilo, iqm_hi=ihi,
                         mean=x.mean(), mean_lo=mlo, mean_hi=mhi))
    df = pd.DataFrame(rows).round(1)
    df.to_csv(REPORTS / "results.csv", index=False)
    print(df.to_string(index=False))
    return df


def learning_curves():
    plt.figure(figsize=(7, 4))
    wm = RUNS / "controller" / "log.csv"
    if wm.exists():
        d = pd.read_csv(wm).dropna(subset=["val_return"])
        offset = np.load(RUNS.parent / "data" / "packed" / "meta.npz")["ep_len"].sum()
        plt.plot(d.env_steps + offset, d.val_return, label="World model + CMA-ES (real env)")
    for run in sorted(RUNS.glob("ppo_s*")):
        if not (run / "evaluations.npz").exists():      # run stopped before its first eval
            continue
        ev = np.load(run / "evaluations.npz")
        plt.plot(ev["timesteps"] * SKIP, ev["results"].mean(1), alpha=.7, label=f"PPO {run.name}")
    plt.axhline(900, ls="--", c="grey", lw=1)
    plt.xscale("log"); plt.xlabel("real environment steps (incl. data collection, frame skip)")
    plt.ylabel("validation return"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORTS / "sample_efficiency.png", dpi=150)


def transfer_gap():
    f = RUNS / "dream_controller" / "log.csv"
    if not f.exists():
        return
    d = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d.gen, d.dream_fit_mean, label="fitness inside the dream")
    r = d.dropna(subset=["real_return"])
    ax.plot(r.gen, r.real_return, "o-", label="same controller on the real track")
    ax.set_xlabel("CMA-ES generation"); ax.set_ylabel("return"); ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS / "transfer_gap.png", dpi=150)


if __name__ == "__main__":
    REPORTS.mkdir(exist_ok=True)
    interval_table(); learning_curves(); transfer_gap()