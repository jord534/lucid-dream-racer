"""Build the figures for the technical note from whatever runs and evaluations exist.

    python diagnostics/make_figures.py
Writes fig1..fig4 into reports/. Skips any figure whose inputs are missing.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path("reports")
DREAM_RUNS = [("runs/dream_controller/log.csv", "before the fixes"),
              ("runs/dream_v3/log.csv", "after the fixes")]
REAL_RUNS = [("runs/controller/log.csv", "against the first predictor"),
             ("runs/controller_v2/log.csv", "against the repaired predictor")]


def fig1_transfer():
    have = [(p, lab) for p, lab in DREAM_RUNS if Path(p).exists()]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(5.2 * len(have), 3.6), sharey=True)
    for ax, (p, lab) in zip(np.atleast_1d(axes), have):
        d = pd.read_csv(p)
        ax.plot(d.gen, d.dream_fit_mean, lw=1, alpha=.5, label="population, in the dream")
        if "mean_dream" in d:
            m = d.dropna(subset=["mean_dream"])
            ax.plot(m.gen, m.mean_dream, "s--", ms=4, label="best driver, in the dream")
        r = d.dropna(subset=["real_return"])
        ax.plot(r.gen, r.real_return, "o-", ms=4, color="tab:green",
                label="best driver, real track")
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("generation")
        ax.legend(fontsize=7)
    np.atleast_1d(axes)[0].set_ylabel("return")
    fig.tight_layout(); fig.savefig(R / "fig1_transfer.png", dpi=160); plt.close(fig)


def fig2_learning():
    have = [(p, lab) for p, lab in REAL_RUNS if Path(p).exists()]
    if not have:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for p, lab in have:
        d = pd.read_csv(p).dropna(subset=["val_return"])
        ax.plot(d.gen, d.val_return, "o-", ms=3, label=lab)
    ax.axhline(900, ls="--", lw=1, c="grey")
    ax.set_xlabel("CMA-ES generation")
    ax.set_ylabel("validation return (16 tracks)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(R / "fig2_learning.png", dpi=160); plt.close(fig)


def _evals():
    out = {}
    for f in sorted(R.glob("eval_*.json")):
        d = json.loads(f.read_text())
        out[d["name"]] = np.asarray(d["returns"], float)
    return out


def _iqm(x):
    lo, hi = np.percentile(x, [25, 75])
    return float(x[(x >= lo) & (x <= hi)].mean())


def _ci(x, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    b = [_iqm(rng.choice(x, len(x), replace=True)) for _ in range(reps)]
    return np.percentile(b, [2.5, 97.5])


def fig3_headline():
    ev = _evals()
    pick = [(k, v) for k, v in ev.items() if "noise" not in k and "grip" not in k
            and "gas" not in k]
    if not pick:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.7 * len(pick) + 1.6))
    for i, (name, x) in enumerate(pick):
        lo, hi = _ci(x)
        ax.barh(i, _iqm(x), color="tab:blue", alpha=.75)
        ax.plot([lo, hi], [i, i], color="black", lw=2)
        ax.text(_iqm(x) + 12, i, f"{_iqm(x):.0f}", va="center", fontsize=9)
    ax.axvline(900, ls="--", lw=1, c="grey")
    ax.set_yticks(range(len(pick)))
    ax.set_yticklabels([k for k, _ in pick], fontsize=9)
    ax.set_xlabel("IQM over 100 unseen tracks, with 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(R / "fig3_headline.png", dpi=160); plt.close(fig)


# Severity order within each perturbation family, rather than the alphabetical order
# glob() returns.
SEVERITY_ORDER = ["gas50", "noise10", "noise25", "noise50", "grip0.8", "grip0.6"]
RELABEL = {"gas50": "throttle 50%", "grip0.8": "grip x0.8", "grip0.6": "grip x0.6"}


def fig4_robustness():
    ev = _evals()
    base = next((v for k, v in ev.items() if k.endswith("_v2")), None)
    pert = {k[len("wm_real_v2_"):]: v for k, v in ev.items()
            if any(t in k for t in ("noise", "grip", "gas"))}
    if base is None or not pert:
        return
    ordered = [s for s in SEVERITY_ORDER if s in pert]
    b = _iqm(base)
    names = ["none"] + ordered
    series = [base] + [pert[s] for s in ordered]
    vals = [_iqm(x) for x in series]
    cis = [_ci(x) for x in series]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(range(len(vals)), vals,
           color=["tab:grey"] + ["tab:red" if v < 0.5 * b else "tab:blue" for v in vals[1:]])
    for i, (v, (lo, hi)) in enumerate(zip(vals, cis)):
        ax.plot([i, i], [lo, hi], color="black", lw=1.5)
        ax.text(i, hi + 10, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([RELABEL.get(n, n) for n in names], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("IQM, with 95% bootstrap CI")
    fig.tight_layout(); fig.savefig(R / "fig4_robustness.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    R.mkdir(exist_ok=True)
    for f in (fig1_transfer, fig2_learning, fig3_headline, fig4_robustness):
        f()
        print(f.__name__, "done")
