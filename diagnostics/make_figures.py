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


# ---------------------------------------------------------------------------------------------
# Figures added in issue 2 of the note (sections 3.5 to 3.9). Every number is read from a file
# under reports/; a figure whose inputs are missing is skipped.
FA = R / "dream_failure_analysis"
LOOP1, LOOP2, LOOP3 = R / "loop_run_2026-10-01", R / "loop_run_2026-10-02_road", R / "loop_run_2026-10-02_head"
GROUPS = [("first loop", "loop", "tab:blue"), ("road-ranked loop", "loop_road", "tab:green"),
          ("with on-road output", "loop_head", "tab:red")]


def _j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fig5_exploitation():
    cl, ac, cov = _j(FA / "dream_closed_loop.json"), _j(FA / "dream_actions_in_sim.json"), _j(FA / "data_coverage.json")
    if not (cl and ac and cov):
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    m = cl["summary"]["mean_p_off"]
    v = [m["real"], m["open"], m["closed"]]
    ax[0].bar(range(3), v, color=["tab:grey", "tab:blue", "tab:red"])
    for i, x in enumerate(v):
        ax[0].text(i, x + 0.02, f"{x:.2f}", ha="center", fontsize=8)
    ax[0].set_xticks(range(3), ["real frames", "model, recorded\nactions", "model, controller\nsteering"], fontsize=8)
    ax[0].set_ylabel("mean P(on road)", fontsize=8)
    ax[0].set_title("(a) While the real car is off the road", fontsize=9)
    ax[0].set_ylim(0, 1.05)
    w, l20 = ac["summary"]["window_share_on_road"], ac["summary"]["last20_share_on_road"]
    x = np.arange(2)
    ax[1].bar(x - 0.18, [w["dream_mean_p_on_road"], l20["dream_mean_p_on_road"]], 0.36, color="tab:red",
              label="model: mean P(on road)")
    ax[1].bar(x + 0.18, [w["real_under_dream_actions"], l20["real_under_dream_actions"]], 0.36,
              color="tab:grey", label="simulator: share on road")
    for xi, a_, b_ in zip(x, [w["dream_mean_p_on_road"], l20["dream_mean_p_on_road"]],
                          [w["real_under_dream_actions"], l20["real_under_dream_actions"]]):
        ax[1].text(xi - 0.18, a_ + 0.02, f"{a_:.2f}", ha="center", fontsize=8)
        ax[1].text(xi + 0.18, b_ + 0.02, f"{b_:.2f}", ha="center", fontsize=8)
    ax[1].set_xticks(x, ["whole window\n(up to 100 steps)", "last 20 steps"], fontsize=8)
    ax[1].set_ylim(0, 1.25)
    ax[1].legend(fontsize=7, loc="upper center", ncol=2)
    ax[1].set_title("(b) The controller's own actions, in both", fontsize=9)
    pp = cov["per_policy"]
    names = [("pursuit", "pursuit\ndriver"), ("brownian", "random\ndriver"), ("controller", "on-policy\ncontroller")]
    vals = [pp[k]["off_road_share"]["value"] for k, _ in names]
    cis = [pp[k]["off_road_share"]["ci95"] for k, _ in names]
    dt = cov["dream_trained_controller_test_rollouts"]["off_road_share"]
    vals.append(dt["value"]); cis.append(dt["ci95"])
    labels = [n for _, n in names] + ["model-trained\ncontroller, test"]
    ax[2].bar(range(4), vals, color=["tab:blue"] * 3 + ["tab:red"])
    for i, (vv, (lo, hi)) in enumerate(zip(vals, cis)):
        ax[2].plot([i, i], [lo, hi], color="black", lw=1.2)
        ax[2].text(i, hi + 0.02, f"{vv:.2f}", ha="center", fontsize=8)
    ax[2].set_xticks(range(4), labels, fontsize=8)
    ax[2].set_ylabel("share of frames off the road", fontsize=8)
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("(c) Training data (blue) and model-trained driving", fontsize=9)
    fig.tight_layout(); fig.savefig(R / "fig5_exploitation.png", dpi=160); plt.close(fig)


def fig6_loop():
    meas = [_j(LOOP1 / f"measure_round{r}.json") for r in range(3)]
    if not all(meas):
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    for r, e in enumerate(meas):
        ax[0].scatter([r] * 3, e["real_return_by_proposer"], color="tab:blue", s=22, zorder=3)
    ax[0].plot(range(3), [e["real_return_mean"] for e in meas], "o-", color="tab:blue", label="mean of 3 drivers")
    ax[0].set_xticks(range(3), ["0\n(original model)", "1", "2"], fontsize=8)
    ax[0].set_xlabel("round", fontsize=8)
    ax[0].set_ylabel("real return, 50 validation tracks", fontsize=8)
    ax[0].set_title("(a) Real return of drivers trained in the model", fontsize=9)
    ax[0].legend(fontsize=7)
    ax[1].plot(range(3), [e["dream_return_window_mean"] for e in meas], "s-", color="tab:red",
               label="model's predicted return")
    ax[1].plot(range(3), [e["real_return_window_mean"] for e in meas], "o-", color="tab:grey",
               label="real return, same actions")
    for r, e in enumerate(meas):
        ax[1].text(r, e["dream_return_window_mean"] + 3, f"gap {e['gap_mean']:.0f}", ha="center", fontsize=8)
    ax[1].set_xticks(range(3), ["0", "1", "2"], fontsize=8)
    ax[1].set_xlabel("round", fontsize=8)
    ax[1].set_ylabel("return over the window", fontsize=8)
    ax[1].set_ylim(0, max(e["dream_return_window_mean"] for e in meas) * 1.25)
    ax[1].set_title("(b) Exploit gap: the model's own actions", fontsize=9)
    ax[1].legend(fontsize=7)
    ev = _evals()
    rows = [("simulator-\ntrained", ["wm_real_v2"], "tab:grey"), ("model-trained,\noriginal", ["wm_dream_v3"], "tab:grey")]
    rows += [(lab.replace(" ", "\n", 1), [f"{t}_r02_p{i}" for i in range(3)], c) for lab, t, c in GROUPS[:2]]
    for i, (lab, names, col) in enumerate(rows):
        for k, n in enumerate(names):
            if n not in ev:
                continue
            x = i + (k - (len(names) - 1) / 2) * 0.22
            v, (lo, hi) = _iqm(ev[n]), _ci(ev[n])
            ax[2].plot([x, x], [lo, hi], color=col, lw=1.5)
            ax[2].scatter([x], [v], color=col, s=24, zorder=3)
    ax[2].set_xticks(range(len(rows)), [r[0] for r in rows], fontsize=8)
    ax[2].set_ylabel("IQM on 100 test tracks, 95% CI", fontsize=8)
    ax[2].set_title("(c) Test tracks: one point per driver", fontsize=9)
    fig.tight_layout(); fig.savefig(R / "fig6_loop.png", dpi=160); plt.close(fig)


def fig7_road_status():
    groups = [("original", [FA], "tab:grey")] + [
        (lab, [FA / f"{t}_r02_p{i}" for i in range(3)], c) for lab, t, c in GROUPS]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    have = False
    for k, (fname, key, title) in enumerate([
            ("dream_closed_loop.json", "primary_closed_minus_real", "(a) Controller steering in the model"),
            ("dream_actions_in_sim.json", "primary_dream_minus_real_latent", "(b) The same actions in the simulator")]):
        for i, (lab, dirs, col) in enumerate(groups):
            for j, d in enumerate(dirs):
                s = _j(d / fname)
                if not s:
                    continue
                have = True
                p = s["summary"][key]
                x = i + (j - (len(dirs) - 1) / 2) * 0.2
                ax[k].plot([x, x], p["ci95"], color=col, lw=1.5)
                ax[k].scatter([x], [p["median"]], color=col, s=24, zorder=3)
        ax[k].axhline(0, color="black", lw=0.6)
        ax[k].set_xticks(range(len(groups)), [g[0].replace(" ", "\n", 1) for g in groups], fontsize=8)
        ax[k].set_title(title, fontsize=9)
    ax[0].set_ylabel("median over tracks of model P(on road)\nminus real, at real off-road steps", fontsize=8)
    if have:
        fig.tight_layout(); fig.savefig(R / "fig7_road_status.png", dpi=160)
    plt.close(fig)


def fig8_horizon():
    h = _j(R / "head_horizon.json")
    if not h:
        return
    res = h["results"]
    sets = [("test-track rollouts of dream-trained drivers | random start", "acc",
             "(a) Driver rollouts, random start: accuracy"),
            ("test-track rollouts of dream-trained drivers | 30 steps before leaving the road", "off_recall",
             "(b) Starts 30 steps before leaving the road:\nshare of real off-road steps recognised")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    for k, (name, key, title) in enumerate(sets):
        bins = res[name]["bins"]
        xs = [b["steps"] for b in bins]
        for c, col, ls, lab in [("teacher-forced latent probe", "tab:grey", "-", "fed real frames"),
                                ("free-running latent probe", "tab:red", "-", "fed its own predictions"),
                                ("free-running head", "tab:red", "--", "fed its own predictions (on-road output)")]:
            # recall is only shown where at least 20% of the steps are off the road; before the
            # car leaves (the first ~30 steps of these windows) there are too few off-road steps
            ax[k].plot(xs, [np.nan if b[c][key] is None or (key == "off_recall" and b["off_share"] < 0.2)
                            else b[c][key] for b in bins], "o" + ls,
                       color=col, ms=4, label=lab)
        ax[k].set_xlabel("imagined steps since the real start", fontsize=8)
        ax[k].set_title(title, fontsize=9)
        ax[k].set_ylim(0, 1.02)
    ax[0].set_ylabel("share correct (independent road probe)", fontsize=8)
    ax[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(R / "fig8_horizon.png", dpi=160); plt.close(fig)


def fig9_fixes():
    road = [_j(LOOP2 / f"measure_round{r}.json") for r in range(3)]
    head = [_j(LOOP3 / f"measure_head_round{r}.json") for r in range(3)]
    ctrl = [_j(LOOP3 / f"measure_ctrl_round{r}.json") for r in range(3)]
    hz = {t: _j(R / f"head_horizon{t}.json") for t in ("", "_freerun", "_probe")}
    if not (all(road) and all(head) and all(ctrl) and all(hz.values())):
        return
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.7))
    arms = [("road-ranked", [2, 3, 4], road, "tab:green"), ("+ on-road output, penalty", [4, 5, 6], head, "tab:red"),
            ("control (neither)", [4, 5, 6], ctrl, "tab:orange")]
    for lab, xs, ms, col in arms:
        ax[0].plot(xs, [e["road_gap_mean"] for e in ms], "o-", color=col, label=lab)
        ax[1].plot(xs, [e["real_return_mean"] for e in ms], "o-", color=col, label=lab)
    ax[0].set_ylabel("road gap: model P(on road) - real share", fontsize=8)
    ax[0].set_title("(a) Validation road gap per round", fontsize=9)
    ax[1].set_ylabel("real return, 50 validation tracks", fontsize=8)
    ax[1].set_title("(b) Validation real return per round", fontsize=9)
    for a in ax[:2]:
        a.set_xlabel("round, counted from the original model", fontsize=8)
        a.legend(fontsize=7)
    rr = "test-track rollouts of dream-trained drivers | random start"
    dep = "test-track rollouts of dream-trained drivers | 30 steps before leaving the road"
    pick = lambda t, n, step, key: next(b for b in hz[t]["results"][n]["bins"] if b["steps"] == step)["free-running latent probe"][key]
    labels = ["before", "trained on own\npredictions", "road-status\nloss"]
    a1 = [pick(t, rr, "70-100", "acc") for t in ("", "_freerun", "_probe")]
    a2 = [pick(t, dep, "20-40", "off_recall") for t in ("", "_freerun", "_probe")]
    x = np.arange(3)
    ax[2].bar(x - 0.18, a1, 0.36, color="tab:blue", label="accuracy, steps 70-100 (target 0.90)")
    ax[2].bar(x + 0.18, a2, 0.36, color="tab:purple", label="off-road steps recognised, 20-40 (target 0.85)")
    ax[2].axhline(0.90, color="tab:blue", ls=":", lw=1)
    ax[2].axhline(0.85, color="tab:purple", ls=":", lw=1)
    for xi, u, v in zip(x, a1, a2):
        ax[2].text(xi - 0.18, u + 0.02, f"{u:.2f}", ha="center", fontsize=8)
        ax[2].text(xi + 0.18, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax[2].set_xticks(x, labels, fontsize=8)
    ax[2].set_ylim(0, 1.1)
    ax[2].set_ylabel("independent road probe, model fed its own predictions", fontsize=7)
    ax[2].set_title("(c) Two changes to the training loss", fontsize=9)
    ax[2].legend(fontsize=6.5, loc="upper right")
    fig.tight_layout(); fig.savefig(R / "fig9_fixes.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    R.mkdir(exist_ok=True)
    for f in (fig1_transfer, fig2_learning, fig3_headline, fig4_robustness, fig5_exploitation,
              fig6_loop, fig7_road_status, fig8_horizon, fig9_fixes):
        f()
        print(f.__name__, "done")
