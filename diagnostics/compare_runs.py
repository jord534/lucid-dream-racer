"""One table and one figure across all the runs: the original dream-trained controller, the first
propose-critic loop, the road-ranked loop, and (when finished) the on-road-head run and its control.

    python diagnostics/compare_runs.py        (a few seconds; reads only files already in reports/)

Writes reports/run_comparison.md and reports/run_comparison.png. Groups whose files are missing are
left out. "Round 2 drivers" means the three dream-trained controllers of the last round of a run,
tested on the 100 test tracks and with the two dream-failure tests (the median, over tracks, of the
dream's P(on road) minus the real car's over the steps where the real car is off the road; 0 would be
agreement). The curve panels use each run's own validation measurements, one point per round, with the
rounds of consecutive runs placed end to end (the road-ranked run starts from the first run's round 2).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent.parent / "reports"
FA = R / "dream_failure_analysis"
L1, L2, L3 = (R / "loop_run_2026-10-01", R / "loop_run_2026-10-02_road", R / "loop_run_2026-10-02_head")
GROUPS = [
    dict(name="original dream controller", color="#888888", evals=["wm_dream_v3"], fail=[FA], x0=None),
    dict(name="first loop", color="#1f77b4", evals=[f"loop_r02_p{i}" for i in range(3)],
         fail=[FA / f"loop_r02_p{i}" for i in range(3)],
         measure=[L1 / f"measure_round{r}.json" for r in range(3)], x0=0),
    dict(name="road-ranked loop", color="#2ca02c", evals=[f"loop_road_r02_p{i}" for i in range(3)],
         fail=[FA / f"loop_road_r02_p{i}" for i in range(3)],
         measure=[L2 / f"measure_round{r}.json" for r in range(3)], x0=2),
    dict(name="on-road head + penalty", color="#d62728", evals=[f"loop_head_r02_p{i}" for i in range(3)],
         fail=[FA / f"loop_head_r02_p{i}" for i in range(3)],
         measure=[L3 / f"measure_head_round{r}.json" for r in range(3)], x0=4),
    dict(name="control (no head)", color="#ff7f0e", evals=[f"loop_ctrl_r02_p{i}" for i in range(3)],
         fail=[FA / f"loop_ctrl_r02_p{i}" for i in range(3)],
         measure=[L3 / f"measure_ctrl_round{r}.json" for r in range(3)], x0=4),
]


def load(p):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def tests(g):
    ret = [load(R / f"eval_{n}.json") for n in g["evals"]]
    ret = [r["mean"] for r in ret if r]
    cl, ac = [], []
    for d in g["fail"]:
        c, a = load(d / "dream_closed_loop.json"), load(d / "dream_actions_in_sim.json")
        if c:
            cl.append(c["summary"]["primary_closed_minus_real"]["median"])
        if a:
            ac.append(a["summary"]["primary_dream_minus_real_latent"]["median"])
    return ret, cl, ac


def main():
    rows, data = [], {}
    for g in GROUPS:
        ret, cl, ac = tests(g)
        if not ret:
            continue
        data[g["name"]] = (ret, cl, ac)
        f = lambda x: ", ".join(f"{v:.2f}" for v in x) if x else "-"
        m = lambda x: f"{np.mean(x):.2f}" if x else "-"
        rows.append(f"| {g['name']} | {', '.join(f'{v:.0f}' for v in ret)} | {np.mean(ret):.0f} | "
                    f"{f(cl)} | {m(cl)} | {f(ac)} | {m(ac)} |")
    meas = []
    for g in GROUPS:
        for r, p in enumerate(g.get("measure", [])):
            e = load(p)
            if e:
                meas.append((g["name"], g["x0"] + r, e))
    mrows = [
        f"| {n} | {x} | {e['real_return_mean']:.0f} | {e['gap_mean']:.1f} | "
        + (f"{e['road_gap_mean']:.2f}" if "road_gap_mean" in e else "-") + " | "
        + (f"{e['p_on_when_real_off_mean']:.2f}" if e.get("p_on_when_real_off_mean") is not None else "-")
        + " | " + (f"{e['head_acc_mean']:.2f}" if "head_acc_mean" in e else "-") + " |"
        for n, x, e in meas
    ]
    (R / "run_comparison.md").write_text(f"""# Comparison across runs

Made by `python diagnostics/compare_runs.py` from the files in `reports/`. Test return: the three
last-round drivers of each run on the 100 test tracks (the original controller is a single one).
Closed-loop and in-simulator columns: median over tracks of the dream's P(on road) minus the real car's,
over the steps where the real car is off the road (0 would be agreement; one value per driver, then the
mean). One run per group and three drivers each: differences smaller than the spread between drivers are
not evidence of anything.

| Group | Test return per driver | Mean | Closed-loop road disagreement per driver | Mean | Dream actions in the simulator, per driver | Mean |
|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Validation measurements per round (50 validation tracks, 3 drivers; rounds of consecutive runs are
placed end to end; the road columns exist from the road-ranked run on; head accuracy only for runs with an
on-road head):

| Run | Cumulative round | Real return | Reward gap | Road gap | Dream says on road while real car is off | Head accuracy on branches |
|---|---|---|---|---|---|---|
{chr(10).join(mrows)}
""")
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    names = list(data)
    for k, (title, idx, ylab) in enumerate(
        [("Test return, 100 tracks", 0, "return"), ("Closed-loop road disagreement", 1, "median P(on) dream - real"),
         ("Dream actions in the simulator", 2, "median P(on) dream - real")]):
        for i, n in enumerate(names):
            col = next(g["color"] for g in GROUPS if g["name"] == n)
            y = data[n][idx]
            ax[k].scatter([i] * len(y), y, color=col, s=36, zorder=3)
            if y:
                ax[k].hlines(np.mean(y), i - 0.25, i + 0.25, color=col, lw=2)
        ax[k].set_xticks(range(len(names)), [n.replace(" ", "\n", 1) for n in names], fontsize=7)
        ax[k].set_title(title, fontsize=10)
        ax[k].set_ylabel(ylab, fontsize=8)
        if idx:
            ax[k].axhline(0, color="k", lw=0.6)
    for g in GROUPS:
        pts = [(x, e) for n, x, e in meas if n == g["name"]]
        if pts:
            ax[3].plot([x for x, _ in pts], [e["gap_mean"] for _, e in pts], "o-", color=g["color"],
                       label=f"{g['name']}: reward gap")
            road = [(x, e["road_gap_mean"] * 100) for x, e in pts if "road_gap_mean" in e]
            if road:
                ax[3].plot(*zip(*road), "s--", color=g["color"], alpha=0.6, label=f"{g['name']}: road gap x100")
    ax[3].set_title("Validation, per round", fontsize=10)
    ax[3].set_xlabel("cumulative round", fontsize=8)
    ax[3].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(R / "run_comparison.png", dpi=130)
    print((R / "run_comparison.md").read_text())


if __name__ == "__main__":
    main()
