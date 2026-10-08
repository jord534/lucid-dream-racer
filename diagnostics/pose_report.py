"""Tables and figures for reports/pose_state/summary.md, from pose_horizon.py's json files.

    python diagnostics/pose_report.py
Writes reports/pose_state/summary.json, tables.md and fig_*.png."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pose_horizon import OUT

ROOT = OUT.parent
ONSET = "test-track rollouts of dream-trained drivers | 30 steps before leaving the road"
RANDOM = "test-track rollouts of dream-trained drivers | random start"
V_ONSET = "original data, held-out validation episodes | 30 steps before leaving the road"
V_RANDOM = "original data, held-out validation episodes | random start"
BINS = ["0-5", "5-10", "10-20", "20-40", "40-70", "70-100"]
X = [2.5, 7.5, 15, 30, 55, 85]
MODELS = ["pose_s0", "pose_s1", "ctrl_s0", "ctrl_s1", "ref_headline"]
# validated categorical slots 1-3 (dataviz reference palette, light surface); seeds differ by dash
COL = {"control": "#2a78d6", "pose": "#eb6834", "oracle": "#1baf7a", "ref": "#52514e"}
INK, MUTED = "#0b0b0b", "#52514e"


def load(name, tau):
    return json.loads((OUT / f"{name}_tau{tau}.json").read_text())["results"]


def cell(r, key):
    v = r.get(key)
    return "-" if v is None else f"{v[0]:.2f} [{v[1]:.2f}, {v[2]:.2f}]"


def style(ax, title, ylab):
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_ylabel(ylab, fontsize=8, color=MUTED)
    ax.set_xlabel("imagined steps since the start (bin centre)", fontsize=8, color=MUTED)
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.axvspan(20, 40, color="#f1f0ec", zorder=0)


def main():
    res = {(m, t): load(m, t) for m in MODELS for t in ("1.15", "0.05")}
    res[("ref_loop_head_r02", "0.05")] = load("ref_loop_head_r02", "0.05")
    summary = {"gate": json.loads((ROOT / "gate.json").read_text()),
               "gate_tau0.05": json.loads((ROOT / "gate_tau0.05.json").read_text()), "rows": {}}
    lines = []
    readouts = [("own", "own predictions"), ("oracle (own latent, true pose)", "oracle: own latent, true pose"),
                ("real + own pose", "real frames, own pose"), ("real: probe on 1-step prediction",
                "real frames (and true pose): 1-step prediction"), ("real: probe on real frame",
                "probe on the real frames (model-independent ceiling)")]
    for setname, title in ((ONSET, "Test-track windows from 30 steps before leaving the road (275)"),
                           (V_ONSET, "Validation windows from 30 steps before leaving the road (52)")):
        for b in ("20-40",):
            lines += [f"\n### {title}: off-road steps recognised, steps {b}\n",
                      "| Model | Fed | tau 1.15 | tau 0.05 |", "|---|---|---|---|"]
            for (m, t), r in sorted(res.items()):
                if t != "1.15" and (m, "1.15") in res:
                    continue
                for ro, lab in readouts:
                    if ro not in r[setname]["readouts"]:
                        continue
                    if ro == "real: probe on real frame" and m != "ctrl_s0":
                        continue
                    get = lambda tt: (cell(res[(m, tt)][setname]["readouts"][ro][b], "off_recall")
                                      if (m, tt) in res else "(0.63 published, no interval)")
                    lines.append(f"| {m if ro != 'real: probe on real frame' else '-'} | {lab} | {get('1.15')} | {get('0.05')} |")
                    summary["rows"][f"{setname} | {m} | {ro} | {b} | off_recall"] = {
                        tt: res[(m, tt)][setname]["readouts"][ro][b]["off_recall"] for tt in ("1.15", "0.05") if (m, tt) in res}
    for setname, title in ((RANDOM, "Test-track random starts (1,800)"), (V_RANDOM, "Validation random starts (210)")):
        lines += [f"\n### {title}: road-status accuracy, fed own predictions\n",
                  "| Model | 20-40, tau 1.15 | 70-100, tau 1.15 | 20-40, tau 0.05 | 70-100, tau 0.05 |", "|---|---|---|---|---|"]
        for m in MODELS + ["ref_loop_head_r02"]:
            c = [cell(res[(m, t)][setname]["readouts"]["own"][b], "acc") if (m, t) in res else "-"
                 for t in ("1.15", "0.05") for b in ("20-40", "70-100")]
            lines.append(f"| {m} | " + " | ".join([c[0], c[1], c[2], c[3]]) + " |")
            for t in ("1.15", "0.05"):
                if (m, t) in res:
                    for b in ("20-40", "70-100"):
                        summary["rows"][f"{setname} | {m} | own | {b} | acc | tau {t}"] = res[(m, t)][setname]["readouts"]["own"][b]["acc"]
    for setname, title in ((ONSET, "Test-track windows from 30 steps before leaving the road"),
                           (RANDOM, "Test-track random starts")):
        lines += [f"\n### Pose error, {title}, tau 1.15 (lateral offset in half-widths / heading in degrees / "
                  "off-road rate implied by the predicted offset vs the true off-road share)\n",
                  "| Model | Fed | " + " | ".join(BINS) + " |", "|---|---|" + "---|" * len(BINS)]
        for m in ("pose_s0", "pose_s1"):
            for ro, lab in readouts[:4]:
                r = res[(m, "1.15")][setname]["readouts"][ro]
                cells = [f"{r[b]['lat_abs_err'][0]:.2f} / {r[b]['hdg_abs_err_deg'][0]:.0f} / "
                         f"{r[b]['implied_off_rate'][0]:.2f} vs {r[b]['off_share']:.2f}" for b in BINS]
                lines.append(f"| {m} | {lab} | " + " | ".join(cells) + " |")
                summary["rows"][f"{setname} | {m} | {ro} | pose"] = {b: {k: r[b][k] for k in (
                    "lat_abs_err", "hdg_abs_err_deg", "implied_off_rate", "implied_acc", "implied_off_recall", "off_share")}
                    for b in BINS}
    (ROOT / "tables.md").write_text("\n".join(lines) + "\n")
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1))

    # Figure 1: off-road recall against horizon, onset windows, both temperatures
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, t in zip(axes, ("1.15", "0.05")):
        for m, ro, col, lab in [("ctrl", "own", "control", "control, own predictions"),
                                ("pose", "own", "pose", "pose model, own predictions"),
                                ("pose", "oracle (own latent, true pose)", "oracle", "pose model, true pose fed (oracle)")]:
            for s, ls in ((0, "-"), (1, "--")):
                r = res[(f"{m}_s{s}", t)][ONSET]["readouts"][ro]
                y = [r[b]["off_recall"][0] if r[b]["off_recall"] else float("nan") for b in BINS]
                ax.plot(X, y, ls, color=COL[col], lw=2, marker="o", ms=4, label=f"{lab}, seed {s}")
        r = res[("ctrl_s0", t)][ONSET]["readouts"]["real: probe on real frame"]
        ax.plot(X, [r[b]["off_recall"][0] if r[b]["off_recall"] else float("nan") for b in BINS], ":",
                color=COL["ref"], lw=1.5, label="probe on the real frames (ceiling)")
        ax.axhline(0.80, color=MUTED, lw=0.8)
        ax.text(98, 0.81, "gate 0.80", fontsize=7, color=MUTED, ha="right", va="bottom")
        style(ax, f"temperature {t}", "share of real off-road steps recognised")
        ax.set_ylim(0, 1)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=7, frameon=False, loc="lower center", ncol=4)
    fig.suptitle("Off-road steps recognised, 275 test-track windows started 30 steps before the car leaves "
                 "the road (steps 20-40 shaded)", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(ROOT / "fig_recall_horizon.png", dpi=150)

    # Figure 2: lateral offset error against horizon
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (setname, title) in zip(axes, ((ONSET, "starts 30 steps before leaving the road"), (RANDOM, "random starts"))):
        for ro, col, lab in [("own", "pose", "own latent, own pose"),
                             ("real + own pose", "ref", "real frames, own pose"),
                             ("oracle (own latent, true pose)", "oracle", "own latent, true pose")]:
            for s, ls in ((0, "-"), (1, "--")):
                r = res[(f"pose_s{s}", "1.15")][setname]["readouts"][ro]
                ax.plot(X, [r[b]["lat_abs_err"][0] for b in BINS], ls, color=COL[col], lw=2, marker="o", ms=4,
                        label=f"{lab}, seed {s}")
        ax.axhline(1.0, color=MUTED, lw=0.8)
        ax.text(98, 1.03, "one road half-width", fontsize=7, color=MUTED, ha="right", va="bottom")
        style(ax, f"test tracks, {title}, temperature 1.15", "mean |lateral offset error| (half-widths)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.suptitle("Predicted pose against the true pose: lateral offset error", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(ROOT / "fig_pose_error.png", dpi=150)
    print((ROOT / "tables.md").read_text())


if __name__ == "__main__":
    main()
