"""Where does the dream lose track of whether the car is on the road?

The on-road head is 99% accurate when the world model is fed real frames and 73 to 83% accurate
inside the dream (loop_run_2026-10-02_head). Two candidate explanations:
  A. the dream's state drifts away from reality within a few steps (compounding error), so the
     head, reading that state, goes wrong;
  B. the dream cannot carry "the car is off the road" forward at all, even from a correct start.
This test separates them. Take real recorded driving with exact on-road flags (never used to train
the model: the held-out validation episodes of the original data, and test-track rollouts of
dream-trained drivers). Start the model after teacher-forcing 40 real steps, then replay the REAL
actions for 100 steps in two ways:
  teacher-forced  every input is a real frame (sampled from the encoder's posterior);
  free-running    every input is the dream's own sample (what the dream does when a controller drives it).
At every step read P(on road) from (1) the on-road head and (2) a probe on the latent (a small
classifier trained on real latents of training episodes only). Report accuracy, and recall on the
steps where the real car is off the road, against the number of steps since the start.
Two kinds of start: random, and 30 steps before the car first leaves the road (as in the failure tests).
Reading it: free-running accuracy that is high at step 1 and falls with the step count is
explanation A; already poor at step 1 on off-road steps is B (or the data). Head and probe falling
together means the dream's latent itself stops showing the road.

    python diagnostics/head_horizon.py [--model runs/loop_head/r02/mdnrnn/best.pt]   (a few minutes)
Writes reports/head_horizon.json, reports/head_horizon.png and reports/head_horizon.md."""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ldr.branch import _fit_probe
from ldr.check_dream import load_mdnrnn
from ldr.config import C, DATA, REPORTS
from ldr.mdnrnn import mdn_sample

torch.set_num_threads(2)
H, WARM, LEAD, RUN, B = 100, 40, 30, 20, 16
BINS = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 70), (70, 100)]
COND = ["teacher-forced head", "free-running head", "free-running latent probe", "teacher-forced latent probe"]


def first_off_run(on):
    n = 0
    for i, o in enumerate(on):
        n = 0 if o else n + 1
        if n == RUN:
            return i - RUN + 1
    return None


def base_episodes(d, side, eps):
    """d: the arrays of data/latents.npz, loaded once (reading an npz key again re-reads the whole array)."""
    for e in eps:
        o, a, T = int(d["obs_start"][e]), int(d["act_start"][e]), int(d["ep_len"][e])
        yield dict(mu=d["mu"][o : o + T + 1].copy(), lv=d["logvar"][o : o + T + 1].copy(),
                   act=d["actions"][a : a + T].copy(), on=side[a : a + T].astype(bool), name=f"base{e}")


def rollout_episodes(path):
    z = np.load(path)
    for s, n, seed in zip(z["start"], z["length"], z["seed"]):
        s, n = int(s), int(n)
        yield dict(mu=z["mu"][s : s + n], lv=z["logvar"][s : s + n], act=z["act"][s : s + n],
                   on=z["onroad"][s : s + n].astype(bool), name=f"{Path(path).stem}:{seed}")


@torch.no_grad()
def run_window(rnn, probe, ep, t0, W, tau, rng):
    mu, lv = torch.from_numpy(ep["mu"]), torch.from_numpy(ep["lv"])
    act = torch.from_numpy(ep["act"])
    samp = lambda k: mu[k] + torch.randn(B, C.z_dim) * (0.5 * lv[k]).exp()
    st0 = rnn.initial_state(B, "cpu")
    for k in range(t0 - min(WARM, t0), t0):
        *_, st0, _ = rnn.step(samp(k), act[k][None].expand(B, -1), st0, return_on=True)
    out = {c: [] for c in COND}
    for mode in ("tf", "free"):
        st, z = tuple(s.clone() for s in st0), samp(t0)
        for j in range(W):
            logit, m, ls, _, _, st, on = rnn.step(z, act[t0 + j][None].expand(B, -1), st, return_on=True)
            nxt = samp(t0 + j + 1) if mode == "tf" else mdn_sample(logit, m, ls, tau)
            out["teacher-forced head" if mode == "tf" else "free-running head"].append(
                torch.sigmoid(on).numpy())
            out["teacher-forced latent probe" if mode == "tf" else "free-running latent probe"].append(
                torch.sigmoid(probe(nxt)[:, 0]).numpy())
            z = nxt
    truth = ep["on"][t0 : t0 + W]
    return {c: np.stack(v) for c, v in out.items()}, truth  # (W,B) each; truth (W,)


def collect(rnn, probe, eps, starts, tau, rng, label):
    rows = {c: [[] for _ in BINS] for c in COND}
    truth_bins = [[] for _ in BINS]
    n_win = 0
    for ep in eps:
        T = len(ep["act"])
        for t0 in starts(ep, T, rng):
            W = min(H, T - 1 - t0)
            if t0 < 1 or W < 20:
                continue
            p, truth = run_window(rnn, probe, ep, t0, W, tau, rng)
            n_win += 1
            for bi, (a, b) in enumerate(BINS):
                lo, hi = a, min(b, W)
                if lo >= hi:
                    continue
                truth_bins[bi].append(np.repeat(truth[lo:hi, None], B, 1).ravel())
                for c in COND:
                    rows[c][bi].append(p[c][lo:hi].ravel())
    res = {"windows": n_win, "bins": []}
    for bi, (a, b) in enumerate(BINS):
        if not truth_bins[bi]:
            continue
        y = np.concatenate(truth_bins[bi])
        r = {"steps": f"{a}-{b}", "n": int(len(y)), "off_share": float((~y).mean())}
        for c in COND:
            pr = np.concatenate(rows[c][bi]) > 0.5
            r[c] = {"acc": float((pr == y).mean()),
                    "off_recall": float((~pr[~y]).mean()) if (~y).any() else None}
        res["bins"].append(r)
    print(f"{label}: {n_win} windows", flush=True)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("runs/loop_head/r02/mdnrnn/best.pt"))
    p.add_argument("--tau", type=float, default=C.tau)
    a = p.parse_args()
    t_start = time.time()
    rnn = load_mdnrnn(a.model)
    assert rnn.has_on, "this world model has no trained on-road head"
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    z = np.load(DATA / "latents.npz")
    d = {k: z[k] for k in ("mu", "logvar", "actions", "obs_start", "act_start", "ep_len", "is_val")}
    side = np.load(DATA / "latents_onroad.npy")
    val = np.flatnonzero(d["is_val"])
    train = np.flatnonzero(~d["is_val"])[::5]  # probe training: labelled real latents of training episodes
    X, y = [], []
    for ep in base_episodes(d, side, train):
        X.append(ep["mu"] + rng.standard_normal(ep["mu"].shape).astype(np.float32) * np.exp(0.5 * ep["lv"]))
        y.append(np.r_[True, ep["on"]])
    probe = _fit_probe(np.concatenate(X), np.concatenate(y))
    print(f"probe trained on {len(train)} training episodes ({sum(len(v) for v in y):,} frames)", flush=True)

    def random_starts(ep, T, rng, n=3):
        return [int(rng.integers(WARM, max(WARM + 1, T - H - 1))) for _ in range(n)] if T > WARM + H + 2 else []

    def onset_starts(ep, T, rng):
        o = first_off_run(ep["on"])
        return [] if o is None else [o + 1 - LEAD]

    sets = {"original data, held-out validation episodes": list(base_episodes(d, side, val))}
    del d, z  # the big arrays are no longer needed
    ro = [r for tag in ("loop_head", "loop_road") for i in range(3)
          for r in (rollout_episodes(DATA / "dream_failure" / f"rollouts_{tag}_r02_p{i}.npz")
                    if (DATA / "dream_failure" / f"rollouts_{tag}_r02_p{i}.npz").exists() else [])]
    if ro:
        sets["test-track rollouts of dream-trained drivers"] = ro
    out = {"model": str(a.model), "tau": a.tau, "results": {}}
    for name, eps in sets.items():
        for kind, fn in (("random start", random_starts), ("30 steps before leaving the road", onset_starts)):
            out["results"][f"{name} | {kind}"] = collect(rnn, probe, eps, fn, a.tau, rng, f"{name} | {kind}")
    (REPORTS / "head_horizon.json").write_text(json.dumps(out, indent=1))
    write_outputs(out, time.time() - t_start)


def write_outputs(out, secs):
    res = out["results"]
    fig, ax = plt.subplots(2, len(res), figsize=(4.2 * len(res), 6.5), squeeze=False)
    cols = dict(zip(COND, ["#1f77b4", "#d62728", "#ff7f0e", "#9ecae1"]))
    lines = []
    for k, (name, r) in enumerate(res.items()):
        xs = [b["steps"] for b in r["bins"]]
        for row, key, lab in ((0, "acc", "accuracy"), (1, "off_recall", "recall on off-road steps")):
            for c in COND:
                ax[row][k].plot(xs, [np.nan if b[c][key] is None else b[c][key] for b in r["bins"]], "o-",
                                color=cols[c], label=c, ms=4, ls="--" if "probe" in c else "-")
            ax[row][k].set_ylim(0, 1.02)
            ax[row][k].set_ylabel(lab, fontsize=8)
            ax[row][k].set_xlabel("steps since the start", fontsize=8)
        ax[0][k].set_title(name.replace(" | ", "\n"), fontsize=7)
        lines.append(f"\n### {name}  ({r['windows']} windows)\n\n| Steps | Off-road share | " +
                     " | ".join(f"{c} acc / off-road recall" for c in COND) + " |\n|---|---|" + "---|" * len(COND))
        for b in r["bins"]:
            lines.append(f"| {b['steps']} | {b['off_share']:.2f} | " + " | ".join(
                f"{b[c]['acc']:.2f} / " + ("-" if b[c]["off_recall"] is None else f"{b[c]['off_recall']:.2f}")
                for c in COND) + " |")
    ax[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(REPORTS / "head_horizon.png", dpi=120)
    (REPORTS / "head_horizon.md").write_text(
        f"# Head accuracy against dream step\n\nModel `{out['model']}`, real actions replayed, 16 dream samples "
        f"per window, temperature {out['tau']}. Made by `python diagnostics/head_horizon.py` ({secs / 60:.1f} min). "
        "See the script's docstring for the question and how to read it.\n" + "\n".join(lines) + "\n")
    print((REPORTS / "head_horizon.md").read_text())


if __name__ == "__main__":
    main()
