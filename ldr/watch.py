"""Live view of a controller run: redraws the curves every few seconds, shows an
ETA, and (with --film) records the current best driver whenever it improves.

    python -m ldr.watch                                   # real-track run (Phase 5)
    python -m ldr.watch --run runs/dream_controller --generations 500   # Phase 6
    python -m ldr.watch --film                            # also save GIFs of the best driver
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from .config import REPORTS, RUNS, SEED_VAL


def film(ckpt: Path, out: Path, seed: int = SEED_VAL, max_steps: int = 1000) -> float:
    """Drive one validation track with the checkpoint and save it as a GIF."""
    import torch
    from PIL import Image

    from .agent import WorldModelAgent
    from .envs import make_env
    theta = torch.load(ckpt, weights_only=False)["theta"]
    agent, env = WorldModelAgent(), make_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    agent.reset()
    frames, total = [], 0.0
    for t in range(max_steps):
        obs, r, term, trunc, _ = env.step(agent.act(theta, obs))
        total += r
        if t % 3 == 0:
            img = Image.fromarray(env.render())
            img.thumbnail((360, 360))
            frames.append(img)
        if term or trunc:
            break
    env.close()
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=60, loop=0)
    return total


def draw(fig, axes, d: pd.DataFrame, total: int, name: str) -> str:
    for ax in axes:
        ax.clear()
    ax = axes[0]
    if "fit_mean" in d:                                       # real-track run
        ax.plot(d.gen, d.fit_mean, alpha=.6, label="population average (training tracks)")
        ax.plot(d.gen, d.fit_max, alpha=.35, label="best candidate this round")
        val = d.dropna(subset=["val_return"]) if "val_return" in d else d.iloc[:0]
        ax.plot(val.gen, val.val_return, "o-", label="validation (unseen tracks)")
    else:                                                     # dream run
        ax.plot(d.gen, d.dream_fit_mean, alpha=.6, label="average score inside the dream")
        real = d.dropna(subset=["real_return"]) if "real_return" in d else d.iloc[:0]
        ax.plot(real.gen, real.real_return, "o-", label="same driver on the real track")
    ax.axhline(900, ls="--", lw=1, c="grey")
    ax.set_ylabel("score per run (return)")
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    if "sigma" in d:
        ax.plot(d.gen, d.sigma, c="tab:purple")
        ax.set_yscale("log")
        ax.set_ylabel("search spread (sigma)")
    else:
        ax.set_visible(False)

    ax = axes[2]
    sec = d.wall_s.diff()
    ax.plot(d.gen, sec, c="tab:gray")
    ax.set_ylabel("seconds per generation")
    ax.set_xlabel("generation")

    done = int(d.gen.iloc[-1])
    rate = float(sec.tail(10).median()) if len(d) > 1 else float("nan")
    eta_h = (total - done) * rate / 3600
    status = (f"{name}: generation {done}/{total}  |  {rate:.0f} s/gen  |  "
              f"about {eta_h:.1f} h left")
    fig.suptitle(status, fontsize=10)
    fig.tight_layout()
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, default=RUNS / "controller")
    p.add_argument("--generations", type=int, default=300, help="the total you launched with")
    p.add_argument("--every", type=float, default=30, help="seconds between refreshes")
    p.add_argument("--film", action="store_true", help="GIF the best driver on each improvement")
    p.add_argument("--no-window", action="store_true", help="only write the PNG (headless)")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    if a.no_window:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS.mkdir(exist_ok=True)
    png = REPORTS / f"{a.run.name}_live.png"
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True,
                             gridspec_kw=dict(height_ratios=[3, 1, 1]))
    if not a.no_window:
        plt.ion()
        plt.show()
    last_best = None
    while True:
        log = a.run / "log.csv"
        if log.exists() and len(pd.read_csv(log)) > 0:
            status = draw(fig, axes, pd.read_csv(log), a.generations, a.run.name)
            fig.savefig(png, dpi=110)
            print(time.strftime("%H:%M:%S"), status, flush=True)
            best = a.run / "best.pt"
            if a.film and best.exists() and best.stat().st_mtime != last_best:
                last_best = best.stat().st_mtime
                gif = REPORTS / f"{a.run.name}_best.gif"
                score = film(best, gif)
                print(f"  filmed new best driver: {score:.1f} on track {SEED_VAL} -> {gif}")
        else:
            print("waiting for the first generation to finish ...", flush=True)
        if a.once:
            break
        if a.no_window:
            time.sleep(a.every)
        else:
            plt.pause(a.every)


if __name__ == "__main__":
    main()