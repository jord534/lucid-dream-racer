"""Film one driver twice: inside the dream and on a real track, side by side.

    python -m ldr.film_dream --ckpt runs/dream_controller/best.pt --seed 1000000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from .agent import WorldModelAgent
from .check_dream import load_mdnrnn
from .config import REPORTS, RUNS, SEED_TEST
from .controller import act_batched
from .dream import DreamSim
from .encode import load_vae
from .envs import make_env
from .utils import load_ckpt, to_uint8

SIZE = 320


def label(img: Image.Image, top: str, bottom: str) -> Image.Image:
    out = Image.new("RGB", (SIZE, SIZE + 40), "black")
    out.paste(img, (0, 20))
    d = ImageDraw.Draw(out)
    d.text((6, 5), top, fill="white")
    d.text((6, SIZE + 25), bottom, fill="white")
    return out


@torch.no_grad()
def film_real(theta, seed, steps):
    agent, env = WorldModelAgent(), make_env(render_mode="rgb_array")
    obs, _ = env.reset(seed=seed)
    agent.reset()
    frames, total = [], 0.0
    for t in range(steps):
        obs, r, term, trunc, _ = env.step(agent.act(theta, obs))
        total += r
        frames.append(label(ImageOps.fit(Image.fromarray(env.render()), (SIZE, SIZE)),
                            f"real track (seed {seed})", f"step {t + 1}   score {total:7.1f}"))
        if term or trunc:
            break
    env.close()
    return frames


@torch.no_grad()
def film_dream(theta, steps, tau, warm, seed):
    vae, rnn = load_vae(), load_mdnrnn()
    sim = DreamSim(rnn, torch.device("cpu"), tau=tau)
    z, h = sim.reset(1, np.random.default_rng(seed), warm=warm)
    th = torch.as_tensor(theta, dtype=torch.float32)[None]
    frames, total = [], 0.0
    for t in range(steps):
        img = Image.fromarray(to_uint8(vae.decode(z))[0]).resize((SIZE, SIZE), Image.NEAREST)
        z, h, r, alive = sim.step(act_batched(th, z, h))
        total += float(r[0])
        frames.append(label(img, f"the dream (tau {tau})",
                            f"step {t + 1}   predicted {total:7.1f}"))
        if float(alive[0]) == 0:
            break
    return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=RUNS / "dream_controller" / "best.pt")
    p.add_argument("--seed", type=int, default=SEED_TEST)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--tau", type=float, default=1.15)
    p.add_argument("--warm", type=int, default=0, help="real steps fed in before dreaming")
    p.add_argument("--stride", type=int, default=2, help="keep every Nth frame in the GIF")
    p.add_argument("--out", type=Path, default=REPORTS / "dream_vs_real.gif")
    a = p.parse_args()
    theta = load_ckpt(a.ckpt)["theta"]
    left = film_dream(theta, a.steps, a.tau, a.warm, a.seed)
    right = film_real(theta, a.seed, a.steps)
    n = max(len(left), len(right))
    pad = lambda f: f + [f[-1]] * (n - len(f))          # hold the last frame of the shorter run
    pairs = []
    for l, r in zip(pad(left)[::a.stride], pad(right)[::a.stride]):
        canvas = Image.new("RGB", (SIZE * 2 + 8, SIZE + 40), "black")
        canvas.paste(l, (0, 0))
        canvas.paste(r, (SIZE + 8, 0))
        pairs.append(canvas)
    a.out.parent.mkdir(exist_ok=True)
    pairs[0].save(a.out, save_all=True, append_images=pairs[1:], duration=60, loop=0)
    print(f"{len(pairs)} frames -> {a.out}")


if __name__ == "__main__":
    main()
