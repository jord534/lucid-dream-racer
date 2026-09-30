"""Phase 3 verification: teacher-force 5 real frames, then dream 100 steps
open-loop with the real recorded actions. Compares against a copy-last-frame
baseline and writes a side-by-side GIF.

    python -m ldr.check_dream --episodes 50
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .config import DATA, REPORTS, RUNS, C
from .encode import load_vae
from .mdnrnn import MDNRNN, mdn_mean, mdn_sample
from .utils import get_device, load_ckpt, to_uint8, upscale


def load_mdnrnn(path=RUNS / "mdnrnn" / "best.pt", device="cpu") -> MDNRNN:
    """Infers shared vs per-dimension mixtures from the checkpoint's head size, so
    models trained before that option existed still load."""
    sd = load_ckpt(path, device)["model"]
    out = sd["head.weight"].shape[0]
    m = MDNRNN(shared=out == C.n_gauss + C.z_dim * C.n_gauss * 2 + 2)
    m.load_state_dict(sd)
    return m.to(device).eval()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--context", type=int, default=5)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--offset", type=int, default=100, help="skip the zoom-in intro")
    p.add_argument("--tau", type=float, default=0.05,
                   help="low = fair error measurement; 1.0 = what the dream really looks like")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    dev = get_device(a.device)
    vae, rnn = load_vae(device=dev), load_mdnrnn(device=dev)
    d = np.load(DATA / "latents.npz")
    need = a.offset + a.context + a.horizon
    eps = [e for e in np.flatnonzero(d["is_val"]) if d["ep_len"][e] >= need][: a.episodes]
    assert eps, "no validation episodes long enough"
    err_model = np.zeros((len(eps), a.horizon)); err_copy = np.zeros_like(err_model)
    gif = None
    for i, e in enumerate(eps):
        o0, a0 = d["obs_start"][e] + a.offset, d["act_start"][e] + a.offset
        mu = torch.from_numpy(d["mu"][o0: o0 + need]).to(dev)
        act = torch.from_numpy(d["actions"][a0: a0 + need]).to(dev)
        state = rnn.initial_state(1, dev)
        for t in range(a.context):                      # teacher forcing (warm-up)
            logit, m, ls, _, _, state = rnn.step(mu[t:t + 1], act[t:t + 1], state)
        z, dream = mdn_mean(logit, m), []
        for t in range(a.horizon):                      # open loop
            dream.append(z)
            tt = a.context + t
            logit, m, ls, _, _, state = rnn.step(z, act[tt:tt + 1], state)
            z = mdn_sample(logit, m, ls, a.tau)
        dream = torch.cat(dream)
        real = mu[a.context: a.context + a.horizon]
        err_model[i] = ((dream - real) ** 2).mean(1).cpu().numpy()
        err_copy[i] = ((mu[a.context - 1] - real) ** 2).mean(1).cpu().numpy()
        if gif is None:                                  # GIF of the first episode
            gif = [np.hstack([upscale(r, 256), upscale(f, 256)]) for r, f in
                   zip(to_uint8(vae.decode(real)), to_uint8(vae.decode(dream)))]
    REPORTS.mkdir(exist_ok=True)
    frames = [Image.fromarray(g) for g in gif]
    frames[0].save(REPORTS / "dream_check.gif", save_all=True,
                   append_images=frames[1:], duration=40, loop=0)
    plt.figure(figsize=(6, 3.5))
    for err, lab in [(err_model, "MDN-RNN dream"), (err_copy, "copy last frame")]:
        mean, sd = err.mean(0), err.std(0) / np.sqrt(len(eps))
        plt.plot(mean, label=lab)
        plt.fill_between(range(a.horizon), mean - sd, mean + sd, alpha=.3)
    plt.xlabel("steps into the dream")
    plt.ylabel("latent MSE vs real")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "dream_check.png", dpi=150)
    ahead = err_model.mean(0) < err_copy.mean(0)
    behind_run = np.convolve(~ahead, np.ones(10, int), mode="valid") == 10
    lost_at = int(np.argmax(behind_run)) if behind_run.any() else a.horizon
    print(f"dream ahead of copy-last-frame on {ahead.sum()} of {a.horizon} steps; "
          f"first falls behind for 10+ straight steps at step {lost_at}")


if __name__ == "__main__":
    main()
