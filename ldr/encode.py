"""Phase 2.5: encode every frame to (mu, logvar) once, so the MDN-RNN never
touches pixels.

    python -m ldr.encode
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .config import DATA, RUNS
from .utils import get_device, load_ckpt, to_tensor
from .vae import VAE


def load_vae(path: Path = RUNS / "vae" / "best.pt", device="cpu") -> VAE:
    vae = VAE()
    vae.load_state_dict(load_ckpt(path, device)["model"])
    return vae.to(device).eval()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae", type=Path, default=RUNS / "vae" / "best.pt")
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--device", default=None)
    a = p.parse_args()
    dev = get_device(a.device)
    vae = load_vae(a.vae, dev)
    frames = np.load(DATA / "packed" / "frames.npy", mmap_mode="r")
    mu = np.zeros((len(frames), vae.z_dim), np.float32)
    logvar = np.zeros_like(mu)
    for s in tqdm(range(0, len(frames), a.batch)):
        m, lv = vae.encode(to_tensor(frames[s: s + a.batch], dev))
        mu[s: s + len(m)], logvar[s: s + len(m)] = m.cpu().numpy(), lv.cpu().numpy()
    meta = dict(np.load(DATA / "packed" / "meta.npz"))
    np.savez(DATA / "latents.npz", mu=mu, logvar=logvar, **meta)
    print(f"encoded {len(mu):,} frames -> {DATA / 'latents.npz'}")


if __name__ == "__main__":
    main()
