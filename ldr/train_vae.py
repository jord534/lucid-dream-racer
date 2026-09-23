"""Phase 2: train the VAE.

    python -m ldr.train_vae --steps 30000 --batch 128
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import RUNS
from .data import FrameSampler
from .utils import CSVLogger, get_device, load_ckpt, save_ckpt, seed_everything, to_tensor, to_uint8
from .vae import VAE, vae_loss

from tqdm import trange


@torch.no_grad()
def save_grid(model, x, path: Path, n: int = 8):
    recon, _, _ = model(x[:n])
    top, bottom = to_uint8(x[:n]), to_uint8(recon)
    Image.fromarray(np.concatenate([np.hstack(top), np.hstack(bottom)], 0)).save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30_000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out", type=Path, default=RUNS / "vae")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    seed_everything(args.seed)
    dev = get_device(args.device)
    data = FrameSampler()
    model = VAE().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    start, ck = 0, args.out / "last.pt"
    if ck.exists():                                          # resume
        s = load_ckpt(ck, dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); start = s["step"]
    log, rng, best = CSVLogger(args.out / "log.csv"), np.random.default_rng(args.seed + start), 1e9
    (args.out / "grids").mkdir(parents=True, exist_ok=True)

    for step in trange(start, args.steps, initial=start, total=args.steps, desc="vae", dynamic_ncols=True):
        x = to_tensor(data.sample(args.batch, "train", rng), dev)
        recon, mu, logvar = model(x)
        loss, rec, kl = vae_loss(recon, x, mu, logvar)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()

        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                xv = to_tensor(data.sample(512, "val", rng), dev)
                rv, muv, lvv = model(xv)
                _, rec_v, kl_v = vae_loss(rv, xv, muv, lvv)
                mse_px = torch.mean((model.decode(muv) - xv) ** 2).item()
            save_grid(model, xv, args.out / "grids" / f"{step + 1:06d}.png")
            model.train()
            log.log(step=step + 1, loss=loss.item(), rec=rec.item(), kl=kl.item(),
                    val_rec=rec_v.item(), val_kl=kl_v.item(), val_mse_px=mse_px)
            print(f"step {step + 1}  rec {rec.item():.1f}  kl {kl.item():.1f}  "
                  f"val mse/px {mse_px:.5f}")
            payload = dict(model=model.state_dict(), opt=opt.state_dict(),
                           sched=sched.state_dict(), step=step + 1)
            save_ckpt(ck, **payload)
            if mse_px < best:
                best = mse_px
                save_ckpt(args.out / "best.pt", **payload)


if __name__ == "__main__":
    main()