"""Phase 3: train the MDN-RNN on encoded sequences.

    python -m ldr.train_mdnrnn --steps 20000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import DATA, RUNS, C
from .mdnrnn import MDNRNN, mdn_nll, mdn_sample
from .utils import CSVLogger, get_device, load_ckpt, save_ckpt, seed_everything


class SequenceSampler:
    def __init__(self, path: Path = DATA / "latents.npz", L: int = C.seq_len):
        d = np.load(path)
        self.mu, self.logvar = d["mu"], d["logvar"]
        self.act, self.rew, self.done = d["actions"], d["rewards"], d["dones"]
        self.L = L
        starts = {"train": [], "val": []}
        for e, T in enumerate(d["ep_len"]):
            if T < L:
                continue
            t = np.arange(T - L + 1)
            pair = np.stack([d["obs_start"][e] + t, d["act_start"][e] + t], 1)
            starts["val" if d["is_val"][e] else "train"].append(pair)
        self.starts = {k: np.concatenate(v) for k, v in starts.items()}

    def sample(self, batch, split, rng, device):
        s = self.starts[split][rng.integers(len(self.starts[split]), size=batch)]
        ar = np.arange(self.L)
        o = s[:, :1] + ar                       # obs indices t .. t+L-1
        a = s[:, 1:] + ar                       # action indices t .. t+L-1
        def z(ix):                              # sample z ~ N(mu, sigma): free augmentation
            mu, lv = torch.from_numpy(self.mu[ix]), torch.from_numpy(self.logvar[ix])
            return mu + torch.randn_like(mu) * (0.5 * lv).exp()
        t = lambda x: torch.from_numpy(x).to(device)
        return (z(o).to(device), t(self.act[a]), z(o + 1).to(device),
                t(np.clip(self.rew[a], -1, 10)), t(self.done[a].astype(np.float32)))


def loss_fn(model, batch, done_pos_weight=50.0, ss_prob: float = 0.0):
    """ss_prob > 0 is scheduled sampling: at each step, with that probability, the input
    latent is the model's own sample from the previous step instead of the recorded one.
    Teacher forcing alone never scores the model on its own outputs, which is exactly the
    regime the dream runs in."""
    z, a, z_next, r, d = batch
    if ss_prob <= 0:
        logit, mu, logsig, r_hat, d_hat, _ = model(z, a)
    else:
        B, L, _ = z.shape
        state, zin, outs = None, z[:, 0], []
        for t in range(L):
            step = model.step(zin, a[:, t], state)
            outs.append(step[:5])
            state = step[5]
            if t + 1 < L:
                own = mdn_sample(step[0], step[1], step[2], 1.0)
                use = (torch.rand(B, 1, device=z.device) < ss_prob).float()
                zin = use * own + (1 - use) * z[:, t + 1]
        logit, mu, logsig, r_hat, d_hat = [torch.stack(o, 1) for o in zip(*outs)]
    nll = mdn_nll(logit, mu, logsig, z_next)
    r_loss = F.mse_loss(r_hat, r)
    d_loss = F.binary_cross_entropy_with_logits(
        d_hat, d, pos_weight=torch.tensor(done_pos_weight, device=d.device))
    return nll + r_loss + d_loss, nll, r_loss, d_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ss-prob", type=float, default=0.0,
                   help="final scheduled-sampling probability, ramped in over the run")
    p.add_argument("--shared-mixture", action="store_true",
                   help="draw one mixture component per frame instead of per dimension")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--out", type=Path, default=RUNS / "mdnrnn")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    seed_everything(args.seed)
    dev = get_device(args.device)
    data = SequenceSampler()
    model = MDNRNN(shared=args.shared_mixture).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    start, ck = 0, args.out / "last.pt"
    if ck.exists():
        s = load_ckpt(ck, dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"]); start = s["step"]
    log, rng, best = CSVLogger(args.out / "log.csv"), np.random.default_rng(args.seed + start), 1e9

    for step in range(start, args.steps):
        ss = args.ss_prob * min(1.0, 2 * step / args.steps)   # ramp in over the first half
        loss, nll, rl, dl = loss_fn(model, data.sample(args.batch, "train", rng, dev), ss_prob=ss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # LSTMs spike
        opt.step(); sched.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            model.eval()
            with torch.no_grad():
                vl, vn, vr, vd = loss_fn(model, data.sample(256, "val", rng, dev))
                # validation always teacher-forced, so the number stays comparable across runs
            model.train()
            log.log(step=step + 1, nll=nll.item(), r_mse=rl.item(), d_bce=dl.item(),
                    val_nll=vn.item(), val_r_mse=vr.item(), val_d_bce=vd.item())
            print(f"step {step + 1}  nll {nll.item():.3f}  "
                  f"val nll {vn.item():.3f}  val r_mse {vr.item():.3f}")
            payload = dict(model=model.state_dict(), opt=opt.state_dict(),
                           sched=sched.state_dict(), step=step + 1)
            save_ckpt(ck, **payload)
            if vl.item() < best:
                best = vl.item()
                save_ckpt(args.out / "best.pt", **payload)


if __name__ == "__main__":
    main()
