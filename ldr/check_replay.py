"""Policy-free closed-loop check: feed the model recorded actions open-loop and compare
what it thinks they earn with what they actually earned.

No controller involved, so it works between retraining the predictor and retraining the
controller, and it measures the dynamics rather than any policy.

    python -m ldr.check_replay --mdnrnn runs/mdnrnn_v2/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .check_dream import load_mdnrnn
from .config import DATA, RUNS, C
from .mdnrnn import mdn_sample
from .utils import get_device


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mdnrnn", type=Path, default=RUNS / "mdnrnn" / "best.pt")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--warm", type=int, default=40)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--tau", type=float, default=C.tau)
    p.add_argument("--best-only", action="store_true",
                   help="only replay episodes from the better-scoring half of the data")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    dev = get_device(a.device)
    rnn = load_mdnrnn(a.mdnrnn, dev)
    d = np.load(DATA / "latents.npz")
    need = a.warm + a.horizon
    ret = np.array([d["rewards"][s: s + T].sum() for s, T in zip(d["act_start"], d["ep_len"])])
    eps = [e for e in np.flatnonzero(d["is_val"]) if d["ep_len"][e] >= need]
    if a.best_only:
        eps = [e for e in eps if ret[e] >= np.median(ret)]
    eps = eps[: a.episodes]
    assert eps, "no validation episodes long enough"

    true_tot, pred_tot, errs = [], [], []
    for e in eps:
        o, k = d["obs_start"][e], d["act_start"][e]
        mu = torch.from_numpy(d["mu"][o: o + need + 1]).to(dev)
        act = torch.from_numpy(d["actions"][k: k + need]).to(dev)
        state = rnn.initial_state(1, dev)
        for t in range(a.warm):                       # teacher forced: real latents in
            logit, m, ls, _, _, state = rnn.step(mu[t:t + 1], act[t:t + 1], state)
        z, pred = mdn_sample(logit, m, ls, a.tau), 0.0
        for t in range(a.warm, need):                 # open loop: same recorded actions
            logit, m, ls, r_hat, _, state = rnn.step(z, act[t:t + 1], state)
            pred += float(r_hat[0])
            z = mdn_sample(logit, m, ls, a.tau)
        true_tot.append(float(d["rewards"][k + a.warm: k + need].sum()))
        pred_tot.append(pred)
        errs.append(float(((z[0] - mu[need]) ** 2).mean()))
    true_tot, pred_tot = np.array(true_tot), np.array(pred_tot)
    print(f"{len(eps)} episodes, {a.horizon} open-loop steps, tau={a.tau}")
    print(f"  true reward      mean {true_tot.mean():8.1f}")
    print(f"  predicted        mean {pred_tot.mean():8.1f}  "
          f"(ratio {pred_tot.mean() / max(abs(true_tot.mean()), 1e-9):5.2f})")
    ok = len(eps) > 2 and true_tot.std() > 0 and pred_tot.std() > 0
    corr = np.corrcoef(true_tot, pred_tot)[0, 1] if ok else float("nan")
    print(f"  per-episode correlation {corr:5.2f}" + ("" if ok else "  (needs >2 episodes)"))
    print(f"  latent MSE at the end of the rollout {np.mean(errs):5.2f} "
          f"(signal variance {d['mu'].var():.2f})")


if __name__ == "__main__":
    main()
