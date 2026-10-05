"""A frozen road-status probe: world-model latent -> P(a wheel is on the road).

    python -m ldr.road_probe [--extra runs/loop_head/extra_r01.npz]        (a few minutes)

Trained on real latents (sampled from the encoder's posterior, as the world model sees them) with the
exact on-road flag from the simulator: the original episodes (labels from ldr.label_road) and the
labelled branches. train_mdnrnn --probe uses it as a task loss: the latent the dream samples must
read, through this probe, as the true road status. To keep that honest, the episodes this probe
trains on exclude the ones diagnostics/head_horizon.py trains its own (separate) probe on, so the
test probe has never been seen by the model and has not seen the same episodes.

The label of observation t+1 is the flag after action t; observation 0 counts as on the road.
Writes runs/road_probe/probe.pt (+ probe.json with the held-out numbers).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import DATA, RUNS, C

PROBE = RUNS / "road_probe" / "probe.pt"


def make_probe() -> nn.Module:
    return nn.Sequential(nn.Linear(C.z_dim, 64), nn.ReLU(), nn.Linear(64, 1))


def load_probe(path: Path = PROBE, device="cpu") -> nn.Module:
    m = make_probe()
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device).eval()


def episodes(npz: dict, on: np.ndarray, ids, need_all_known=False):
    """Yield (mu, logvar, label) for each episode id; label has T+1 entries (obs 0..T)."""
    for e in ids:
        o, a, T = int(npz["obs_start"][e]), int(npz["act_start"][e]), int(npz["ep_len"][e])
        flags = on[a : a + T]
        if need_all_known and (flags < 0).any():
            continue
        yield npz["mu"][o : o + T + 1], npz["logvar"][o : o + T + 1], np.r_[1, flags].astype(np.int8)


def gather(npz, on, ids, rng, need_all_known=False):
    X, y = [], []
    for mu, lv, lab in episodes(npz, on, ids, need_all_known):
        known = lab >= 0
        x = mu + rng.standard_normal(mu.shape).astype(np.float32) * np.exp(0.5 * lv)
        X.append(x[known]), y.append(lab[known])
    return np.concatenate(X), np.concatenate(y).astype(np.float32)


def auc(p, y):
    from scipy.stats import rankdata

    r = rankdata(p)
    n1, n0 = int((y > 0.5).sum()), int((y < 0.5).sum())
    return float((r[y > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--extra", type=Path, default=None, help="labelled branches npz to add")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", type=Path, default=PROBE)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    z = np.load(DATA / "latents.npz")
    d = {k: z[k] for k in ("mu", "logvar", "obs_start", "act_start", "ep_len", "is_val")}
    side = np.load(DATA / "latents_onroad.npy")
    nonval = np.flatnonzero(~d["is_val"])
    test_probe_eps = set(nonval[::5].tolist())  # the episodes head_horizon.py's probe trains on
    train_ids = [e for e in nonval if e not in test_probe_eps]
    X, y = gather(d, side, train_ids, rng)
    Xv, yv = gather(d, side, np.flatnonzero(d["is_val"]), rng)
    if a.extra:
        x = np.load(a.extra)
        xd = {k: x[k] for k in ("mu", "logvar", "obs_start", "act_start", "ep_len", "is_val")}
        xon = x["onroad"]
        tr = [e for e in range(len(xd["ep_len"])) if not xd["is_val"][e]]
        va = [e for e in range(len(xd["ep_len"])) if xd["is_val"][e]]
        Xe, ye = gather(xd, xon, tr, rng, need_all_known=True)
        Xev, yev = gather(xd, xon, va, rng, need_all_known=True)
        X, y = np.concatenate([X, Xe]), np.concatenate([y, ye])
        print(f"added {len(ye):,} labelled branch frames ({len(yev):,} held out)", flush=True)
    else:
        Xev = yev = None
    print(f"training on {len(y):,} frames ({1 - y.mean():.1%} off the road)", flush=True)
    probe = make_probe()
    opt = torch.optim.Adam(probe.parameters(), 2e-3, weight_decay=1e-4)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)
    for s in range(a.steps):
        ix = torch.from_numpy(rng.integers(len(yt), size=4096))
        loss = nn.functional.binary_cross_entropy_with_logits(probe(Xt[ix])[:, 0], yt[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
    probe.eval()
    res = {}
    for name, (xx, yy) in {"held-out original episodes": (Xv, yv), "held-out branches": (Xev, yev)}.items():
        if xx is None:
            continue
        with torch.no_grad():
            pr = torch.sigmoid(probe(torch.from_numpy(xx))[:, 0]).numpy()
        off = yy < 0.5
        res[name] = {"auc": auc(pr, yy), "accuracy": float(((pr > 0.5) == (yy > 0.5)).mean()),
                     "off_road_recall": float((pr[off] < 0.5).mean()), "frames": int(len(yy))}
        print(name, json.dumps({k: round(v, 3) for k, v in res[name].items()}), flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.state_dict(), a.out)
    a.out.with_suffix(".json").write_text(json.dumps(res, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
