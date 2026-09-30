"""Phase 5 prep: find a latent direction that encodes upcoming track curvature
with ridge regression on the privileged curvature label.

    python -m ldr.directions
"""
from __future__ import annotations

import numpy as np

from .config import DATA, RUNS


def main(lam: float = 1.0, skip_intro: int = 60):
    d = np.load(DATA / "latents.npz")
    mask = np.ones(len(d["mu"]), bool)
    for s in d["obs_start"]:
        mask[s: s + skip_intro] = False                  # zoom-in frames have no track view
    val = np.zeros_like(mask)
    for e in np.flatnonzero(d["is_val"]):
        s = d["obs_start"][e]; val[s: s + d["ep_len"][e] + 1] = True
    X, y = d["mu"], d["curv"]
    tr, te = mask & ~val, mask & val
    mx, my = X[tr].mean(0), y[tr].mean()
    Xc = X[tr] - mx
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ (y[tr] - my))
    pred = (X[te] - mx) @ w + my
    ss_tot = ((y[te] - y[te].mean()) ** 2).sum()
    r2 = float(1 - ((y[te] - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")
    direction = w / np.linalg.norm(w)
    scale = float(((X[tr] - mx) @ direction).std())      # 1 slider unit = 1 std along it
    RUNS.mkdir(exist_ok=True)
    np.savez(RUNS / "curvature_direction.npz", direction=direction.astype(np.float32),
             scale=scale, r2=r2)
    print(f"curvature direction: held-out R^2 = {r2:.3f}, std along direction = {scale:.3f}")


if __name__ == "__main__":
    main()