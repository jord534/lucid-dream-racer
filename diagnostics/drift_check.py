"""How far outside the real latent distribution does each driver's dream wander?"""
import numpy as np, torch
from ldr.check_dream import load_mdnrnn
from ldr.controller import act_batched
from ldr.dream import DreamSim
from ldr.config import DATA
from ldr.utils import get_device, load_ckpt

if __name__ == "__main__":
    dev = get_device()
    d = np.load(DATA / "latents.npz")
    mu = d["mu"].astype(np.float64)
    m, C = mu.mean(0), np.cov(mu.T) + 1e-4 * np.eye(mu.shape[1])
    P = np.linalg.inv(C)
    maha = lambda X: np.sqrt(np.einsum("ij,jk,ik->i", X - m, P, X - m))
    ref = np.percentile(maha(mu[np.random.default_rng(0).choice(len(mu), 20000)]), [50, 99])
    print(f"real latents: median {ref[0]:.1f}, 99th pct {ref[1]:.1f} (32 dims -> expect ~5.6)")
    sim = DreamSim(load_mdnrnn(device=dev), dev, tau=1.15)
    for name, ck in [("phase5", "runs/controller/best.pt"),
                     ("dream", "runs/dream_v2/best.pt")]:
        th = torch.as_tensor(load_ckpt(ck)["theta"], device=dev)[None].repeat(64, 1)
        z, h = sim.reset(64, np.random.default_rng(0), warm=40)
        dist, rew = [], 0.0
        for t in range(200):
            z, h, r, alive = sim.step(act_batched(th, z, h))
            rew += float(r.mean())
            if t % 20 == 19:
                dist.append(maha(z.cpu().numpy().astype(np.float64)).mean())
        print(f"{name:7s} dream reward {rew:7.1f} | distance at steps 20..200: "
              + " ".join(f"{x:.1f}" for x in dist))
