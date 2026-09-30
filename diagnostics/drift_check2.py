"""Where does the dream leave the real latent cloud: at the first sampled step, or later?
Also measures the real-environment states each driver actually visits."""
import numpy as np, torch
from ldr.agent import WorldModelAgent
from ldr.check_dream import load_mdnrnn
from ldr.config import DATA, SEED_VAL
from ldr.controller import act_batched
from ldr.dream import DreamSim
from ldr.envs import make_env
from ldr.mdnrnn import mdn_sample
from ldr.utils import get_device, load_ckpt

CKPTS = [("phase5", "runs/controller/best.pt"), ("dream", "runs/dream_v2/best.pt")]

if __name__ == "__main__":
    dev = get_device()
    d = np.load(DATA / "latents.npz")
    mu, lv = d["mu"].astype(np.float64), d["logvar"].astype(np.float64)
    m, C = mu.mean(0), np.cov(mu.T) + 1e-4 * np.eye(mu.shape[1])
    P = np.linalg.inv(C)
    maha = lambda X: np.sqrt(np.einsum("ij,jk,ik->i", X - m, P, X - m))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(mu), 20000)
    zs = mu[pick] + rng.standard_normal((20000, mu.shape[1])) * np.exp(0.5 * lv[pick])
    print(f"reference | posterior means: median {np.median(maha(mu[pick])):.1f}, "
          f"99th {np.percentile(maha(mu[pick]), 99):.1f}")
    print(f"reference | sampled latents: median {np.median(maha(zs)):.1f}, "
          f"99th {np.percentile(maha(zs), 99):.1f}")

    rnn = load_mdnrnn(device=dev)
    # one teacher-forced step: real latent in, sampled prediction out
    idx = torch.as_tensor(pick[:2048], device=dev)
    z0 = torch.from_numpy(mu[pick[:2048]]).float().to(dev)
    a0 = torch.from_numpy(d["actions"][pick[:2048] % len(d["actions"])]).to(dev)
    with torch.no_grad():
        logit, mu_k, ls, *_ = rnn.step(z0, a0)
        for tau in (0.05, 1.0, 1.15):
            zp = mdn_sample(logit, mu_k, ls, tau).cpu().numpy().astype(np.float64)
            print(f"one-step prediction from a real latent, tau={tau:4.2f}: "
                  f"median {np.median(maha(zp)):.1f}")

    sim = DreamSim(rnn, dev, tau=1.15)
    for name, ck in CKPTS:
        th = torch.as_tensor(load_ckpt(ck)["theta"], device=dev)[None].repeat(64, 1)
        z, h = sim.reset(64, np.random.default_rng(0), warm=40)
        print(f"{name:7s} dream, first steps:", end=" ")
        for t in range(1, 21):
            z, h, r, alive = sim.step(act_batched(th, z, h))
            if t in (1, 2, 3, 5, 10, 20):
                print(f"t{t}={maha(z.cpu().numpy().astype(np.float64)).mean():.1f}", end=" ")
        print()

    for name, ck in CKPTS:                       # real environment, same measurement
        theta = load_ckpt(ck)["theta"]
        agent, env = WorldModelAgent(), make_env()
        obs, _ = env.reset(seed=SEED_VAL)
        agent.reset()
        dist, total = [], 0.0
        for t in range(600):
            z = agent.encode(obs)
            dist.append(float(maha(z[None].astype(np.float64))[0]))
            obs, r, term, trunc, _ = env.step(agent.act(theta, obs))
            total += r
            if term or trunc:
                break
        env.close()
        dist = np.array(dist)
        print(f"{name:7s} real env: return {total:7.1f} | latent distance "
              f"median {np.median(dist):.1f}, 99th {np.percentile(dist, 99):.1f}")
