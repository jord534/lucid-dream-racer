"""Where does a controller rollout spend its time: physics, VAE, LSTM, or overhead?"""
import time
import numpy as np, torch
from ldr.agent import WorldModelAgent
from ldr.config import SEED_VAL
from ldr.controller import N_PARAMS, act_np
from ldr.envs import make_env
from ldr.utils import preprocess, to_tensor

N = 300
if __name__ == "__main__":
    torch.set_num_threads(1)
    ag, env = WorldModelAgent(), make_env()
    theta = np.zeros(N_PARAMS, np.float32)
    obs, _ = env.reset(seed=SEED_VAL)
    ag.reset()
    t = {}
    with torch.inference_mode():
        for _ in range(N):
            t0 = time.perf_counter()
            x = to_tensor(preprocess(obs)[None], ag.dev)
            t1 = time.perf_counter()
            z = ag.vae.encode(x)[0][0].cpu().numpy()
            t2 = time.perf_counter()
            a = act_np(theta, z, ag.h)
            t3 = time.perf_counter()
            ag.observe(z, a)
            t4 = time.perf_counter()
            obs, r, term, trunc, _ = env.step(a)
            t5 = time.perf_counter()
            for k, dt in [("preprocess", t1 - t0), ("vae", t2 - t1), ("controller", t3 - t2),
                          ("lstm", t4 - t3), ("env.step", t5 - t4)]:
                t[k] = t.get(k, 0) + dt
            if term or trunc:
                obs, _ = env.reset(seed=SEED_VAL); ag.reset()
    env.close()
    total = sum(t.values())
    for k, v in sorted(t.items(), key=lambda kv: -kv[1]):
        print(f"{k:11s} {v / N * 1000:6.2f} ms/step  ({100 * v / total:4.1f}%)")
    print(f"{'total':11s} {total / N * 1000:6.2f} ms/step -> {N / total:5.0f} steps/s per worker")
