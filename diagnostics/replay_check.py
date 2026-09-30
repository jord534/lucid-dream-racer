"""Replay a real action sequence inside the dream: no policy involved, so any gap
is the dynamics alone.

For each driver: drive the real track, record (latent, action, reward). Then feed the
SAME actions to the MDN-RNN, teacher-forced for `warm` steps and open-loop after that,
and compare what the dream thinks those actions earn.
"""
import numpy as np, torch
from ldr.agent import WorldModelAgent
from ldr.check_dream import load_mdnrnn
from ldr.config import SEED_VAL
from ldr.envs import make_env
from ldr.mdnrnn import mdn_sample
from ldr.utils import get_device, load_ckpt

CKPTS = [("phase5", "runs/controller/best.pt"), ("dream_v2", "runs/dream_v2/best.pt")]
WARM, STEPS, TAU = 40, 400, 1.15

if __name__ == "__main__":
    dev = get_device()
    rnn = load_mdnrnn(device=dev)
    for name, ck in CKPTS:
        theta = load_ckpt(ck)["theta"]
        agent, env = WorldModelAgent(), make_env()
        obs, _ = env.reset(seed=SEED_VAL)
        agent.reset()
        Z, A, R = [], [], []
        for _ in range(STEPS):
            Z.append(agent.encode(obs))
            a = agent.act(theta, obs)
            obs, r, term, trunc, _ = env.step(a)
            A.append(a); R.append(r)
            if term or trunc:
                break
        env.close()
        Z = torch.from_numpy(np.array(Z)).to(dev)
        A = torch.from_numpy(np.array(A)).to(dev)
        n = len(A)
        with torch.no_grad():
            state, forced, openloop = rnn.initial_state(1, dev), 0.0, 0.0
            for t in range(WARM):                       # teacher forced on real latents
                logit, mu, ls, r_hat, _, state = rnn.step(Z[t:t + 1], A[t:t + 1], state)
                forced += float(r_hat[0])
            z = mdn_sample(logit, mu, ls, TAU)
            drift = []
            for t in range(WARM, n):                    # open loop, same recorded actions
                logit, mu, ls, r_hat, _, state = rnn.step(z, A[t:t + 1], state)
                openloop += float(r_hat[0])
                z = mdn_sample(logit, mu, ls, TAU)
                if (t - WARM) % 100 == 99:
                    drift.append(float(((z[0] - Z[min(t + 1, n - 1)]) ** 2).mean()))
        true_tail = float(sum(R[WARM:]))
        print(f"{name:9s} steps {n:4d} | real reward after warm-up {true_tail:8.1f} | "
              f"dream's estimate of the same actions {openloop:8.1f} | "
              f"latent error {' '.join(f'{d:.2f}' for d in drift)}")
