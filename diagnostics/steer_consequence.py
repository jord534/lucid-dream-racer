"""Does the dream punish a steering error the way the track does?

Hold a constant hard steering input from a mid-track state, in the dream and in the real
simulator, and compare what each pays. On a real track, locking the wheel at speed runs
you off the road and the reward stops. If the dream keeps paying, it is regenerating road
under the car instead of modelling the consequence, and no amount of search inside it can
teach cornering."""
import numpy as np, torch
from ldr.agent import WorldModelAgent
from ldr.check_dream import load_mdnrnn
from ldr.config import SEED_VAL
from ldr.dream import DreamSim
from ldr.envs import PursuitPolicy, make_env
from ldr.utils import get_device

WARM, HOLD, BATCH, TRACKS = 40, 100, 64, 8
ACTIONS = {"straight": [0.0, 0.5, 0.0], "half left": [-0.5, 0.5, 0.0],
           "hard left": [-1.0, 0.5, 0.0], "hard right": [1.0, 0.5, 0.0]}

if __name__ == "__main__":
    dev = get_device()
    sim = DreamSim(load_mdnrnn(device=dev), dev, tau=1.15)
    print(f"predicted reward over {HOLD} dream steps (batch {BATCH}):")
    for name, a in ACTIONS.items():
        z, h = sim.reset(BATCH, np.random.default_rng(0), warm=WARM)
        act = torch.tensor([a], dtype=torch.float32, device=dev).repeat(BATCH, 1)
        tot = 0.0
        for _ in range(HOLD):
            z, h, r, alive = sim.step(act)
            tot += float(r.mean())
        print(f"  {name:11s} {tot:8.1f}")

    print(f"true reward over the same {HOLD} steps on {TRACKS} real tracks:")
    for name, a in ACTIONS.items():
        act, totals = np.array(a, np.float32), []
        for k in range(TRACKS):
            env = make_env()
            obs, _ = env.reset(seed=SEED_VAL + k)
            pol = PursuitPolicy(np.random.default_rng(k))       # drive properly to mid-track
            for _ in range(WARM + 60):
                obs, *_ = env.step(pol(env))
            tot = 0.0
            for _ in range(HOLD):
                obs, r, term, trunc, _ = env.step(act)
                tot += r
                if term or trunc:
                    break
            env.close()
            totals.append(tot)
        print(f"  {name:11s} {np.mean(totals):8.1f}  +/- {np.std(totals):5.1f}")
