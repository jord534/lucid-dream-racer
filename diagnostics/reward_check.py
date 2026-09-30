"""Is the reward head accurate on REAL states? Teacher-forced, no imagination:
drive the real track, push the actual (z, a) through the model, compare its predicted
reward with the true reward. Isolates the reward head from the dynamics."""
import torch
from ldr.agent import WorldModelAgent
from ldr.check_dream import load_mdnrnn
from ldr.config import SEED_VAL
from ldr.envs import make_env
from ldr.utils import load_ckpt

CKPTS = [("real-trained", "runs/controller/best.pt"), ("dream-trained", "runs/dream_v2/best.pt")]
STEPS = 600

if __name__ == "__main__":
    rnn = load_mdnrnn()
    for name, ck in CKPTS:
        theta = load_ckpt(ck)["theta"]
        agent, env = WorldModelAgent(), make_env()
        obs, _ = env.reset(seed=SEED_VAL)
        agent.reset()
        state = rnn.initial_state(1, torch.device("cpu"))
        true, pred = 0.0, 0.0
        with torch.no_grad():
            for _ in range(STEPS):
                z = agent.encode(obs)
                a = agent.act(theta, obs)
                *_, r_hat, _, state = rnn.step(torch.from_numpy(z[None]),
                                               torch.from_numpy(a[None]), state)
                pred += float(r_hat[0])
                obs, r, term, trunc, _ = env.step(a)
                true += r
                if term or trunc:
                    break
        env.close()
        print(f"{name:14s} true {true:8.1f}   predicted on the same real states {pred:8.1f}")
