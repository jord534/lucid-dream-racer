import torch
from ldr.agent import WorldModelAgent
from ldr.envs import make_env, set_road_friction
from ldr.config import SEED_TEST
from ldr.utils import load_ckpt

if __name__ == "__main__":
    theta = load_ckpt("runs/controller_v2/best.pt")["theta"]
    for grip in (1.0, 0.8):
        agent, env = WorldModelAgent(), make_env(render_mode="human")
        obs, _ = env.reset(seed=SEED_TEST)
        set_road_friction(env, grip)
        agent.reset()
        total, t = 0.0, 0
        for t in range(1000):
            obs, r, term, trunc, _ = env.step(agent.act(theta, obs))
            total += r
            if term or trunc:
                break
        env.close()
        print(f"grip {grip}: return {total:.1f} after {t + 1} steps")
