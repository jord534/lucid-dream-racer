"""Split env.step into Box2D physics and the pygame render of the 96x96 observation."""
import time
import numpy as np
from ldr.envs import make_env
from ldr.config import SEED_VAL

N = 400
if __name__ == "__main__":
    env = make_env()
    u = env.unwrapped
    env.reset(seed=SEED_VAL)
    a = np.array([0.0, 0.3, 0.0], np.float32)
    t_step = t_render = 0.0
    for _ in range(N):
        t0 = time.perf_counter()
        env.step(a)                       # physics + render
        t1 = time.perf_counter()
        u._render("state_pixels")         # render only, repeated
        t2 = time.perf_counter()
        t_step += t1 - t0
        t_render += t2 - t1
    env.close()
    print(f"env.step total   {t_step / N * 1000:5.2f} ms")
    print(f"render only      {t_render / N * 1000:5.2f} ms  "
          f"({100 * t_render / t_step:4.1f}% of step)")
    print(f"physics (approx) {(t_step - t_render) / N * 1000:5.2f} ms")
