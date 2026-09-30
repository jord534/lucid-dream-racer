"""Phase 5B: the Lucid Dream dashboard (Gradio). Real track vs. the model's
reconstruction vs. its imagined future, with latent interventions.

    python -m ldr.app            # http://127.0.0.1:7860
"""
from __future__ import annotations

import gradio as gr
import numpy as np
import torch

from .agent import WorldModelAgent
from .config import C, RUNS, SEED_TEST
from .controller import act_np
from .envs import make_env
from .mdnrnn import mdn_sample
from .utils import load_ckpt, to_uint8, upscale

AGENT = WorldModelAgent(device="cpu")
THETA = load_ckpt(RUNS / "controller" / "best.pt")["theta"]
_cd = np.load(RUNS / "curvature_direction.npz")
DIRECTION = (_cd["direction"] * float(_cd["scale"])).astype(np.float32)


class Session:
    def __init__(self, seed: int):
        self.env = make_env()
        self.obs, _ = self.env.reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        AGENT.reset()
        self.state = AGENT.state
        self.ret, self.t = 0.0, 0

    @torch.no_grad()
    def imagine(self, z, horizon, tau, n_show=5):
        """Roll the controller forward inside the model from the current belief."""
        state = self.state
        zt = torch.from_numpy(z[None]).float()
        shown = []
        for k in range(horizon):
            a = act_np(THETA, zt[0].numpy(), state[0][0, 0].numpy())
            logit, mu, ls, _, _, state = AGENT.rnn.step(zt, torch.from_numpy(a[None]), state)
            zt = mdn_sample(logit, mu, ls, tau)
            if (k + 1) % max(1, horizon // n_show) == 0:
                shown.append(zt)
        frames = to_uint8(AGENT.vae.decode(torch.cat(shown[:n_show])))
        return np.hstack([upscale(f, 128) for f in frames])

    def step(self, bend, noise, tau, horizon, manual, steer, gas):
        AGENT.state = self.state
        z = AGENT.encode(self.obs) + bend * DIRECTION
        z = z + noise * self.rng.standard_normal(C.z_dim).astype(np.float32)
        a = (np.array([steer, gas, 0.0], np.float32) if manual
             else act_np(THETA, z, AGENT.h))
        recon = to_uint8(AGENT.vae.decode(torch.from_numpy(z[None])))[0]
        strip = self.imagine(z, horizon, tau)
        AGENT.observe(z, a)
        self.state = AGENT.state
        self.obs, r, term, trunc, _ = self.env.step(a)
        self.ret, self.t = self.ret + r, self.t + 1
        info = (f"t={self.t}  return={self.ret:.1f}  steer={a[0]:+.2f} gas={a[1]:.2f} "
                f"brake={a[2]:.2f}" + ("  [episode over]" if term or trunc else ""))
        return upscale(self.obs, 384), upscale(recon, 384), strip, info


def build() -> gr.Blocks:
    with gr.Blocks(title="Lucid Dream Racer") as demo:
        gr.Markdown("# Lucid Dream Racer\nLeft: the real track. Middle: what the agent "
                    "believes it sees. Bottom: what it imagines will happen next.")
        sess = gr.State(None)
        with gr.Row():
            real = gr.Image(label="Real environment", height=384)
            recon = gr.Image(label="VAE belief (after interventions)", height=384)
        strip = gr.Image(label="Imagined future (MDN-RNN)")
        info = gr.Textbox(label="Telemetry")
        with gr.Row():
            bend = gr.Slider(-3, 3, 0, step=0.1, label="Hallucinated curvature (std units)")
            noise = gr.Slider(0, 2, 0, step=0.05, label="Nightmare noise on z")
            tau = gr.Slider(0.1, 2.5, 1.0, step=0.05, label="Dream temperature")
            horizon = gr.Slider(5, 100, 50, step=5, label="Imagination horizon")
        with gr.Row():
            manual = gr.Checkbox(False, label="Manual override")
            steer = gr.Slider(-1, 1, 0, step=0.05, label="Steer")
            gas = gr.Slider(0, 1, 0.3, step=0.05, label="Gas")
        with gr.Row():
            seed = gr.Number(SEED_TEST, label="Track seed", precision=0)
            reset_b, step_b = gr.Button("New track"), gr.Button("Step")
            play = gr.Checkbox(False, label="Play")
        timer = gr.Timer(0.1, active=False)

        controls = [bend, noise, tau, horizon, manual, steer, gas]

        def do_reset(s):
            return Session(int(s))

        def do_step(session, *ctrl):
            session = session or Session(SEED_TEST)
            return (session, *session.step(*ctrl))

        outputs = [sess, real, recon, strip, info]
        reset_b.click(do_reset, seed, sess).then(do_step, [sess, *controls], outputs)
        step_b.click(do_step, [sess, *controls], outputs)
        timer.tick(do_step, [sess, *controls], outputs)
        play.change(lambda on: gr.Timer(active=on), play, timer)
    return demo


if __name__ == "__main__":
    build().launch()