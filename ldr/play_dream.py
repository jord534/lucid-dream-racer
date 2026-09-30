"""Phase 5A: the playable dream. Drive inside the world model with the keyboard.

    python -m ldr.play_dream
Keys: arrows = steer/gas/brake, A = autopilot toggle, Q/E = bend track left/right,
N = nightmare noise, [ ] = temperature, R = new dream, Esc = quit.
"""
from __future__ import annotations

import argparse

import numpy as np
import pygame
import torch

from .check_dream import load_mdnrnn
from .config import RUNS
from .controller import act_batched
from .dream import DreamSim
from .encode import load_vae
from .utils import get_device, load_ckpt, to_uint8


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", default=str(RUNS / "controller" / "best.pt"))
    p.add_argument("--size", type=int, default=640)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=0, help="auto-quit (headless tests)")
    p.add_argument("--device", default=None)
    a = p.parse_args()
    dev = get_device(a.device)
    vae, rnn = load_vae(device=dev), load_mdnrnn(device=dev)
    sim = DreamSim(rnn, dev, tau=1.0)
    theta = torch.as_tensor(load_ckpt(a.controller)["theta"], device=dev)[None]
    cd = np.load(RUNS / "curvature_direction.npz")
    direction = torch.as_tensor(cd["direction"], device=dev) * float(cd["scale"])
    rng = np.random.default_rng()

    pygame.init()
    screen = pygame.display.set_mode((a.size, a.size))
    pygame.display.set_caption("Lucid Dream Racer")
    font, clock = pygame.font.SysFont("monospace", 16), pygame.time.Clock()
    z, h = sim.reset(1, rng, warm=40)
    tau, autopilot, bend, nightmare, frame = 1.0, False, 0.0, False, 0

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_a: autopilot = not autopilot
                if ev.key == pygame.K_n: nightmare = not nightmare
                if ev.key == pygame.K_r: z, h = sim.reset(1, rng, warm=40)
                if ev.key == pygame.K_LEFTBRACKET: tau = max(0.1, tau - 0.1)
                if ev.key == pygame.K_RIGHTBRACKET: tau = min(3.0, tau + 0.1)
        keys = pygame.key.get_pressed()
        bend = (keys[pygame.K_e] - keys[pygame.K_q]) * 2.0
        offset = bend * direction
        if nightmare:
            offset = offset + 0.8 * torch.randn_like(z)
        if autopilot:
            act = act_batched(theta, z + offset, h)
        else:
            act = torch.tensor([[keys[pygame.K_RIGHT] - keys[pygame.K_LEFT],
                                 float(keys[pygame.K_UP]), 0.8 * keys[pygame.K_DOWN]]],
                               dtype=torch.float32, device=dev)
        with torch.no_grad():
            img = to_uint8(vae.decode(z + offset))[0]
        z, h, _, _ = sim.step(act, tau=tau, z_offset=offset)

        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(pygame.transform.scale(surf, (a.size, a.size)), (0, 0))
        hud = (f"tau {tau:.1f}  autopilot {'ON' if autopilot else 'off'}  "
               f"bend {bend:+.0f}  nightmare {'ON' if nightmare else 'off'}")
        screen.blit(font.render(hud, True, (255, 255, 255), (0, 0, 0)), (8, 8))
        pygame.display.flip()
        clock.tick(a.fps)
        frame += 1
        if a.frames and frame >= a.frames:
            running = False
    pygame.quit()


if __name__ == "__main__":
    main()