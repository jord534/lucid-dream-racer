"""C-model: a single linear layer from [z_t, h_t] to actions (867 parameters)."""
from __future__ import annotations

import numpy as np
import torch

from .config import C

IN_DIM = C.z_dim + C.h_dim
N_PARAMS = IN_DIM * C.a_dim + C.a_dim


def squash(o):
    """Raw outputs -> valid CarRacing action (works for numpy and torch)."""
    lib = torch if isinstance(o, torch.Tensor) else np
    t = lib.tanh(o)
    steer, gas, brake = t[..., 0], (t[..., 1] + 1) / 2, lib.clip(t[..., 2], 0, 1)
    return lib.stack([steer, gas, brake], -1)


def act_np(theta: np.ndarray, z: np.ndarray, h: np.ndarray) -> np.ndarray:
    W = theta[: IN_DIM * C.a_dim].reshape(C.a_dim, IN_DIM)
    b = theta[IN_DIM * C.a_dim:]
    return squash(W @ np.concatenate([z, h]) + b).astype(np.float32)


def act_batched(theta: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """theta: (B, N_PARAMS), z: (B, z), h: (B, h) -> (B, 3). One controller per row."""
    W = theta[:, : IN_DIM * C.a_dim].view(-1, C.a_dim, IN_DIM)
    b = theta[:, IN_DIM * C.a_dim:]
    x = torch.cat([z, h], -1).unsqueeze(-1)
    return squash(torch.bmm(W, x).squeeze(-1) + b)