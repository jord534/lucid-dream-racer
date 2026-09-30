"""V-model: convolutional VAE, 64x64x3 -> z in R^32 (Ha & Schmidhuber 2018)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import C


class VAE(nn.Module):
    def __init__(self, z_dim: int = C.z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(                       # 64 -> 31 -> 14 -> 6 -> 2
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(),
            nn.Flatten(),                               # 256 * 2 * 2 = 1024
        )
        self.fc_mu = nn.Linear(1024, z_dim)
        self.fc_logvar = nn.Linear(1024, z_dim)
        self.fc_dec = nn.Linear(z_dim, 1024)
        self.dec = nn.Sequential(                       # 1 -> 5 -> 13 -> 30 -> 64
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 5)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(self.fc_dec(z).view(-1, 1024, 1, 1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()   # reparameterisation
        return self.decode(z), mu, logvar


def vae_loss(recon, x, mu, logvar, free_nats: float = C.kl_free_nats):
    rec = F.mse_loss(recon, x, reduction="none").sum((1, 2, 3)).mean()
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(1).mean()
    kl_term = torch.clamp(kl, min=free_nats * mu.shape[1])   # KL tolerance
    return rec + kl_term, rec, kl
