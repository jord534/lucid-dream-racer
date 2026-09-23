"""M-model: LSTM + mixture density head over each latent dimension, plus
reward and termination heads.  p(z_{t+1}, r_t, d_t | z_t, a_t, h_t)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import C


class MDNRNN(nn.Module):
    def __init__(self, z_dim=C.z_dim, a_dim=C.a_dim, h_dim=C.h_dim, k=C.n_gauss):
        super().__init__()
        self.z_dim, self.h_dim, self.k = z_dim, h_dim, k
        self.lstm = nn.LSTM(z_dim + a_dim, h_dim, batch_first=True)
        self.head = nn.Linear(h_dim, z_dim * k * 3 + 2)

    def forward(self, z, a, state=None):
        """z: (B,L,z), a: (B,L,3) -> mixture params (B,L,z,k) x3, reward, done logit, state."""
        out, state = self.lstm(torch.cat([z, a], -1), state)
        p = self.head(out)
        B, L, _ = p.shape
        mix = p[..., :-2].view(B, L, self.z_dim, self.k, 3)
        logit, mu, logsig = mix.unbind(-1)
        return logit, mu, logsig.clamp(-7, 2), p[..., -2], p[..., -1], state

    def step(self, z, a, state=None):
        """Single step for rollouts: z (B,z), a (B,3)."""
        logit, mu, logsig, r, d, state = self(z[:, None], a[:, None], state)
        return logit[:, 0], mu[:, 0], logsig[:, 0], r[:, 0], d[:, 0], state

    def initial_state(self, batch: int, device):
        zeros = torch.zeros(1, batch, self.h_dim, device=device)
        return zeros, zeros.clone()


def mdn_nll(logit, mu, logsig, target):
    """Negative log-likelihood of target under a per-dimension Gaussian mixture."""
    logpi = F.log_softmax(logit, -1)
    t = target.unsqueeze(-1)
    logp = -0.5 * ((t - mu) / logsig.exp()) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
    return -torch.logsumexp(logpi + logp, -1).mean()


def mdn_sample(logit, mu, logsig, tau: float = 1.0):
    """Temperature-controlled sample: tau>1 = wilder dreams, tau->0 = mean-ish."""
    idx = torch.distributions.Categorical(logits=logit / max(tau, 1e-3)).sample()
    m = mu.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
    s = logsig.gather(-1, idx.unsqueeze(-1)).squeeze(-1).exp()
    return m + s * math.sqrt(tau) * torch.randn_like(m)


def mdn_mean(logit, mu):
    """Expected value of the mixture (deterministic, blurrier)."""
    return (F.softmax(logit, -1) * mu).sum(-1)