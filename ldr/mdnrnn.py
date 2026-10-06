"""M-model: LSTM + mixture density head over each latent dimension, plus
reward and termination heads.  p(z_{t+1}, r_t, d_t | z_t, a_t, h_t).

on_head (optional, off unless asked for with return_on=True): P(a wheel is on the road after the
action), read from the LSTM output like the reward and done heads. Trained with the real on-road
flag (train_mdnrnn --on-weight); `has_on` says whether a loaded checkpoint carries a trained one."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .config import C


class MDNRNN(nn.Module):
    """shared=False (the World Models original): every latent dimension picks its own
    mixture component, so one draw can combine 5**z_dim futures, most of them not
    coherent scenes. shared=True: one component is drawn for the whole frame, so a
    sampled future is internally consistent."""

    def __init__(self, z_dim=C.z_dim, a_dim=C.a_dim, h_dim=C.h_dim, k=C.n_gauss,
                 shared: bool = C.shared_mixture):
        super().__init__()
        self.z_dim, self.a_dim, self.h_dim, self.k, self.shared = z_dim, a_dim, h_dim, k, shared
        self.lstm = nn.LSTM(z_dim + a_dim, h_dim, batch_first=True)
        self.head = nn.Linear(h_dim, (k + z_dim * k * 2 if shared else z_dim * k * 3) + 2)
        self.on_head = nn.Linear(h_dim, 1)
        self.has_on = False

    def forward(self, z, a, state=None, return_on=False):
        """z: (B,L,z), a: (B,L,3) -> logits, mu, logsig, reward, done logit, state.
        mu and logsig are (B,L,z,k); logits are (B,L,k) if shared else (B,L,z,k)."""
        out, state = self.lstm(torch.cat([z, a], -1), state)
        p = self.head(out)
        B, L, _ = p.shape
        if self.shared:
            logit = p[..., : self.k]
            mu, logsig = p[..., self.k: -2].view(B, L, self.z_dim, self.k, 2).unbind(-1)
        else:
            logit, mu, logsig = p[..., :-2].view(B, L, self.z_dim, self.k, 3).unbind(-1)
        out6 = logit, mu, logsig.clamp(-7, 2), p[..., -2], p[..., -1], state
        return (*out6, self.on_head(out)[..., 0]) if return_on else out6

    def step(self, z, a, state=None, return_on=False):
        """Single step for rollouts: z (B,z), a (B,3). With return_on, a 7th output: on-road logit."""
        o = self(z[:, None], a[:, None], state, return_on)
        out = (o[0][:, 0], o[1][:, 0], o[2][:, 0], o[3][:, 0], o[4][:, 0], o[5])
        return (*out, o[6][:, 0]) if return_on else out

    def initial_state(self, batch: int, device):
        zeros = torch.zeros(1, batch, self.h_dim, device=device)
        return zeros, zeros.clone()


def _shared(logit, mu) -> bool:
    return logit.dim() == mu.dim() - 1


def mdn_nll(logit, mu, logsig, target):
    """Negative log-likelihood of target under the mixture, per latent dimension."""
    logpi = F.log_softmax(logit, -1)
    t = target.unsqueeze(-1)
    logp = -0.5 * ((t - mu) / logsig.exp()) ** 2 - logsig - 0.5 * math.log(2 * math.pi)
    if _shared(logit, mu):
        logp = logp.sum(-2)                      # joint over dimensions, one weight per component
        return -torch.logsumexp(logpi + logp, -1).mean() / mu.shape[-2]
    return -torch.logsumexp(logpi + logp, -1).mean()


def mdn_sample(logit, mu, logsig, tau: float = 1.0):
    """Temperature-controlled sample: tau>1 = wilder dreams, tau->0 = mean-ish."""
    k = torch.distributions.Categorical(logits=logit / max(tau, 1e-3)).sample()
    if _shared(logit, mu):                       # one component for the whole frame
        idx = k.unsqueeze(-1).unsqueeze(-1).expand(*mu.shape[:-1], 1)
    else:
        idx = k.unsqueeze(-1)
    m = mu.gather(-1, idx).squeeze(-1)
    s = logsig.gather(-1, idx).squeeze(-1).exp()
    return m + s * math.sqrt(tau) * torch.randn_like(m)


def mdn_mean(logit, mu):
    """Expected value of the mixture (deterministic, blurrier)."""
    pi = F.softmax(logit, -1)
    if _shared(logit, mu):
        pi = pi.unsqueeze(-2)
    return (pi * mu).sum(-1)
