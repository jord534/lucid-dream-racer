"""M-model: LSTM + mixture density head over each latent dimension, plus
reward and termination heads.  p(z_{t+1}, r_t, d_t | z_t, a_t, h_t).

on_head (optional, off unless asked for with return_on=True): P(a wheel is on the road after the
action), read from the LSTM output like the reward and done heads. Trained with the real on-road
flag (train_mdnrnn --on-weight); `has_on` says whether a loaded checkpoint carries a trained one.

pose (optional, off by default: pose_dim=0): the car relative to the road (ldr.label_pose: lateral
offset, heading error as sin and cos, speed), in the normalised features of pose_features().
pose_dim > 0 adds pose_head, a regression from the LSTM output to the pose after the action, q_{t+1};
pose_input=True also feeds the current pose q_t to the LSTM, input [z, a, q]. In training q_t is the
true pose (sometimes the model's own prediction, scheduled sampling); in the dream it is the model's
own prediction from the step before, so the state carries the car's place on the road forward.
Callers pass q= and ask for the prediction with return_pose=True; without them the outputs are
exactly those of a model without pose.

road map (optional, off by default: map_dim=0; ldr.road_map): map_dim > 0 feeds the road patch read
off a map at the car's pose as an extra LSTM input, [z, a, (q,) patch], and adds motion_head, the car's
motion over the step (du, dv, da in the car frame, divided by road_map.MOTION_SCALE). Callers pass
patch= and ask for the motion with return_motion=True (it comes last)."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import C


POSE_DIM = 4
POSE_LAT_CLIP = 5.0   # half-widths; beyond this the car is far out on the grass either way
POSE_SPEED_SCALE = 50.0


def pose_features(q):
    """Raw pose labels (lateral offset in half-widths, sin, cos, speed) -> network features:
    lateral clipped to +-POSE_LAT_CLIP and halved, speed / POSE_SPEED_SCALE. numpy or torch."""
    lib = torch if isinstance(q, torch.Tensor) else np
    lat = lib.clip(q[..., 0], -POSE_LAT_CLIP, POSE_LAT_CLIP) / 2
    return lib.stack([lat, q[..., 1], q[..., 2], q[..., 3] / POSE_SPEED_SCALE], -1)


def pose_lateral(f):
    """Lateral offset in half-widths from pose features (the inverse of pose_features' scaling)."""
    return f[..., 0] * 2


class MDNRNN(nn.Module):
    """shared=False (the World Models original): every latent dimension picks its own
    mixture component, so one draw can combine 5**z_dim futures, most of them not
    coherent scenes. shared=True: one component is drawn for the whole frame, so a
    sampled future is internally consistent."""

    def __init__(self, z_dim=C.z_dim, a_dim=C.a_dim, h_dim=C.h_dim, k=C.n_gauss,
                 shared: bool = C.shared_mixture, pose_dim: int = 0, pose_input: bool = False,
                 map_dim: int = 0):
        super().__init__()
        assert pose_dim or not pose_input, "pose_input needs a pose head to predict the pose fed back"
        self.z_dim, self.a_dim, self.h_dim, self.k, self.shared = z_dim, a_dim, h_dim, k, shared
        self.pose_dim, self.pose_input = pose_dim, pose_input
        self.map_dim = map_dim
        self.lstm = nn.LSTM(z_dim + a_dim + (pose_dim if pose_input else 0) + map_dim, h_dim, batch_first=True)
        self.head = nn.Linear(h_dim, (k + z_dim * k * 2 if shared else z_dim * k * 3) + 2)
        self.on_head = nn.Linear(h_dim, 1)
        if pose_dim:
            self.pose_head = nn.Linear(h_dim, pose_dim)
        if map_dim:
            self.motion_head = nn.Linear(h_dim, 3)
        self.has_on = False
        self.has_pose = False   # set by load_mdnrnn for a checkpoint with a trained pose head
        self.has_map = False    # set by load_mdnrnn for a checkpoint with a trained motion head

    @property
    def map_input(self) -> bool:
        return self.map_dim > 0

    def forward(self, z, a, state=None, return_on=False, q=None, return_pose=False, patch=None,
                return_motion=False):
        """z: (B,L,z), a: (B,L,3) -> logits, mu, logsig, reward, done logit, state.
        mu and logsig are (B,L,z,k); logits are (B,L,k) if shared else (B,L,z,k).
        q: (B,L,pose_dim) pose features, required when pose_input. return_pose appends the
        predicted pose after the action (B,L,pose_dim) as the last output."""
        x = [z, a]
        if self.pose_input:
            assert q is not None, "this world model reads the pose: pass q"
            x.append(q)
        if self.map_dim:
            assert patch is not None, "this world model reads a road patch: pass patch"
            x.append(patch)
        out, state = self.lstm(torch.cat(x, -1), state)
        p = self.head(out)
        B, L, _ = p.shape
        if self.shared:
            logit = p[..., : self.k]
            mu, logsig = p[..., self.k: -2].view(B, L, self.z_dim, self.k, 2).unbind(-1)
        else:
            logit, mu, logsig = p[..., :-2].view(B, L, self.z_dim, self.k, 3).unbind(-1)
        out6 = logit, mu, logsig.clamp(-7, 2), p[..., -2], p[..., -1], state
        if return_on:
            out6 = (*out6, self.on_head(out)[..., 0])
        if return_pose:
            assert self.pose_dim, "this world model has no pose head"
            out6 = (*out6, self.pose_head(out))
        if return_motion:
            assert self.map_dim, "this world model has no motion head"
            out6 = (*out6, self.motion_head(out))
        return out6

    def step(self, z, a, state=None, return_on=False, q=None, return_pose=False, patch=None,
             return_motion=False):
        """Single step for rollouts: z (B,z), a (B,3), q (B,pose_dim) if pose_input, patch (B,map_dim)
        if map_dim. Extra outputs in this order: on-road logit (return_on), predicted pose
        (return_pose), predicted scaled motion (B,3) (return_motion)."""
        o = self(z[:, None], a[:, None], state, return_on, None if q is None else q[:, None], return_pose,
                 None if patch is None else patch[:, None], return_motion)
        out = (o[0][:, 0], o[1][:, 0], o[2][:, 0], o[3][:, 0], o[4][:, 0], o[5])
        return (*out, *(x[:, 0] for x in o[6:]))

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
