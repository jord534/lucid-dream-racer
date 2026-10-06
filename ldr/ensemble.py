"""Explicit ensemble helpers for epistemic-uncertainty experiments.

The production world-model path still uses :class:`MDNRNN` directly.  This module
keeps ensemble inference and checkpoints opt-in so existing checkpoints and callers
remain unchanged.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .check_dream import load_mdnrnn
from .config import C
from .mdnrnn import MDNRNN, mdn_mean
from .utils import save_ckpt


@dataclass
class EnsembleStep:
    """Predictions from all models for the same ``(z_t, a_t)`` inputs.

    Tensor fields have a leading member axis. ``states`` contains one recurrent
    state per model. ``next_latent`` is the deterministic mixture expectation,
    so disagreement does not include a fresh aleatoric sample.
    """

    next_latent: torch.Tensor
    logits: torch.Tensor
    means: torch.Tensor
    log_scales: torch.Tensor
    rewards: torch.Tensor
    done_logits: torch.Tensor
    states: list
    on_logits: torch.Tensor | None = None


class MDNRNNEnsemble(nn.Module):
    """An ensemble of independent, identically shaped MDN-RNNs.

    ``members`` must be independently constructed models; parameters are never
    copied or tied by this wrapper. Use :meth:`initialize` for reproducible
    independent initialization, or :meth:`load` for trained members.
    """

    def __init__(self, members: Sequence[MDNRNN], seeds: Sequence[int] | None = None):
        super().__init__()
        if len(members) < 2:
            raise ValueError("an epistemic ensemble requires at least two members")
        shape = (members[0].z_dim, members[0].h_dim, members[0].k, members[0].shared,
                 members[0].a_dim)
        if any((m.z_dim, m.h_dim, m.k, m.shared, m.a_dim) != shape for m in members):
            raise ValueError("all ensemble members must have the same architecture")
        self.members = nn.ModuleList(members)
        self.seeds = list(seeds) if seeds is not None else [None] * len(members)
        if len(self.seeds) != len(members):
            raise ValueError("one seed entry is required per member")

    @classmethod
    def initialize(cls, n: int, seeds: Sequence[int], *, shared: bool = C.shared_mixture,
                   device="cpu") -> MDNRNNEnsemble:
        if n < 2 or len(seeds) != n:
            raise ValueError("provide one seed for each of at least two members")
        members = []
        for seed in seeds:
            # Isolate each member's initialization and do not perturb the caller RNG.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(seed))
                members.append(MDNRNN(shared=shared).to(device))
        return cls(members, seeds).to(device)

    @property
    def n_members(self) -> int:
        return len(self.members)

    @torch.no_grad()
    def step(self, z: torch.Tensor, a: torch.Tensor, states: Sequence | None = None,
             return_on: bool = False) -> EnsembleStep:
        """Propagate identical inputs through every model with member-specific state."""
        if states is None:
            states = [None] * self.n_members
        if len(states) != self.n_members:
            raise ValueError("one recurrent state is required per ensemble member")
        outputs = [m.step(z, a, s, return_on=return_on)
                   for m, s in zip(self.members, states)]
        logits, mus, log_scales, rewards, dones, new_states = zip(*[o[:6] for o in outputs])
        means = torch.stack(mus)
        logits = torch.stack(logits)
        log_scales = torch.stack(log_scales)
        representative = torch.stack([mdn_mean(o[0], o[1]) for o in outputs])
        on = torch.stack([o[6] for o in outputs]) if return_on else None
        return EnsembleStep(representative, logits, means, log_scales,
                            torch.stack(rewards), torch.stack(dones), list(new_states), on)

    @staticmethod
    def disagreement(next_latent: torch.Tensor) -> torch.Tensor:
        """Maximum pairwise Euclidean distance between deterministic member means.

        Input shape is ``(members, batch, z_dim)``; output shape is ``(batch,)``.
        This measures epistemic disagreement only. Mixture spread is an aleatoric
        quantity and is available separately from ``means`` and ``log_scales``.
        """
        if next_latent.ndim != 3 or next_latent.shape[0] < 2:
            raise ValueError("expected (members>=2, batch, latent_dim) predictions")
        return torch.cdist(next_latent.transpose(0, 1),
                           next_latent.transpose(0, 1)).amax(dim=(-1, -2))

    @staticmethod
    def aleatoric_scale(means: torch.Tensor, log_scales: torch.Tensor,
                        logits: torch.Tensor) -> torch.Tensor:
        """Mean predictive standard deviation within each member's MDN mixture.

        This is a compact descriptive scale, not part of epistemic disagreement.
        For a shared mixture, component probabilities are shared across dimensions.
        """
        pi = torch.softmax(logits, dim=-1)
        if logits.ndim == log_scales.ndim - 1:
            pi = pi.unsqueeze(-2)
        mean = (pi * means).sum(-1)
        second_moment = (pi * (log_scales.mul(2).exp() + means.square())).sum(-1)
        var = (second_moment - mean.square()).clamp_min(0)
        return var.sqrt().mean(-1)

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, member in enumerate(self.members):
            name = f"member_{i:02d}.pt"
            save_ckpt(directory / name, model=member.state_dict(), on_trained=member.has_on,
                      seed=self.seeds[i])
            paths.append(name)
        manifest = {"format": 1, "n_members": self.n_members,
                    "seeds": self.seeds, "members": paths, "metadata": metadata or {}}
        (directory / "ensemble.json").write_text(json.dumps(manifest, indent=2) + "\n")

    @classmethod
    def load(cls, directory: str | Path, device="cpu") -> MDNRNNEnsemble:
        directory = Path(directory)
        manifest = json.loads((directory / "ensemble.json").read_text())
        members = [load_mdnrnn(directory / p, device=device) for p in manifest["members"]]
        return cls(members, manifest.get("seeds")).to(device).eval()
