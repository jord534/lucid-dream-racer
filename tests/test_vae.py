"""Phase 3 contract: VAE shapes and a finite, differentiable loss."""
import torch

from ldr.config import C
from ldr.vae import VAE, vae_loss


def test_vae_shapes_and_loss():
    vae = VAE()
    x = torch.rand(4, 3, 64, 64)
    recon, mu, logvar = vae(x)
    assert recon.shape == x.shape and mu.shape == (4, C.z_dim)
    loss, rec, kl = vae_loss(recon, x, mu, logvar)
    loss.backward()
    assert torch.isfinite(loss)