"""Phase 4 contract: MDN-RNN output shapes, finite NLL, sampling."""
import torch

from ldr.config import C
from ldr.mdnrnn import MDNRNN, mdn_nll, mdn_sample


def test_mdnrnn():
    m = MDNRNN()
    z, a = torch.randn(2, 10, C.z_dim), torch.rand(2, 10, 3)
    logit, mu, ls, r, d, _ = m(z, a)
    assert logit.shape == (2, 10, C.z_dim, C.n_gauss) and r.shape == (2, 10)
    assert torch.isfinite(mdn_nll(logit, mu, ls, torch.randn(2, 10, C.z_dim)))
    assert mdn_sample(logit, mu, ls, 1.15).shape == (2, 10, C.z_dim)