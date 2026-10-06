"""Contracts for the opt-in epistemic ensemble implementation."""
import torch

from ldr.config import C
from ldr.ensemble import MDNRNNEnsemble
from ldr.mdnrnn import MDNRNN, mdn_mean


def test_ensemble_construction_independence_and_reproducibility():
    a = MDNRNNEnsemble.initialize(3, [101, 102, 103])
    b = MDNRNNEnsemble.initialize(3, [101, 102, 103])
    assert a.n_members == 3
    assert all(not torch.equal(a.members[0].state_dict()[k], a.members[1].state_dict()[k])
               for k in a.members[0].state_dict())
    assert all(torch.equal(a.members[i].state_dict()[k], b.members[i].state_dict()[k])
               for i in range(3) for k in a.members[i].state_dict())
    assert a.members[0].lstm.weight_ih_l0.data_ptr() != a.members[1].lstm.weight_ih_l0.data_ptr()


def test_identical_input_propagation_shapes_and_mixture_mean():
    ensemble = MDNRNNEnsemble.initialize(2, [8, 9]).eval()
    z, action = torch.randn(4, C.z_dim), torch.rand(4, C.a_dim)
    pred = ensemble.step(z, action)
    assert pred.next_latent.shape == (2, 4, C.z_dim)
    assert pred.rewards.shape == (2, 4)
    assert pred.means.shape == (2, 4, C.z_dim, C.n_gauss)
    assert torch.allclose(pred.next_latent[0], mdn_mean(pred.logits[0], pred.means[0]))
    for i, member in enumerate(ensemble.members):
        out = member.step(z, action)
        assert torch.allclose(pred.next_latent[i], mdn_mean(out[0], out[1]))
    # Representative predictions are deterministic; no independent MDN samples are mixed in.
    again = ensemble.step(z, action)
    assert torch.equal(pred.next_latent, again.next_latent)


def test_disagreement_calculation_and_probabilistic_scale():
    means = torch.tensor([[[0.0, 0.0]], [[3.0, 4.0]], [[0.0, 2.0]]])
    assert torch.allclose(MDNRNNEnsemble.disagreement(means), torch.tensor([5.0]))
    ensemble = MDNRNNEnsemble.initialize(2, [18, 19])
    pred = ensemble.step(torch.zeros(2, C.z_dim), torch.zeros(2, C.a_dim))
    scale = ensemble.aleatoric_scale(pred.means, pred.log_scales, pred.logits)
    assert scale.shape == (2, 2)
    assert torch.isfinite(scale).all()
    shared = MDNRNNEnsemble.initialize(2, [28, 29], shared=True)
    shared_pred = shared.step(torch.zeros(2, C.z_dim), torch.zeros(2, C.a_dim))
    assert shared_pred.logits.shape == (2, 2, C.n_gauss)
    assert shared_pred.means.shape == (2, 2, C.z_dim, C.n_gauss)
    assert shared.aleatoric_scale(shared_pred.means, shared_pred.log_scales,
                                  shared_pred.logits).shape == (2, 2)


def test_checkpoint_roundtrip_and_single_model_api_compatibility(tmp_path):
    ensemble = MDNRNNEnsemble.initialize(2, [31, 32])
    ensemble.save(tmp_path, {"purpose": "test"})
    restored = MDNRNNEnsemble.load(tmp_path)
    assert restored.n_members == 2
    assert restored.seeds == [31, 32]
    for i in range(2):
        for key, value in ensemble.members[i].state_dict().items():
            assert torch.equal(value, restored.members[i].state_dict()[key])
    # Existing single-MDN-RNN construction and six-value forward API remain intact.
    single = MDNRNN()
    out = single(torch.zeros(1, 2, C.z_dim), torch.zeros(1, 2, C.a_dim))
    assert len(out) == 6 and out[1].shape == (1, 2, C.z_dim, C.n_gauss)
