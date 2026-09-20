"""Tests for the L2 misfit."""

import pytest
import torch

from seistomo.misfit import L2Misfit


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

def test_zero_when_identical():
    """L2 misfit must be exactly zero when d_syn == d_obs."""
    d = torch.randn(8, 100)
    loss = L2Misfit(normalize=False)(d, d)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_zero_when_identical_normalized():
    """Normalized L2 must also be zero when d_syn == d_obs."""
    d = torch.randn(8, 100)
    loss = L2Misfit(normalize=True)(d, d)
    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_unnormalized_matches_manual():
    """Unnormalized L2 must equal 0.5 * sum((d_syn - d_obs)^2)."""
    d_syn = torch.randn(4, 50)
    d_obs = torch.randn(4, 50)
    expected = 0.5 * ((d_syn - d_obs) ** 2).sum().item()
    loss = L2Misfit(normalize=False)(d_syn, d_obs)
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_normalized_zero_prediction_gives_one():
    """Relative misfit = 1.0 when d_syn is all zeros.

    This is the canonical sanity check for the normalization convention.
    """
    d_obs = torch.randn(8, 100)
    d_syn = torch.zeros_like(d_obs)
    loss = L2Misfit(normalize=True)(d_syn, d_obs)
    assert loss.item() == pytest.approx(1.0, rel=1e-5)


def test_normalization_is_scale_invariant():
    """Scaling both d_syn and d_obs by a constant must not change the loss."""
    d_syn = torch.randn(4, 50)
    d_obs = torch.randn(4, 50)
    misfit = L2Misfit(normalize=True)

    loss_1 = misfit(d_syn, d_obs)
    loss_10 = misfit(10.0 * d_syn, 10.0 * d_obs)

    assert loss_1.item() == pytest.approx(loss_10.item(), rel=1e-5)


def test_unnormalized_scales_quadratically():
    """Unnormalized L2 must scale as s^2 when the residual scales as s."""
    d_syn = torch.randn(4, 50)
    d_obs = torch.zeros_like(d_syn)
    misfit = L2Misfit(normalize=False)

    loss_1 = misfit(d_syn, d_obs)
    loss_3 = misfit(3.0 * d_syn, d_obs)

    assert loss_3.item() == pytest.approx(9.0 * loss_1.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

def test_mask_zeros_contribution():
    """Elements where mask == 0 must not affect the loss."""
    d_syn = torch.randn(4, 50)
    d_obs = torch.zeros_like(d_syn)

    mask = torch.ones_like(d_syn)
    mask[:, :25] = 0.0

    misfit = L2Misfit(normalize=False)
    loss_masked = misfit(d_syn, d_obs, mask=mask)

    # Manually zero out the masked region and recompute
    d_syn_zeroed = d_syn.clone()
    d_syn_zeroed[:, :25] = 0.0
    expected = 0.5 * (d_syn_zeroed ** 2).sum().item()

    assert loss_masked.item() == pytest.approx(expected, rel=1e-5)


def test_mask_normalized_uses_masked_energy():
    """Normalized misfit must divide by the masked observed energy."""
    d_obs = torch.randn(4, 50)
    d_syn = torch.zeros_like(d_obs)

    mask = torch.ones_like(d_obs)
    mask[:, :25] = 0.0

    misfit = L2Misfit(normalize=True)
    loss = misfit(d_syn, d_obs, mask=mask)

    # With d_syn = 0, numerator = masked energy of d_obs, denominator the same.
    # So the result must be 1.0.
    assert loss.item() == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Autodiff
# ---------------------------------------------------------------------------

def test_gradient_flows_to_d_syn():
    """backward() must produce a gradient on d_syn."""
    d_syn = torch.randn(4, 50, requires_grad=True)
    d_obs = torch.randn(4, 50)

    loss = L2Misfit()(d_syn, d_obs)
    loss.backward()

    assert d_syn.grad is not None
    assert d_syn.grad.shape == d_syn.shape
    assert torch.isfinite(d_syn.grad).all()


def test_gradient_is_residual():
    """Analytical gradient of unnormalized L2 w.r.t. d_syn is (d_syn - d_obs)."""
    d_syn = torch.randn(4, 50, requires_grad=True)
    d_obs = torch.randn(4, 50)

    loss = L2Misfit(normalize=False)(d_syn, d_obs)
    loss.backward()

    expected = d_syn.detach() - d_obs
    assert torch.allclose(d_syn.grad, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# API / validation
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    """Different shapes must raise a ValueError."""
    d_syn = torch.randn(4, 50)
    d_obs = torch.randn(4, 51)
    with pytest.raises(ValueError, match="same shape"):
        L2Misfit()(d_syn, d_obs)


def test_invalid_eps_raises():
    """Non-positive eps must raise at construction time."""
    with pytest.raises(ValueError, match="eps must be positive"):
        L2Misfit(eps=0.0)


def test_returns_scalar_tensor():
    """The misfit must return a 0-dim tensor (not a Python float)."""
    d = torch.randn(4, 50)
    loss = L2Misfit()(d, d)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0