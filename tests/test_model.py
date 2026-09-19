# tests/test_model.py
"""Testes do ElasticModel."""
import pytest
import torch

from seistomo.model.elastic import ElasticModel


def test_model_shapes(device):
    vp = torch.ones(50, 60, device=device)
    model = ElasticModel(vp=vp, vs=vp, rho=vp, dx=1.0, dz=1.0)
    assert model.shape == (50, 60)
    assert model.nz == 50
    assert model.nx == 60


def test_model_rejects_mismatched_shapes(device):
    vp = torch.ones(50, 60, device=device)
    vs = torch.ones(50, 61, device=device)
    rho = torch.ones(50, 60, device=device)
    with pytest.raises(ValueError):
        ElasticModel(vp=vp, vs=vs, rho=rho)


def test_model_dtype(device):
    vp = torch.ones(10, 10, dtype=torch.float64, device=device)
    model = ElasticModel(vp=vp, vs=vp, rho=vp)
    assert model.vp.dtype == torch.float32