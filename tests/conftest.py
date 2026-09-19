# tests/conftest.py
"""Fixtures compartilhadas entre os testes."""
import pytest
import torch

from seistomo.model.elastic import ElasticModel


@pytest.fixture
def device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


@pytest.fixture
def homogeneous_model(device):
    """Modelo homogêneo 100x100 com Vp=2000, Vs=1000, rho=2000."""
    nz, nx = 100, 100
    vp = torch.ones(nz, nx, device=device) * 2000.0
    vs = torch.ones(nz, nx, device=device) * 1000.0
    rho = torch.ones(nz, nx, device=device) * 2000.0
    return ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)


@pytest.fixture
def two_layer_model(device):
    """Modelo com duas camadas para gerar reflexão."""
    nz, nx = 100, 100
    vp = torch.ones(nz, nx, device=device) * 2000.0
    vs = torch.ones(nz, nx, device=device) * 1000.0
    rho = torch.ones(nz, nx, device=device) * 2000.0
    # Camada inferior mais rápida
    vp[50:, :] = 3000.0
    vs[50:, :] = 1700.0
    rho[50:, :] = 2400.0
    return ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)