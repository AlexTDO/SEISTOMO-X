# tests/test_fd_operators.py
"""Testes para os operadores de diferenças finitas."""
import pytest
import torch

from seistomo.physics.fd.operators import FiniteDifferenceOperators


@pytest.mark.parametrize("order", [2, 4, 8])
def test_d_dx_linear_field(order, device):
    """A derivada de um campo linear deve ser constante."""
    fd = FiniteDifferenceOperators(dx=1.0, dz=1.0, order=order, device=device)
    nx = 50
    x = torch.arange(nx, dtype=torch.float32, device=device)
    field = x.unsqueeze(0).expand(10, nx)  # (10, 50)

    d = fd.d_dx(field)

    # No interior (sem efeitos de borda), d/dx deve ser 1.0
    interior = d[2:-2, order:-order]
    assert torch.allclose(interior, torch.ones_like(interior), atol=1e-4)


@pytest.mark.parametrize("order", [2, 4])
def test_laplacian_of_quadratic(order, device):
    """O Laplaciano de x² é 2 (constante)."""
    fd = FiniteDifferenceOperators(dx=1.0, dz=1.0, order=order, device=device)
    nx = 50
    x = torch.arange(nx, dtype=torch.float32, device=device)
    field = (x ** 2).unsqueeze(0).expand(10, nx)

    lap = fd.laplacian(field)
    interior = lap[2:-2, 2:-2]
    assert torch.allclose(interior, 2.0 * torch.ones_like(interior), atol=1e-3)


def test_d_dz_linear_field(device):
    """Mesmo teste, mas na direção z."""
    fd = FiniteDifferenceOperators(dx=1.0, dz=1.0, order=4, device=device)
    nz = 50
    z = torch.arange(nz, dtype=torch.float32, device=device)
    field = z.unsqueeze(1).expand(nz, 10)

    d = fd.d_dz(field)
    interior = d[2:-2, 2:-2]
    assert torch.allclose(interior, torch.ones_like(interior), atol=1e-4)