# tests/test_pml.py
"""Testes para a CPML."""
import torch

from seistomo.physics.boundary.pml import CPML


def test_pml_zero_in_interior(device):
    """Fora da região da PML, o coeficiente a deve ser zero."""
    pml = CPML(model_shape=(100, 100), dx=1.0, dz=1.0, width=20,
               vp_max=3000.0, order=4, device=device)
    pml.update_coefficients(dt=0.001)

    # No centro, sigma = 0 → a = 0
    center = 50
    assert torch.allclose(pml.a_x[center], torch.tensor(0.0, device=device))
    assert torch.allclose(pml.a_z[center], torch.tensor(0.0, device=device))


def test_pml_positive_in_boundary(device):
    """Dentro da PML, o coeficiente sigma deve ser positivo."""
    pml = CPML(model_shape=(100, 100), dx=1.0, dz=1.0, width=20,
               vp_max=3000.0, order=4, device=device)

    assert pml.sigma_x[0] > 0
    assert pml.sigma_x[99] > 0
    assert pml.sigma_z[0] > 0
    assert pml.sigma_z[99] > 0


def test_pml_memory_resets(device):
    """reset_memory deve zerar todas as variáveis de memória."""
    pml = CPML(model_shape=(50, 50), dx=1.0, dz=1.0, width=10,
               vp_max=3000.0, order=4, device=device)
    pml.update_coefficients(dt=0.001)

    # Enche a memória
    dummy = torch.ones(50, 50, device=device)
    for key in pml.memory:
        pml.memory[key] = dummy.clone()

    pml.reset_memory()

    for key in pml.memory:
        assert torch.allclose(pml.memory[key], torch.zeros_like(dummy))