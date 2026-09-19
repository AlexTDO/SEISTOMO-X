# tests/test_elastic2d.py
"""Testes do solver elástico 2D."""
import pytest
import torch

from seistomo.acquisition.source import Ricker
from seistomo.acquisition.survey import CrosswellSurvey, SurfaceSurvey
from seistomo.physics.elastic.elastic2d import Elastic2D


def _small_survey(device):
    return SurfaceSurvey(
        source_x=[50],
        receiver_x=[30, 40, 50, 60, 70],
        z=0.0,
        dt=0.001,
        nt=100,
    )


def test_forward_runs_and_shape(homogeneous_model, device):
    survey = _small_survey(device)
    solver = Elastic2D(
        model=homogeneous_model, dt=survey.dt, nt=survey.nt,
        pml_width=10, fd_order=4, device=device,
    )
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))

    assert data.shape == (1, 5, 100)
    assert torch.isfinite(data).all()


def test_forward_differentiable(homogeneous_model, device):
    """O gradiente deve fluir até Vp, Vs e rho."""
    vp = homogeneous_model.vp.clone().requires_grad_(True)
    vs = homogeneous_model.vs.clone().requires_grad_(True)
    rho = homogeneous_model.rho.clone().requires_grad_(True)

    from seistomo.model.elastic import ElasticModel
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

    survey = _small_survey(device)
    solver = Elastic2D(model=model, dt=survey.dt, nt=survey.nt,
                       pml_width=10, fd_order=2, device=device)
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))

    loss = data.pow(2).sum()
    loss.backward()

    assert vp.grad is not None and torch.isfinite(vp.grad).all()
    assert vs.grad is not None and torch.isfinite(vs.grad).all()
    assert rho.grad is not None and torch.isfinite(rho.grad).all()
    assert vp.grad.abs().sum() > 0


def test_crosswell_uses_same_solver(homogeneous_model, device):
    """O mesmo Elastic2D deve funcionar para Crosswell."""
    survey = CrosswellSurvey(
        source_z=[20, 40],
        receiver_z=[20, 40, 60],
        source_x=0.0,
        receiver_x=99.0,
        dt=0.001,
        nt=100,
    )
    solver = Elastic2D(model=homogeneous_model, dt=survey.dt, nt=survey.nt,
                       pml_width=10, fd_order=4, device=device)
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))

    assert data.shape == (2, 3, 100)
    assert torch.isfinite(data).all()


def test_energy_decays_with_cpml(two_layer_model, device):
    """
    Com CPML, a energia total do campo deve decair após a onda sair do domínio.
    Este é um teste de sanidade da absorção de borda.
    """
    survey = SurfaceSurvey(
        source_x=[50],
        receiver_x=[50],
        z=10.0,
        dt=0.001,
        nt=800,
    )
    solver = Elastic2D(model=two_layer_model, dt=survey.dt, nt=survey.nt,
                       pml_width=20, fd_order=4, device=device)

    # Roda alguns passos manualmente e mede a energia
    solver._init_fields()
    solver.pml.reset_memory()
    w = Ricker(f0=25)(survey.t.to(device))

    energies = []
    src = survey.source_coords.to(device)
    for it in range(survey.nt):
        src_field = solver._inject_source(w[it].expand(1), src)
        solver._step(source_term_vx=src_field, source_term_vz=src_field)
        if it % 50 == 0:
            energy = (solver.vx.pow(2) + solver.vz.pow(2)).sum()
            energies.append(energy.item())

    # A energia no final deve ser menor que o pico
    peak = max(energies)
    final = energies[-1]
    assert final < peak * 0.5, f"CPML não absorveu: peak={peak}, final={final}"