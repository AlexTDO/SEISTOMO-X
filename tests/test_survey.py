# tests/test_survey.py
"""Testes das classes de aquisição."""
import torch

from seistomo.acquisition.source import Ricker
from seistomo.acquisition.survey import CrosswellSurvey, SurfaceSurvey


def test_surface_survey_coords():
    survey = SurfaceSurvey(
        source_x=[10, 20], receiver_x=[30, 40, 50], z=0.0,
        dt=0.001, nt=100,
    )
    assert survey.source_coords.shape == (2, 2)
    assert survey.receiver_coords.shape == (3, 2)
    assert torch.allclose(survey.source_coords[:, 1], torch.zeros(2))


def test_crosswell_survey_coords():
    survey = CrosswellSurvey(
        source_z=[10, 20], receiver_z=[30, 40], source_x=0.0, receiver_x=99.0,
        dt=0.001, nt=100,
    )
    assert survey.source_coords.shape == (2, 2)
    assert torch.allclose(survey.source_coords[:, 0], torch.zeros(2))
    assert torch.allclose(survey.receiver_coords[:, 0], torch.full((2,), 99.0))


def test_ricker_zero_mean():
    """A wavelet de Ricker deve ter integral aproximadamente zero."""
    ricker = Ricker(f0=25)
    t = torch.linspace(-0.2, 0.2, 1000)
    w = ricker(t)
    # A integral deve ser próxima de zero (a Ricker tem média zero por construção)
    integral = torch.trapz(w, t)
    assert abs(integral.item()) < 1e-4