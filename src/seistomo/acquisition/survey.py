# src/seistomo/acquisition/survey.py
import torch
from .source import Ricker


class Survey:
    """
    Classe base para geometrias de aquisição.
    """
    def __init__(self, sources, receivers, wavelet=None, dt=0.001, nt=1000):
        self.sources = sources  # Lista de tuplas (x, z)
        self.receivers = receivers  # Lista de tuplas (x, z)
        self.wavelet = wavelet if wavelet is not None else Ricker(f0=100)
        self.dt = dt
        self.nt = nt

        # Converte para tensores para uso no solver
        self.source_coords = torch.tensor(sources, dtype=torch.float32)
        self.receiver_coords = torch.tensor(receivers, dtype=torch.float32)

        self.t = torch.arange(nt, dtype=torch.float32) * dt


class SurfaceSurvey(Survey):
    """
    Geometria de superfície: fontes e receptores na superfície (z=0).
    """
    def __init__(self, source_x, receiver_x, z=0.0, **kwargs):
        sources = [(x, z) for x in source_x]
        receivers = [(x, z) for x in receiver_x]
        super().__init__(sources, receivers, **kwargs)


class CrosswellSurvey(Survey):
    """
    Geometria crosswell: fontes em um poço, receptores em outro.
    """
    def __init__(self, source_z, receiver_z, source_x=0.0, receiver_x=100.0, **kwargs):
        sources = [(source_x, z) for z in source_z]
        receivers = [(receiver_x, z) for z in receiver_z]
        super().__init__(sources, receivers, **kwargs)