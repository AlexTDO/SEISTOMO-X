"""Full-waveform inversion (FWI) built on the GEN-1 differentiable solver."""

from .fwi import FWI
from .result import FWIResult

__all__ = ["FWI", "FWIResult"]