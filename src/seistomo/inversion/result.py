"""Container for FWI results.

Keeping results in a dedicated dataclass keeps the FWI loop lean and gives
downstream code (visualization, analysis, tests) a stable contract to
depend on. Fields may be added here without touching the FWI class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class FWIResult:
    """Results of an FWI run.

    Attributes
    ----------
    vp : torch.Tensor
        The final inverted Vp model, shape ``(nz, nx)``. Detached from the
        autodiff graph (safe to store, mutate, or use in plots).
    loss_history : list[float]
        Value of the misfit at each outer iteration. Length equals the
        number of iterations actually performed (which may be less than
        ``max_iter`` if early stopping triggered).
    vp_history : list[torch.Tensor]
        Snapshot of ``vp`` at each outer iteration (detached, cloned).
        Useful for making inversion-evolution videos. Can be large; set
        ``FWI(store_history=False)`` to skip.
    optimizer_name : str
        Which optimizer was used ("sgd", "adam", or "lbfgs").
    n_iter : int
        Number of outer iterations performed.
    wall_time_s : float
        Total wall-clock time of the run, in seconds.
    metadata : dict
        Free-form dict for experiment-specific bookkeeping (e.g., the
        learning rate, the true model's misfit at iteration 0, etc.).
    """

    vp: torch.Tensor
    loss_history: list[float] = field(default_factory=list)
    vp_history: list[torch.Tensor] = field(default_factory=list)
    optimizer_name: str = ""
    n_iter: int = 0
    wall_time_s: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        """Misfit value at the last iteration (or NaN if no iterations ran)."""
        return self.loss_history[-1] if self.loss_history else float("nan")

    @property
    def initial_loss(self) -> float:
        """Misfit value at the first iteration (or NaN if no iterations ran)."""
        return self.loss_history[0] if self.loss_history else float("nan")

    def __repr__(self) -> str:
        return (
            f"FWIResult(optimizer={self.optimizer_name!r}, "
            f"n_iter={self.n_iter}, "
            f"initial_loss={self.initial_loss:.6f}, "
            f"final_loss={self.final_loss:.6f}, "
            f"wall_time_s={self.wall_time_s:.1f})"
        )