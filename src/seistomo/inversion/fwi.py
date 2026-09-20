"""Full-waveform inversion (FWI) loop.

This module implements the classic FWI workflow:

    for iter in range(max_iter):
        d_syn = forward(model)
        loss  = misfit(d_syn, d_obs)
        loss.backward()
        optimizer.step()

built on top of the GEN-1 differentiable elastic solver. The same
computational graph used for forward modeling is traversed in reverse
to obtain the gradient of the misfit w.r.t. Vp.

Design notes
------------
- Only Vp is inverted in GEN-2. Vs and rho are held fixed. This is a
  deliberate simplification: multi-parameter FWI suffers from Vp/Vs
  trade-offs that are better studied one parameter at a time.
- The solver holds precomputed coefficients (lambda_, mu, b) derived
  from the model. Those must be refreshed at every outer iteration,
  which is why the loop calls ``solver.update_model(model)`` inside the
  closure. Forgetting this would freeze the gradient at iteration 0.
- The optimizer's closure is called one or more times per outer step
  (L-BFGS does a line search, Adam/SGD call it once). Putting
  ``update_model`` inside the closure guarantees the solver is always
  consistent with the current ``vp``, no matter how many times the
  closure runs.
"""

from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn as nn

from ..model.elastic import ElasticModel
from .result import FWIResult


_OPTIMIZERS = {
    "sgd": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "lbfgs": torch.optim.LBFGS,
}


class FWI:
    """Full-waveform inversion driver.

    Parameters
    ----------
    solver : Elastic2D
        A differentiable elastic solver instance. Must have an
        ``update_model`` method (added in GEN-2).
    misfit : nn.Module
        Misfit function. Must accept ``(d_syn, d_obs)`` and return a
        scalar tensor (e.g., ``L2Misfit()``).
    optimizer : str, default="lbfgs"
        One of ``"sgd"``, ``"adam"``, ``"lbfgs"``.
    lr : float, default=1.0
        Learning rate. For L-BFGS, PyTorch uses ``lr`` as the initial
        step size in the line search; ``1.0`` is a reasonable default.
        For SGD/Adam, tune this: typical values are 1e-3 to 1e-1
        depending on data scale.
    max_iter : int, default=50
        Maximum number of outer iterations (optimizer steps).
    grad_clip : float or None, default=None
        If set, clip the gradient norm to this value before each step.
        Recommended for stability, especially when Vp has sharp
        contrasts.
    store_history : bool, default=True
        If True, store a snapshot of Vp at each iteration. Enables
        inversion-evolution videos but costs memory.
    verbose : bool, default=True
        If True, print loss every ``log_every`` iterations.
    log_every : int, default=5
        Logging frequency (outer iterations).

    Notes
    -----
    The ``solver``, ``survey``, and ``wavelet`` must be mutually
    consistent: ``solver.nt`` should match ``survey.nt``, and
    ``solver.dt`` should match ``survey.dt``. The FWI class does not
    re-check this (the solver does not store them in a comparable way),
    but a mismatch will produce silently wrong results.
    """

    def __init__(
        self,
        solver,
        misfit: nn.Module,
        optimizer: str = "lbfgs",
        lr: float = 1.0,
        max_iter: int = 50,
        grad_clip: float | None = None,
        store_history: bool = True,
        verbose: bool = True,
        log_every: int = 5,
    ) -> None:
        if optimizer not in _OPTIMIZERS:
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. "
                f"Choose one of {sorted(_OPTIMIZERS)}."
            )
        if max_iter <= 0:
            raise ValueError(f"max_iter must be positive, got {max_iter}")
        if log_every <= 0:
            raise ValueError(f"log_every must be positive, got {log_every}")

        self.solver = solver
        self.misfit = misfit
        self.optimizer_name = optimizer
        self.lr = lr
        self.max_iter = max_iter
        self.grad_clip = grad_clip
        self.store_history = store_history
        self.verbose = verbose
        self.log_every = log_every

    def run(
        self,
        survey,
        wavelet,
        data_observed: torch.Tensor,
        vp_initial: torch.Tensor,
        vs_fixed: torch.Tensor,
        rho_fixed: torch.Tensor,
        dx: float | None = None,
        dz: float | None = None,
    ) -> FWIResult:
        """Run FWI.

        Parameters
        ----------
        survey : Survey
            Acquisition geometry. Source/receiver coordinates must be in
            the physical-model system (z=0 = surface).
        wavelet : Ricker or callable
            Source wavelet. ``wavelet(t)`` must return a 1D tensor of
            length ``solver.nt``.
        data_observed : torch.Tensor
            Observed seismograms, shape ``(n_sources, n_receivers, nt)``.
            Typically produced by running the solver on the true model.
        vp_initial : torch.Tensor
            Starting Vp model, shape ``(nz, nx)``. Usually a smoothed or
            depth-averaged version of the true model.
        vs_fixed : torch.Tensor
            Fixed Vs model, shape ``(nz, nx)``. Same shape as ``vp_initial``.
        rho_fixed : torch.Tensor
            Fixed density model, shape ``(nz, nx)``.
        dx, dz : float, optional
            Grid spacing. If None, falls back to ``solver.model.dx`` /
            ``solver.model.dz``.

        Returns
        -------
        FWIResult
            Contains the final Vp, loss history, and metadata.
        """
        # ----- Setup ---------------------------------------------------
        if dx is None:
            dx = self.solver.model.dx
        if dz is None:
            dz = self.solver.model.dz

        device = self.solver.device

        # The Vp tensor becomes a leaf in the autodiff graph.
        vp = nn.Parameter(vp_initial.clone().detach().to(device).float())

        # vs and rho are constants (no gradient).
        vs_fixed = vs_fixed.to(device).float()
        rho_fixed = rho_fixed.to(device).float()
        data_observed = data_observed.to(device).float()

        # Working model: holds a reference to the SAME vp Parameter,
        # so optimizer.step() updates it in place and the model sees it.
        model = ElasticModel(
            vp=vp, vs=vs_fixed, rho=rho_fixed, dx=dx, dz=dz
        )

        # Sanity check: shapes
        if data_observed.dim() != 3:
            raise ValueError(
                f"data_observed must be 3D (n_sources, n_receivers, nt), "
                f"got shape {tuple(data_observed.shape)}"
            )

        # ----- Optimizer ----------------------------------------------
        opt_cls = _OPTIMIZERS[self.optimizer_name]
        if self.optimizer_name == "lbfgs":
            optimizer = opt_cls(
                [vp],
                lr=self.lr,
                max_iter=20,        # inner line-search iterations
                history_size=10,
                line_search_fn="strong_wolfe",
            )
        else:
            optimizer = opt_cls([vp], lr=self.lr)

        # ----- Loop ---------------------------------------------------
        loss_history: list[float] = []
        vp_history: list[torch.Tensor] = []

        def closure() -> torch.Tensor:
            """One forward + backward pass. Called >=1 time per outer step."""
            optimizer.zero_grad(set_to_none=True)

            # Critical: refresh the solver's coefficients from the
            # current vp. Without this, the solver keeps using the
            # coefficients computed at iteration 0.
            self.solver.update_model(model)

            d_syn = self.solver.forward(survey, wavelet)
            loss = self.misfit(d_syn, data_observed)
            loss.backward()

            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([vp], self.grad_clip)

            return loss

        t0 = time.perf_counter()

        for it in range(self.max_iter):
            loss = optimizer.step(closure)
            loss_value = float(loss.detach())

            loss_history.append(loss_value)
            if self.store_history:
                vp_history.append(vp.detach().clone())

            if self.verbose and (it % self.log_every == 0 or it == self.max_iter - 1):
                print(
                    f"[FWI] iter {it:4d}/{self.max_iter}  "
                    f"loss = {loss_value:.6e}"
                )

        wall_time = time.perf_counter() - t0

        return FWIResult(
            vp=vp.detach().clone(),
            loss_history=loss_history,
            vp_history=vp_history,
            optimizer_name=self.optimizer_name,
            n_iter=len(loss_history),
            wall_time_s=wall_time,
            metadata={
                "lr": self.lr,
                "grad_clip": self.grad_clip,
                "max_iter": self.max_iter,
            },
        )