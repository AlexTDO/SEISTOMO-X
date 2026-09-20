"""L2 (least-squares) misfit for waveform inversion.

The L2 misfit is the workhorse of full-waveform inversion (FWI).
It measures the squared difference between observed and synthetic
seismograms:

    L = 0.5 * || d_syn - d_obs ||^2

Optionally normalized by the energy of the observed data, yielding a
dimensionless relative misfit that is independent of the number of
receivers, number of time samples, and overall amplitude scale:

    L_rel = 0.5 * || d_syn - d_obs ||^2
            ---------------------------
            0.5 * || d_obs ||^2 + eps

With this convention, a synthetic dataset of zero produces L_rel = 1.0,
which serves as a useful sanity check.

References
----------
Tarantola, A. (1984). Inversion of seismic reflection data in the
acoustic approximation. Geophysics, 49(8), 1259-1266.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class L2Misfit(nn.Module):
    """Least-squares waveform misfit.

    Parameters
    ----------
    normalize : bool, default=True
        If True, divide the residual energy by the observed-data energy,
        producing a dimensionless relative misfit. Recommended for FWI
        because it makes the loss magnitude independent of survey size
        and amplitude scale, which stabilizes optimizer step sizes.
    eps : float, default=1e-12
        Small constant added to the denominator to avoid division by zero
        when ``d_obs`` is identically zero (e.g., in synthetic tests).

    Notes
    -----
    Both ``d_syn`` and ``d_obs`` are expected to be tensors of the same
    shape. Common conventions in this codebase:
        - ``(n_receivers, nt)`` for a single source
        - ``(n_sources, n_receivers, nt)`` for a full survey

    The misfit is agnostic to the time axis position. It treats all
    elements as independent samples.

    A ``mask`` can be supplied with the same shape as ``d_obs`` (or
    broadcastable to it). Elements where ``mask == 0`` are excluded from
    the loss. Useful for dead receivers, irregular acquisition geometry,
    or time-windowing.
    """

    def __init__(self, normalize: bool = True, eps: float = 1e-12) -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        d_syn: torch.Tensor,
        d_obs: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the L2 misfit.

        Parameters
        ----------
        d_syn : torch.Tensor
            Synthetic (predicted) data.
        d_obs : torch.Tensor
            Observed (target) data. Must have the same shape as ``d_syn``.
        mask : torch.Tensor, optional
            Binary mask (0/1) with shape broadcastable to ``d_obs``.
            Elements where the mask is 0 are excluded from the loss.

        Returns
        -------
        loss : torch.Tensor
            Scalar tensor (0-dim). Differentiable w.r.t. ``d_syn``.
        """
        if d_syn.shape != d_obs.shape:
            raise ValueError(
                f"d_syn and d_obs must have the same shape, "
                f"got {tuple(d_syn.shape)} vs {tuple(d_obs.shape)}"
            )

        residual = d_syn - d_obs

        if mask is not None:
            residual = residual * mask

        numerator = 0.5 * (residual ** 2).sum()

        if not self.normalize:
            return numerator

        if mask is not None:
            denom_data = d_obs * mask
        else:
            denom_data = d_obs

        denominator = 0.5 * (denom_data ** 2).sum() + self.eps
        return numerator / denominator

    def extra_repr(self) -> str:
        return f"normalize={self.normalize}, eps={self.eps:.1e}"