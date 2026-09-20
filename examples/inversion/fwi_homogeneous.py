"""FWI on a homogeneous background with a Gaussian anomaly.

Runs end-to-end on CPU in ~1-2 minutes. Prints loss per iteration and
saves a PNG comparing the true, initial, and inverted Vp models.

Run with:
    python examples/inversion/fwi_homogeneous.py
"""

import os
import torch
import matplotlib.pyplot as plt

from seistomo.acquisition import SurfaceSurvey, Ricker
from seistomo.inversion import FWI
from seistomo.misfit import L2Misfit
from seistomo.model import ElasticModel
from seistomo.physics.elastic import Elastic2D


def main():
    # --- Setup ------------------------------------------------------
    nz, nx = 60, 60
    dx = dz = 10.0
    nt = 400
    dt = 0.001
    pml_width = 10

    # True model: 2000 m/s background with a +250 m/s Gaussian blob
    zz, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    cx, cz, sigma = nx / 2, nz / 2, 6.0
    anomaly = 250.0 * torch.exp(
        -((xx - cx) ** 2 + (zz - cz) ** 2) / (2 * sigma ** 2)
    )
    vp_true = 2000.0 + anomaly
    vs_true = vp_true / 1.732
    rho_true = 1000.0 + 0.3 * vp_true

    # Initial model: smooth background only (no anomaly)
    vp_init = torch.full((nz, nx), 2000.0)

    # Survey
    source_x = list(torch.linspace(5, nx - 6, 5).tolist())
    receiver_x = list(torch.linspace(2, nx - 3, 30).tolist())
    survey = SurfaceSurvey(
        source_x=source_x, receiver_x=receiver_x,
        z=2.0, dt=dt, nt=nt,
    )
    wavelet = Ricker(f0=12.0)

    print("Generating observed data from the true model...")
    model_true = ElasticModel(vp_true, vs_true, rho_true, dx, dz)
    solver_true = Elastic2D(
        model_true, dt=dt, nt=nt, pml_width=pml_width,
        fd_order=8, device="cpu",
    )
    with torch.no_grad():
        d_obs = solver_true.forward(survey, wavelet)
    print(f"  d_obs shape: {tuple(d_obs.shape)}")

    # --- FWI --------------------------------------------------------
    print("\nStarting FWI...")
    model_init = ElasticModel(vp_init, vs_true, rho_true, dx, dz)
    solver_fwi = Elastic2D(
        model_init, dt=dt, nt=nt, pml_width=pml_width,
        fd_order=8, device="cpu",
    )
    fwi = FWI(
        solver=solver_fwi,
        misfit=L2Misfit(),
        optimizer="lbfgs",
        lr=1.0,
        max_iter=15,
        grad_clip=None,
        store_history=True,
        verbose=True,
        log_every=1,
    )
    result = fwi.run(
        survey=survey,
        wavelet=wavelet,
        data_observed=d_obs,
        vp_initial=vp_init,
        vs_fixed=vs_true,
        rho_fixed=rho_true,
    )

    print(f"\n{result}")
    vp_err = (result.vp - vp_true).abs().mean().item()
    print(f"Mean absolute Vp error (final): {vp_err:.2f} m/s")

    # --- Plot -------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmin, vmax = vp_true.min().item(), vp_true.max().item()

    for ax, data, title in zip(
        axes,
        [vp_true, vp_init, result.vp],
        ["True Vp", "Initial Vp", f"Inverted Vp ({result.n_iter} iters)"],
    ):
        im = ax.imshow(data.numpy(), cmap="jet", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "fwi_homogeneous_result.png")
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved comparison figure to: {out_path}")

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(result.loss_history, marker="o")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("L2 misfit (normalized)")
    ax.set_title("FWI convergence")
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "fwi_homogeneous_loss.png")
    plt.savefig(loss_path, dpi=120)
    print(f"Saved loss curve to: {loss_path}")


if __name__ == "__main__":
    main()