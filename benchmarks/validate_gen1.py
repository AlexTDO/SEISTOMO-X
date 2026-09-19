# benchmarks/validate_gen1.py
"""
Script mestre de validação da GEN-1.

Roda toda a suíte de validação:
  1. Forward em modelo homogêneo
  2. Forward em modelo de duas camadas
  3. Crosswell
  4. Gradiente por autodiff
  5. Teste de absorção da CPML
  6. Comparação CPU vs GPU

Gera um relatório no terminal e figuras em ./validation_output/.
"""
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from seistomo.acquisition.source import Ricker
from seistomo.acquisition.survey import CrosswellSurvey, SurfaceSurvey
from seistomo.model.elastic import ElasticModel
from seistomo.physics.elastic.elastic2d import Elastic2D


OUTDIR = "validation_output"
os.makedirs(OUTDIR, exist_ok=True)


def section(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def test_homogeneous(device):
    section("1. Forward em meio homogêneo")
    nz, nx = 150, 300
    vp = torch.ones(nz, nx, device=device) * 2000.0
    vs = torch.ones(nz, nx, device=device) * 1000.0
    rho = torch.ones(nz, nx, device=device) * 2000.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

    survey = SurfaceSurvey(
        source_x=[150], receiver_x=list(range(20, 280, 10)),
        z=5.0, dt=0.001, nt=800,
    )
    solver = Elastic2D(model=model, dt=survey.dt, nt=survey.nt,
                       pml_width=20, fd_order=4, device=device)
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))

    plt.figure(figsize=(10, 6))
    plt.imshow(data[0].cpu().numpy().T, aspect='auto', cmap='seismic',
               extent=[0, len(survey.receivers), survey.nt * survey.dt, 0])
    plt.title("GEN-1 | Homogeneous | Surface shot gather")
    plt.xlabel("Receptor index")
    plt.ylabel("Time (s)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/01_homogeneous.png", dpi=150)
    plt.close()
    print(f"  ✓ Data shape: {data.shape}, finite: {torch.isfinite(data).all().item()}")


def test_two_layer(device):
    section("2. Forward em duas camadas (reflexão)")
    nz, nx = 150, 300
    vp = torch.ones(nz, nx, device=device) * 2000.0
    vs = torch.ones(nz, nx, device=device) * 1000.0
    rho = torch.ones(nz, nx, device=device) * 2000.0
    vp[75:, :] = 3000.0
    vs[75:, :] = 1700.0
    rho[75:, :] = 2400.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

    survey = SurfaceSurvey(
        source_x=[150], receiver_x=list(range(20, 280, 10)),
        z=5.0, dt=0.001, nt=1000,
    )
    solver = Elastic2D(model=model, dt=survey.dt, nt=survey.nt,
                       pml_width=20, fd_order=4, device=device)
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))

    plt.figure(figsize=(10, 6))
    plt.imshow(data[0].cpu().numpy().T, aspect='auto', cmap='seismic',
               extent=[0, len(survey.receivers), survey.nt * survey.dt, 0])
    plt.title("GEN-1 | Two layers | Surface shot gather (with reflection)")
    plt.xlabel("Receptor index")
    plt.ylabel("Time (s)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/02_two_layer.png", dpi=150)
    plt.close()
    print(f"  ✓ Data shape: {data.shape}")


def test_crosswell(device):
    section("3. Crosswell")
    nz, nx = 150, 300
    vp = torch.ones(nz, nx, device=device) * 2500.0
    vs = torch.ones(nz, nx, device=device) * 1400.0
    rho = torch.ones(nz, nx, device=device) * 2200.0
    vp[75:, :] = 3200.0
    vs[75:, :] = 1800.0
    rho[75:, :] = 2500.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=5.0, dz=5.0)

    survey = CrosswellSurvey(
        source_z=list(range(20, 130, 20)),
        receiver_z=list(range(10, 140, 10)),
        source_x=0.0, receiver_x=float(nx - 1),
        dt=0.0005, nt=1500,
    )
    solver = Elastic2D(model=model, dt=survey.dt, nt=survey.nt,
                       pml_width=20, fd_order=4, device=device)
    data = solver.forward(survey=survey, wavelet=Ricker(f0=50))

    plt.figure(figsize=(10, 6))
    shot = len(survey.sources) // 2
    plt.imshow(data[shot].cpu().numpy().T, aspect='auto', cmap='seismic',
               extent=[0, len(survey.receivers), survey.nt * survey.dt, 0])
    plt.title(f"GEN-1 | Crosswell | Shot gather (source #{shot})")
    plt.xlabel("Receiver index (right well)")
    plt.ylabel("Time (s)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/03_crosswell.png", dpi=150)
    plt.close()
    print(f"  ✓ Data shape: {data.shape}")


def test_autodiff(device):
    section("4. Autodiff (gradiente)")
    nz, nx = 100, 200
    vp = torch.ones(nz, nx, device=device) * 2000.0
    vs = torch.ones(nz, nx, device=device) * 1000.0
    rho = torch.ones(nz, nx, device=device) * 2000.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

    survey = SurfaceSurvey(
        source_x=[100], receiver_x=list(range(20, 180, 20)),
        z=5.0, dt=0.001, nt=300,
    )
    solver = Elastic2D(model=model, dt=survey.dt, nt=survey.nt,
                       pml_width=10, fd_order=2, device=device)

    vp_t = model.vp.clone().requires_grad_(True)
    model_t = ElasticModel(vp=vp_t, vs=model.vs, rho=model.rho, dx=10.0, dz=10.0)
    solver_t = Elastic2D(model=model_t, dt=survey.dt, nt=survey.nt,
                         pml_width=10, fd_order=2, device=device)
    data = solver_t.forward(survey=survey, wavelet=Ricker(f0=25))

    loss = data.pow(2).sum()
    loss.backward()

    grad_norm = vp_t.grad.norm().item()
    print(f"  ✓ loss = {loss.item():.4e}")
    print(f"  ✓ |dVp grad| = {grad_norm:.4e}")
    assert grad_norm > 0, "Gradiente nulo — autodiff quebrado"

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(model.vp.cpu().numpy(), cmap='jet', aspect='auto')
    plt.title("Vp (input)")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(vp_t.grad.cpu().numpy(), cmap='seismic', aspect='auto')
    plt.title("dVp gradient (autodiff)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/04_autodiff.png", dpi=150)
    plt.close()


def test_cpml_absorption(device):
    section("5. Absorção da CPML")
    nz, nx = 120, 120
    vp = torch.ones(nz, nx, device=device) * 2000.0
    vs = torch.ones(nz, nx, device=device) * 1000.0
    rho = torch.ones(nz, nx, device=device) * 2000.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

    survey = SurfaceSurvey(
        source_x=[60], receiver_x=[60],
        z=10.0, dt=0.001, nt=1200,
    )
    solver = Elastic2D(model=model, dt=survey.dt, nt=survey.nt,
                       pml_width=20, fd_order=4, device=device)

    solver._init_fields()
    solver.pml.reset_memory()
    w = Ricker(f0=25)(survey.t.to(device))
    src = survey.source_coords.to(device)

    energies = []
    for it in range(survey.nt):
        src_field = solver._inject_source(w[it].expand(1), src)
        solver._step(source_term_vx=src_field, source_term_vz=src_field)
        energies.append((solver.vx.pow(2) + solver.vz.pow(2)).sum().item())

    energies = np.array(energies)
    peak = energies.max()
    final = energies[-1]

    plt.figure(figsize=(10, 5))
    plt.plot(energies, label="Total energy")
    plt.axhline(peak, color='r', linestyle='--', alpha=0.5, label=f"Peak = {peak:.2e}")
    plt.title("GEN-1 | CPML absorption | total kinetic energy")
    plt.xlabel("Time step")
    plt.ylabel("Energy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTDIR}/05_cpml_absorption.png", dpi=150)
    plt.close()

    print(f"  ✓ Peak energy: {peak:.4e}")
    print(f"  ✓ Final energy: {final:.4e}")
    print(f"  ✓ Attenuation ratio: {final / peak:.4%}")
    assert final < peak * 0.5, "CPML não absorveu energia suficiente"


def test_performance(device):
    section(f"6. Performance em {device}")
    nz, nx = 200, 400
    vp = torch.linspace(1500, 4500, nz, device=device).unsqueeze(1).expand(nz, nx).clone()
    vs = vp / 2.0
    rho = 1000.0 + 0.3 * vp
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

    survey = SurfaceSurvey(
        source_x=list(range(20, 380, 40)),
        receiver_x=list(range(10, 390, 10)),
        z=5.0, dt=0.001, nt=500,
    )
    solver = Elastic2D(model=model, dt=survey.dt, nt=survey.nt,
                       pml_width=20, fd_order=4, device=device)

    # Warm-up
    _ = solver.forward(survey=survey, wavelet=Ricker(f0=25))
    if device == 'cuda':
        torch.cuda.synchronize()

    t0 = time.time()
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))
    if device == 'cuda':
        torch.cuda.synchronize()
    dt_forward = time.time() - t0

    print(f"  ✓ Forward: {dt_forward:.3f} s")
    print(f"  ✓ Data shape: {data.shape}")
    print(f"  ✓ Throughput: {data.numel() / dt_forward / 1e6:.2f} M samples/s")


def main():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    for dev in devices:
        print(f"\n{'#' * 60}")
        print(f"#  DEVICE: {dev}")
        print(f"{'#' * 60}")

        test_homogeneous(dev)
        test_two_layer(dev)
        test_crosswell(dev)
        test_autodiff(dev)
        test_cpml_absorption(dev)
        test_performance(dev)

    print("\n" + "=" * 60)
    print("  GEN-1 VALIDATION COMPLETE")
    print(f"  Figures saved to: {OUTDIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()