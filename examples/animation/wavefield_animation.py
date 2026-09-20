# examples/animation/wavefield_animation.py
"""
Animação: wavefield se propagando no modelo Marmousi Small (real).

Padrão da indústria:
  - Fonte explosiva (injetada em pressão, gera predominantemente onda P)
  - Filtro passa-baixa no gather (remove alta frequência espúria)
  - AGC suave para revelar reflexões fracas
"""
import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from seistomo.data.marmousi_loader import load_marmousi_small
from seistomo.model.elastic import ElasticModel
from seistomo.acquisition.source import Ricker
from seistomo.acquisition.survey import Survey
from seistomo.physics.elastic.elastic2d import Elastic2D
from seistomo.visualization.wavefield import plot_wavefield_and_gather


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")

    print("Carregando Marmousi Small...")
    props = load_marmousi_small(dx=20.0, dz=20.0, device=device)
    nz, nx = props['vp'].shape
    dx = props['dx']
    dz = props['dz']
    print(f"  Modelo: {nz}x{nx}, dx={dx}m, dz={dz}m")
    print(f"  Extensão física: {nx*dx/1000:.2f} km x {nz*dz/1000:.2f} km")

    model = ElasticModel(
        vp=props['vp'], vs=props['vs'], rho=props['rho'],
        dx=dx, dz=dz,
    )

    vp_max = props['vp'].max().item()
    dt_safe = dx / (vp_max * np.sqrt(2))
    dt = min(0.001, dt_safe * 0.8)

    pml_width = 40
    nt = 3000
    n_receivers = 80
    source_x = nx // 2
    source_z = 2
    receiver_z = 2
    frame_every = 15
    fps = 30

    print(f"Aquisição: dt={dt:.5f}s, nt={nt}, n_receivers={n_receivers}")
    print(f"Fonte explosiva (pressão) em (x={source_x}, z={source_z})")
    print(f"Receptores em z={receiver_z} (superfície do modelo físico)")

    receiver_x = np.linspace(5, nx - 5, n_receivers).astype(int)
    receivers = [(int(rx), receiver_z) for rx in receiver_x]
    sources = [(source_x, source_z)]

    survey = Survey(sources=sources, receivers=receivers, dt=dt, nt=nt)

    print("Inicializando solver (grid estendido)...")
    solver = Elastic2D(
        model=model, dt=dt, nt=nt,
        pml_width=pml_width, fd_order=8, device=device,
    )
    print(f"  Grid físico:    {nz} x {nx}")
    print(f"  Grid estendido: {solver.nz_ext} x {solver.nx_ext} (PML={pml_width})")

    wavelet = Ricker(f0=8)
    t = survey.t.to(device)
    w = wavelet(t)

    source_coords = survey.source_coords.to(device)
    receiver_coords = survey.receiver_coords.to(device)
    src_coord = source_coords[0:1]

    gather = torch.zeros((nt, n_receivers), device=device)

    # --- Warm-up para escala do wavefield ---
    print("Estimando escala do wavefield...")
    solver._init_fields()
    solver.pml.reset_memory()
    wf_max_estimate = 0.0
    with torch.no_grad():
        for it in range(150):
            source_amplitude = w[it].expand(1)
            source_field = solver._inject_source(source_amplitude, src_coord)
            # FONTE EXPLOSIVA: injeta em pressão (τxx, τzz), não em velocidade
            solver._step()
            solver.tau_xx = solver.tau_xx + source_field
            solver.tau_zz = solver.tau_zz + source_field
            wf_phys = solver.get_wavefield_physical()
            wf_max_estimate = max(wf_max_estimate, wf_phys.abs().max().item())
    wavefield_scale = wf_max_estimate
    print(f"  wavefield_scale = {wavefield_scale:.4e}")

    # --- Figura ---
    fig, (ax_wf, ax_gather) = plt.subplots(2, 1, figsize=(10, 9))
    plt.subplots_adjust(hspace=0.28, left=0.08, right=0.98, top=0.95, bottom=0.07)

    vp_np = model.vp.cpu().numpy()

    solver._init_fields()
    solver.pml.reset_memory()

    outdir = os.path.dirname(os.path.abspath(__file__))
    outpath = os.path.join(outdir, "wavefield_animation.mp4")
    print(f"Salvando em: {outpath}")

    try:
        writer = FFMpegWriter(fps=fps, bitrate=5000)
    except Exception as e:
        print(f"ERRO: ffmpeg não encontrado ({e}).")
        return

    frame_count = 0
    t0 = time.time()

    print("Iniciando animação...")
    with writer.saving(fig, outpath, dpi=120):
        with torch.no_grad():
            for it in range(nt):
                source_amplitude = w[it].expand(1)
                source_field = solver._inject_source(source_amplitude, src_coord)

                # FONTE EXPLOSIVA: injeta em pressão
                solver._step()
                solver.tau_xx = solver.tau_xx + source_field
                solver.tau_zz = solver.tau_zz + source_field

                pressure_ext = solver.tau_xx + solver.tau_zz
                rec_values = solver._record_receivers(pressure_ext, receiver_coords)
                gather[it, :] = rec_values

                if it % frame_every == 0:
                    wf_phys = solver.get_wavefield_physical().cpu().numpy()
                    gather_np = gather[:it + 1, :].cpu().numpy()

                    plot_wavefield_and_gather(
                        ax_wf, ax_gather,
                        wavefield=wf_phys,
                        vp=vp_np,
                        gather=gather_np,
                        t_current=it * dt,
                        t_max=nt * dt,
                        dt=dt, dx=dx, dz=dz,
                        n_receivers=n_receivers,
                        wavefield_scale=wavefield_scale,
                    )
                    writer.grab_frame()
                    frame_count += 1

                    if frame_count % 15 == 0:
                        elapsed = time.time() - t0
                        pct = 100.0 * it / nt
                        eta = elapsed / (pct / 100.0) - elapsed if pct > 0 else 0
                        print(f"  frame {frame_count}, t = {it*dt*1000:.0f} ms, "
                              f"{pct:.1f}%, elapsed {elapsed:.1f}s, "
                              f"ETA {eta/60:.1f}min")

    print(f"Pronto! {frame_count} frames salvos em {outpath}")
    print(f"Tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
