# src/seistomo/visualization/wavefield.py
"""
Funções de visualização para o wavefield e a seção sísmica.

Versão profissional (padrão da indústria):
  - Wavefield com escala dinâmica por frame (sempre visível)
  - Seção sísmica com:
      * Filtro passa-baixa temporal (remove alta frequência espúria)
      * AGC (Automatic Gain Control) suave
      * Escala percentil estável
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


def apply_agc(gather: np.ndarray, window_samples: int = 120) -> np.ndarray:
    """Automatic Gain Control — normaliza pela RMS local (AGC suave)."""
    rms = np.sqrt(uniform_filter1d(gather**2, size=window_samples, axis=0) + 1e-12)
    return gather / rms


def apply_lowpass(gather: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Filtro passa-baixa temporal (Gaussiano) na seção sísmica.

    Remove alta frequência espúria da seção sísmica, deixando os eventos
    mais definidos e legíveis (padrão em processamento sísmico).

    Args:
        gather: (nt, n_receivers)
        sigma: desvio padrão do filtro gaussiano (em amostras).
               Valores maiores = mais suavização.

    Returns:
        gather filtrada.
    """
    return gaussian_filter1d(gather, sigma=sigma, axis=0)


def plot_wavefield_and_gather(
    ax_wavefield: plt.Axes,
    ax_gather: plt.Axes,
    wavefield: np.ndarray,
    vp: np.ndarray,
    gather: np.ndarray,
    t_current: float,
    t_max: float,
    dt: float,
    dx: float,
    dz: float,
    n_receivers: int,
    wavefield_scale: float = None,
    gather_scale: float = None,
    source_x_km: float = None,
    receiver_positions_km: np.ndarray = None,
    vp_for_theory: float = None,
    vs_for_theory: float = None,
    title_wavefield: str = "Wavefield",
    title_gather: str = "Seismic Section",
):
    nz, nx = wavefield.shape

    # ==================================================================
    # PAINEL 1: Wavefield
    # ==================================================================
    ax_wavefield.clear()

    extent = [0, nx * dx / 1000.0, nz * dz / 1000.0, 0]

    # Modelo de fundo
    ax_wavefield.imshow(
        vp, cmap='gray_r', aspect='auto', extent=extent,
        alpha=0.55, vmin=np.percentile(vp, 2), vmax=np.percentile(vp, 98)
    )

    # Escala dinâmica por frame (sempre visível)
    wf_abs = np.abs(wavefield)
    wf_scale = np.percentile(wf_abs, 99.5)
    if wf_scale < 1e-10:
        wf_scale = 1.0

    ax_wavefield.imshow(
        wavefield, cmap='seismic', aspect='auto', extent=extent,
        alpha=0.85, vmin=-wf_scale, vmax=wf_scale
    )

    ax_wavefield.set_title(
        f"{title_wavefield}  |  t = {t_current*1000:.0f} ms  "
        f"({100*t_current/t_max:.0f}%)",
        fontsize=11, fontweight='bold'
    )
    ax_wavefield.set_xlabel("Distance (km)", fontsize=9)
    ax_wavefield.set_ylabel("Depth (km)", fontsize=9)
    ax_wavefield.tick_params(labelsize=8)

    # ==================================================================
    # PAINEL 2: Seção sísmica — filtro + AGC
    # ==================================================================
    ax_gather.clear()

    if gather.shape[0] > 0:
        # 1. Filtro passa-baixa (remove alta frequência)
        gather_filt = apply_lowpass(gather, sigma=2.0)

        # 2. AGC (revela eventos fracos)
        gather_agc = apply_agc(gather_filt, window_samples=120)

        # 3. Escala por percentil (estável)
        g_scale = np.percentile(np.abs(gather_agc), 95)
        if g_scale < 1e-10:
            g_scale = 1.0

        ax_gather.imshow(
            gather_agc.T, cmap='seismic', aspect='auto',
            extent=[0, n_receivers, gather.shape[0] * dt, 0],
            vmin=-g_scale, vmax=g_scale
        )

        # Curvas teóricas opcionais
        if (source_x_km is not None
                and receiver_positions_km is not None
                and vp_for_theory is not None):
            dist_km = np.abs(receiver_positions_km - source_x_km)
            dist_m = dist_km * 1000.0

            t_p = dist_m / vp_for_theory
            ax_gather.plot(
                np.arange(n_receivers), t_p,
                'c--', linewidth=1.2, alpha=0.85, label='t = d/Vp'
            )

            if vs_for_theory is not None:
                t_s = dist_m / vs_for_theory
                ax_gather.plot(
                    np.arange(n_receivers), t_s,
                    'm--', linewidth=1.2, alpha=0.85, label='t = d/Vs'
                )

            ax_gather.legend(loc='lower right', fontsize=8)

    ax_gather.set_title(
        f"{title_gather}  |  {gather.shape[0]} of {int(t_max/dt)} time samples",
        fontsize=11, fontweight='bold'
    )
    ax_gather.set_xlabel("Receiver index", fontsize=9)
    ax_gather.set_ylabel("Time (s)", fontsize=9)
    ax_gather.tick_params(labelsize=8)
    ax_gather.set_ylim(t_max, 0)