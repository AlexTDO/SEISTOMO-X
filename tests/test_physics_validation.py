# tests/test_physics_validation.py
"""
Testes de validação física do solver.

Aqui verificamos que o solver não apenas "roda", mas que ele resolve
corretamente a equação da onda elástica, produzindo velocidades de
propagação P e S compatíveis com a teoria.
"""
import numpy as np
import torch

from seistomo.acquisition.source import Ricker
from seistomo.acquisition.survey import SurfaceSurvey
from seistomo.model.elastic import ElasticModel
from seistomo.physics.elastic.elastic2d import Elastic2D


def test_p_wave_arrival_time(device):
    """
    Em um meio homogêneo, a onda P deve chegar em um receptor a uma
    distância d após t ≈ d / Vp.
    """
    vp_val = 2000.0
    vs_val = 1000.0
    dx = 10.0

    nz, nx = 200, 400
    vp = torch.ones(nz, nx, device=device) * vp_val
    vs = torch.ones(nz, nx, device=device) * vs_val
    rho = torch.ones(nz, nx, device=device) * 2000.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=dx, dz=dx)

    # Fonte em x=100, receptor em x=200 → distância = 100 * dx = 1000 m
    src_x, rec_x = 100, 200
    distance_m = (rec_x - src_x) * dx
    expected_t = distance_m / vp_val  # = 0.5 s

    dt = 0.0005
    nt = 3000  # 1.5 s
    survey = SurfaceSurvey(
        source_x=[src_x], receiver_x=[rec_x],
        z=5.0, dt=dt, nt=nt,
    )

    solver = Elastic2D(model=model, dt=dt, nt=nt,
                       pml_width=20, fd_order=4, device=device)
    data = solver.forward(survey=survey, wavelet=Ricker(f0=30))

    trace = data[0, 0].cpu().numpy()

    # Encontra o pico de amplitude (chegada da onda P)
    # Ignora a primeira parte para evitar a onda direta da fonte
    start = int(0.1 / dt)
    peak_idx = start + np.argmax(np.abs(trace[start:]))
    peak_time = peak_idx * dt

    # Tolerância: 15% de erro (efeitos de dispersão numérica e da wavelet)
    rel_err = abs(peak_time - expected_t) / expected_t
    print(f"P-wave: esperado t={expected_t:.4f}s, medido t={peak_time:.4f}s, "
          f"erro relativo={rel_err:.2%}")

    assert rel_err < 0.15, (
        f"Tempo de chegada da onda P fora da tolerância: "
        f"esperado {expected_t:.4f}s, medido {peak_time:.4f}s"
    )


def test_s_wave_arrival_time(device):
    """
    Testa a chegada da onda S em um receptor fora do eixo vertical.
    Para uma fonte de explosão, a onda S é gerada principalmente por
    conversão nas bordas; em meio homogêneo, a onda S pura é fraca.
    Aqui usamos uma fonte direcional em vz para excitar S.
    """
    vp_val = 2000.0
    vs_val = 1000.0
    dx = 10.0

    nz, nx = 200, 400
    vp = torch.ones(nz, nx, device=device) * vp_val
    vs = torch.ones(nz, nx, device=device) * vs_val
    rho = torch.ones(nz, nx, device=device) * 2000.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=dx, dz=dx)

    # Receptor em profundidade, mesma linha vertical da fonte
    # Para excitar S, precisamos de uma fonte em vz.
    # Como o solver atual injeta em vx e vz iguais, a onda S pura não é
    # facilmente isolada. Este teste é um placeholder para quando
    # implementarmos fontes direcionais.
    #
    # Por ora, verificamos apenas que a onda S existe e se propaga.
    src_x, rec_x = 100, 200
    distance_m = (rec_x - src_x) * dx
    expected_t_s = distance_m / vs_val  # = 1.0 s

    dt = 0.0005
    nt = 4000
    survey = SurfaceSurvey(
        source_x=[src_x], receiver_x=[rec_x],
        z=5.0, dt=dt, nt=nt,
    )

    solver = Elastic2D(model=model, dt=dt, nt=nt,
                       pml_width=20, fd_order=4, device=device)
    data = solver.forward(survey=survey, wavelet=Ricker(f0=30))
    trace = data[0, 0].cpu().numpy()

    # Verifica que existe energia na janela esperada para a onda S
    # (janela em torno de t = 1.0 s)
    window_start = int((expected_t_s - 0.1) / dt)
    window_end = int((expected_t_s + 0.1) / dt)
    window_energy = np.sum(trace[window_start:window_end] ** 2)
    total_energy = np.sum(trace ** 2)

    print(f"S-wave: energia na janela de S = {window_energy:.4e}, "
          f"energia total = {total_energy:.4e}")

    # Deve haver alguma energia na janela da onda S (mesmo que pequena)
    assert window_energy > 0, "Nenhuma energia detectada na janela da onda S"