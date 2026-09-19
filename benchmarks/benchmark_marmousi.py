# benchmarks/benchmark_marmousi.py
"""
Benchmark do solver Elastic2D no modelo Marmousi (ou um proxy).
Mede tempo de forward e tempo de backward, em CPU e GPU.
"""
import time
import torch

from seistomo.acquisition.source import Ricker
from seistomo.acquisition.survey import SurfaceSurvey
from seistomo.model.elastic import ElasticModel
from seistomo.physics.elastic.elastic2d import Elastic2D


def make_marmousi_proxy(nz=200, nx=400, device='cpu'):
    """
    Proxy sintético inspirado no Marmousi: gradiente vertical + camada.
    Não é o Marmousi real, mas serve para medir performance.
    """
    vp = torch.linspace(1500, 4500, nz, device=device).unsqueeze(1).expand(nz, nx).clone()
    vs = vp / 2.0
    rho = 1000.0 + 0.3 * vp

    # Adiciona uma camada horizontal para gerar reflexão
    vp[nz // 2, :] *= 1.15
    vs[nz // 2, :] *= 1.15
    rho[nz // 2, :] *= 1.1

    return ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)


def benchmark(device, fd_order=4, pml_width=20, nt=500):
    print(f"\n=== Benchmark device={device}, fd_order={fd_order}, nt={nt} ===")
    model = make_marmousi_proxy(device=device)

    survey = SurfaceSurvey(
        source_x=list(range(20, 380, 40)),
        receiver_x=list(range(10, 390, 10)),
        z=0.0, dt=0.001, nt=nt,
    )

    solver = Elastic2D(
        model=model, dt=survey.dt, nt=survey.nt,
        pml_width=pml_width, fd_order=fd_order, device=device,
    )

    # Warm-up (para torch.compile e alocação de memória)
    _ = solver.forward(survey=survey, wavelet=Ricker(f0=25))
    if device == 'cuda':
        torch.cuda.synchronize()

    # Forward
    t0 = time.time()
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))
    if device == 'cuda':
        torch.cuda.synchronize()
    t_forward = time.time() - t0

    # Backward
    vp = model.vp.clone().requires_grad_(True)
    vs = model.vs.clone().requires_grad_(True)
    rho = model.rho.clone().requires_grad_(True)
    model_g = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)
    solver_g = Elastic2D(model=model_g, dt=survey.dt, nt=survey.nt,
                         pml_width=pml_width, fd_order=fd_order, device=device)

    if device == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    data_g = solver_g.forward(survey=survey, wavelet=Ricker(f0=25))
    loss = data_g.pow(2).sum()
    loss.backward()
    if device == 'cuda':
        torch.cuda.synchronize()
    t_backward = time.time() - t0 - t_forward if t_forward < 1e9 else 0.0

    print(f"Forward:  {t_forward:.3f} s")
    print(f"Backward: {t_backward:.3f} s")
    print(f"Data shape: {data.shape}")


if __name__ == "__main__":
    benchmark('cpu', fd_order=4, nt=500)

    if torch.cuda.is_available():
        benchmark('cuda', fd_order=4, nt=500)
        benchmark('cuda', fd_order=8, nt=500)
    else:
        print("\nCUDA não disponível — pulando benchmarks de GPU.")