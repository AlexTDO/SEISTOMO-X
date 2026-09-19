# benchmarks/benchmark_cpml_width.py
"""
Estuda o efeito da largura da CPML na absorção de reflexões.
"""
import torch
import matplotlib.pyplot as plt

from seistomo.acquisition.source import Ricker
from seistomo.acquisition.survey import SurfaceSurvey
from seistomo.model.elastic import ElasticModel
from seistomo.physics.elastic.elastic2d import Elastic2D


def run_with_pml_width(width, device='cpu'):
    nz, nx = 150, 150
    vp = torch.ones(nz, nx, device=device) * 2000.0
    vs = torch.ones(nz, nx, device=device) * 1000.0
    rho = torch.ones(nz, nx, device=device) * 2000.0
    model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

    survey = SurfaceSurvey(
        source_x=[75],
        receiver_x=[75],
        z=5.0, dt=0.001, nt=1000,
    )

    solver = Elastic2D(
        model=model, dt=survey.dt, nt=survey.nt,
        pml_width=width, fd_order=4, device=device,
    )
    data = solver.forward(survey=survey, wavelet=Ricker(f0=25))
    return data[0, 0].cpu().numpy()


if __name__ == "__main__":
    widths = [5, 10, 20, 30]
    plt.figure(figsize=(12, 6))
    for w in widths:
        trace = run_with_pml_width(w)
        plt.plot(trace, label=f"PML width={w}")

    plt.title("Efeito da largura da CPML no traço sísmico (receptor no centro)")
    plt.xlabel("Amostra temporal")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("cpml_width_benchmark.png", dpi=150)
    print("Figura salva em cpml_width_benchmark.png")
    plt.show()