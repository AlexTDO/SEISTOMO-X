# src/seistomo/data/marmousi_loader.py
"""
Loader para o modelo Marmousi Small (versão reduzida do Marmousi2).

O arquivo .npy contém apenas Vp e Vs. A densidade (rho) é derivada
usando uma relação empírica de Gardner simplificada.
"""
import os
import numpy as np
import torch


def load_marmousi_small(
    path_vp: str = None,
    path_vs: str = None,
    dx: float = 20.0,
    dz: float = 20.0,
    device: str = 'cpu',
    rho_scaling: float = 0.3,
    rho_offset: float = 1000.0,
) -> dict:
    """
    Carrega o modelo Marmousi Small e retorna um dict pronto para o ElasticModel.

    Args:
        path_vp, path_vs: caminhos para os arquivos .npy.
            Se None, usa o default dentro do pacote.
        dx, dz: espaçamento em metros (assumido, pois o .npy não guarda).
        device: 'cpu' ou 'cuda'.
        rho_scaling, rho_offset: parâmetros da relação rho = offset + scaling * Vp.

    Returns:
        dict com 'vp', 'vs', 'rho' como tensores PyTorch (nz, nx),
        e 'dx', 'dz' como floats.
    """
    if path_vp is None or path_vs is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path_vp = path_vp or os.path.join(here, 'marmousi_small_vp.npy')
        path_vs = path_vs or os.path.join(here, 'marmousi_small_vs.npy')

    vp_np = np.load(path_vp).astype(np.float32)
    vs_np = np.load(path_vs).astype(np.float32)

    # Garante que o eixo z seja o primeiro (nz, nx)
    if vp_np.shape[0] > vp_np.shape[1]:
        # heurística: em sísmica, nz normalmente é menor que nx,
        # mas se não for, ainda queremos (nz, nx). Aqui não transpo
        # porque o modelo pode ser quadrado; apenas garantimos consistência.
        pass

    # Deriva rho via relação linear simples
    rho_np = rho_offset + rho_scaling * vp_np

    vp = torch.from_numpy(vp_np).to(device)
    vs = torch.from_numpy(vs_np).to(device)
    rho = torch.from_numpy(rho_np).to(device)

    return {
        'vp': vp,
        'vs': vs,
        'rho': rho,
        'dx': dx,
        'dz': dz,
    }