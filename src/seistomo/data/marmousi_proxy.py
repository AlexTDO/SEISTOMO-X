# src/seistomo/data/marmousi_proxy.py
"""
Gerador de um modelo sísmico inspirado no Marmousi.

Não é o Marmousi oficial (que tem licença específica), mas captura
as características principais: gradiente vertical de velocidade,
camadas inclinadas, falhas e contrastes laterais.
"""
import torch


def marmousi_proxy(nz: int = 200, nx: int = 400,
                   vp_min: float = 1500.0, vp_max: float = 4500.0,
                   device: str = 'cpu') -> dict:
    """
    Gera um modelo Marmousi-like.

    Args:
        nz, nx: dimensões da grade (profundidade, distância)
        vp_min, vp_max: faixa de velocidades P (m/s)
        device: 'cpu' ou 'cuda'

    Returns:
        dict com 'vp', 'vs', 'rho' como tensores (nz, nx)
    """
    # Coordenadas normalizadas
    z = torch.linspace(0, 1, nz, device=device).unsqueeze(1)  # (nz, 1)
    x = torch.linspace(0, 1, nx, device=device).unsqueeze(0)  # (1, nx)

    # Gradiente vertical suave (velocidade cresce com a profundidade)
    vp = vp_min + (vp_max - vp_min) * (0.3 + 0.7 * z)

    # Adiciona camadas horizontais com contraste
    vp = vp + 300.0 * torch.sin(6.0 * torch.pi * z)

    # Adiciona inclinação lateral (como no Marmousi)
    vp = vp + 400.0 * torch.tanh(3.0 * (x - 0.5)) * z

    # Adiciona estrutura tipo falha (descontinuidade lateral)
    fault_mask = (x > 0.35) & (x < 0.40)
    vp = vp + 500.0 * fault_mask.float() * (z > 0.3).float()

    # Adiciona bloco de alta velocidade no fundo
    block = ((x - 0.7) ** 2 + (z - 0.8) ** 2) < 0.03
    vp = vp + 800.0 * block.float()

    # Adiciona lente de baixa velocidade no meio
    lens = ((x - 0.2) ** 2 / 0.02 + (z - 0.5) ** 2 / 0.01) < 1.0
    vp = vp - 400.0 * lens.float()

    # Suaviza com um filtro gaussiano simples (via média local)
    # Isso evita descontinuidades que causariam reflexões espúrias
    from torch.nn.functional import avg_pool2d
    vp_4d = vp.unsqueeze(0).unsqueeze(0)  # (1, 1, nz, nx)
    # padding manual para preservar dimensões
    import torch.nn.functional as F
    vp_padded = F.pad(vp_4d, (2, 2, 2, 2), mode='replicate')
    vp_smooth = F.avg_pool2d(vp_padded, kernel_size=5, stride=1).squeeze(0).squeeze(0)

    # Vs e rho derivados (relação empírica)
    vs = vp_smooth / 2.0
    rho = 1000.0 + 0.3 * vp_smooth

    return {
        'vp': vp_smooth.contiguous(),
        'vs': vs.contiguous(),
        'rho': rho.contiguous(),
    }