# src/seistomo/physics/boundary/extend.py
"""
Extensão de campos 2D para acomodar a PML fora do domínio físico.

A PML é adicionada FORA do modelo físico. Para isso, o solver cria um
grid maior que inclui o modelo original no centro e camadas de PML nas
bordas. As células da PML são preenchidas replicando as bordas do modelo.
"""
import torch
import torch.nn.functional as F


def extend_with_pml(field: torch.Tensor, pml_width: int) -> torch.Tensor:
    """
    Estende um campo 2D replicando as bordas para a região da PML.

    Args:
        field: tensor (nz, nx) — o modelo físico.
        pml_width: número de células da PML em cada direção.

    Returns:
        tensor (nz + 2*pml_width, nx + 2*pml_width) — modelo estendido.
    """
    if pml_width <= 0:
        return field
    extended = F.pad(
        field.unsqueeze(0).unsqueeze(0),
        (pml_width, pml_width, pml_width, pml_width),
        mode='replicate',
    )
    return extended.squeeze(0).squeeze(0)