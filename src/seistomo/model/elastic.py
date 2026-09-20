# src/seistomo/model/elastic.py
import torch


class ElasticModel:
    """
    Representa um modelo elástico 2D (Vp, Vs, rho).
    As propriedades são armazenadas como tensores do PyTorch para permitir
    a diferenciação automática.
    """
    def __init__(self, vp, vs, rho, dx=1.0, dz=1.0):
        # Garante que os inputs sejam tensores do PyTorch
        self.vp = torch.as_tensor(vp, dtype=torch.float32)
        self.vs = torch.as_tensor(vs, dtype=torch.float32)
        self.rho = torch.as_tensor(rho, dtype=torch.float32)

        self.dx = dx
        self.dz = dz

        # Validação básica de shapes
        if not (self.vp.shape == self.vs.shape == self.rho.shape):
            raise ValueError("Vp, Vs e rho devem ter as mesmas dimensões.")

        self.shape = self.vp.shape
        self.nz, self.nx = self.shape

    def __repr__(self):
        return (f"ElasticModel(shape={self.shape}, dx={self.dx}, dz={self.dz})\n"
                f"  Vp range: [{self.vp.min():.2f}, {self.vp.max():.2f}]\n"
                f"  Vs range: [{self.vs.min():.2f}, {self.vs.max():.2f}]\n"
                f"  Rho range: [{self.rho.min():.2f}, {self.rho.max():.2f}]")