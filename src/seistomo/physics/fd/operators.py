# src/seistomo/physics/fd/operators.py
import torch
import torch.nn.functional as F


class FiniteDifferenceOperators:
    """
    Operadores de diferenças finitas otimizados usando convolução.
    Suporta ordem 2, 4 e 8. Para produção, ordem 4 é um bom equilíbrio.
    """
    def __init__(self, dx: float, dz: float, order: int = 4, device='cpu'):
        self.dx = dx
        self.dz = dz
        self.order = order
        self.device = device

        # Coeficientes de diferenças finitas centrais para a primeira derivada
        # Fórmula: f'(x) ≈ (1/dx) * sum(c_k * f(x + k*dx))
        # Para ordem 2: c = [-0.5, 0, 0.5]
        # Para ordem 4: c = [1/12, -2/3, 0, 2/3, -1/12]
        # Para ordem 8: c = [1/280, -4/105, 1/5, -4/5, 0, 4/5, -1/5, 4/105, -1/280]

        if order == 2:
            coeffs = torch.tensor([-0.5, 0.0, 0.5], dtype=torch.float32, device=device)
        elif order == 4:
            coeffs = torch.tensor([1/12, -2/3, 0.0, 2/3, -1/12], dtype=torch.float32, device=device)
        elif order == 8:
            coeffs = torch.tensor(
                [1/280, -4/105, 1/5, -4/5, 0.0, 4/5, -1/5, 4/105, -1/280],
                dtype=torch.float32, device=device
            )
        else:
            raise ValueError(f"Ordem {order} não suportada. Use 2, 4 ou 8.")

        self.coeffs = coeffs
        self.pad = order // 2

        # Cria os kernels de convolução para as derivadas em x e z
        # O kernel tem shape (out_channels=1, in_channels=1, kH, kW)
        # Para d/dx, o kernel é horizontal (1, 1, 1, 2*pad+1)
        # Para d/dz, o kernel é vertical (1, 1, 2*pad+1, 1)
        self.kernel_dx = coeffs.view(1, 1, 1, -1) / dx
        self.kernel_dz = coeffs.view(1, 1, -1, 1) / dz

    def d_dx(self, field: torch.Tensor) -> torch.Tensor:
        """Calcula a derivada parcial em relação a x usando convolução."""
        # Padding para manter as dimensões
        padded = F.pad(field.unsqueeze(0).unsqueeze(0), (self.pad, self.pad, 0, 0), mode='replicate')
        return F.conv2d(padded, self.kernel_dx).squeeze(0).squeeze(0)

    def d_dz(self, field: torch.Tensor) -> torch.Tensor:
        """Calcula a derivada parcial em relação a z usando convolução."""
        padded = F.pad(field.unsqueeze(0).unsqueeze(0), (0, 0, self.pad, self.pad), mode='replicate')
        return F.conv2d(padded, self.kernel_dz).squeeze(0).squeeze(0)

    def laplacian(self, field: torch.Tensor) -> torch.Tensor:
        """Calcula o Laplaciano. Útil para equações acústicas."""
        # Para o Laplaciano, usamos a segunda derivada.
        # Coeficientes para segunda derivada:
        # Ordem 2: [1, -2, 1]
        # Ordem 4: [-1/12, 4/3, -5/2, 4/3, -1/12]
        if self.order == 2:
            coeffs = torch.tensor([1.0, -2.0, 1.0], dtype=torch.float32, device=self.device)
        elif self.order == 4:
            coeffs = torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12], dtype=torch.float32, device=self.device)
        else:
            raise ValueError("Laplaciano implementado apenas para ordem 2 e 4.")

        pad = len(coeffs) // 2
        kernel_x = (coeffs / (self.dx**2)).view(1, 1, 1, -1)
        kernel_z = (coeffs / (self.dz**2)).view(1, 1, -1, 1)

        padded_x = F.pad(field.unsqueeze(0).unsqueeze(0), (pad, pad, 0, 0), mode='replicate')
        padded_z = F.pad(field.unsqueeze(0).unsqueeze(0), (0, 0, pad, pad), mode='replicate')

        d2x = F.conv2d(padded_x, kernel_x).squeeze(0).squeeze(0)
        d2z = F.conv2d(padded_z, kernel_z).squeeze(0).squeeze(0)

        return d2x + d2z