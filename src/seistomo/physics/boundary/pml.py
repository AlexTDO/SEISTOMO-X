# src/seistomo/physics/boundary/pml.py
"""
Convolutional Perfectly Matched Layer (CPML) para o solver elástico 2D.

A CPML é aplicada sobre o **grid estendido** que inclui o modelo físico
no centro e a PML nas bordas. O solver (Elastic2D) é responsável por
criar o grid estendido e passar o shape correto para a CPML.

Referência:
    Komatitsch, D., & Martin, R. (2007). An unsplit convolutional
    perfectly matched layer improved at grazing incidence for the
    seismic wave equation. Geophysics, 72(4), SM155-SM167.
"""
import torch


class CPML:
    """
    Convolutional Perfectly Matched Layer (CPML) para o solver elástico 2D.

    IMPORTANTE: o `model_shape` passado deve ser o shape do GRID ESTENDIDO
    (modelo físico + 2*pml_width em cada direção), não o shape do modelo
    físico original. O solver Elastic2D cuida disso.
    """
    def __init__(self, model_shape, dx, dz, width=20,
                 vp_max=5000.0, order=4, device='cpu'):
        self.width = width
        self.nz, self.nx = model_shape  # shape do GRID ESTENDIDO
        self.dx = dx
        self.dz = dz
        self.device = device

        # Parâmetros da CPML
        alpha_max = torch.pi * 100.0
        sigma_max = (order + 1) * vp_max * torch.log(torch.tensor(1e6)) / (2 * width * dx)

        # Perfis de amortecimento (sigma) e frequência (alpha) nas bordas
        self.sigma_x = torch.zeros(self.nx, device=device)
        self.sigma_z = torch.zeros(self.nz, device=device)
        self.alpha_x = torch.zeros(self.nx, device=device)
        self.alpha_z = torch.zeros(self.nz, device=device)

        for i in range(width):
            dist = (width - i) / width
            val_sigma = sigma_max * (dist ** 3)
            val_alpha = alpha_max * (1 - dist)

            self.sigma_x[i] = val_sigma
            self.sigma_z[i] = val_sigma
            self.alpha_x[i] = val_alpha
            self.alpha_z[i] = val_alpha

            self.sigma_x[-(i + 1)] = val_sigma
            self.sigma_z[-(i + 1)] = val_sigma
            self.alpha_x[-(i + 1)] = val_alpha
            self.alpha_z[-(i + 1)] = val_alpha

        # Variáveis de memória (acumuladores) para cada derivada de cada campo
        self.memory = {}
        for field in ['vx', 'vz', 'tau_xx', 'tau_zz', 'tau_xz']:
            for deriv in ['dx', 'dz']:
                self.memory[f'{field}_{deriv}'] = torch.zeros(model_shape, device=device)

        # Coeficientes a e b (calculados em update_coefficients)
        self.a_x = None
        self.b_x = None
        self.a_z = None
        self.b_z = None

    def update_coefficients(self, dt):
        """Calcula os coeficientes a e b da CPML para um dado dt."""
        self.b_x = torch.exp(-(self.sigma_x + self.alpha_x) * dt)
        self.a_x = torch.where(
            self.sigma_x > 0,
            self.sigma_x / (self.sigma_x + self.alpha_x) * (self.b_x - 1),
            torch.zeros_like(self.sigma_x),
        )

        self.b_z = torch.exp(-(self.sigma_z + self.alpha_z) * dt)
        self.a_z = torch.where(
            self.sigma_z > 0,
            self.sigma_z / (self.sigma_z + self.alpha_z) * (self.b_z - 1),
            torch.zeros_like(self.sigma_z),
        )

    def apply(self, field_name, deriv_name, deriv_value):
        """Aplica a CPML a uma derivada espacial de um campo."""
        mem_key = f'{field_name}_{deriv_name}'
        mem = self.memory[mem_key]

        if deriv_name == 'dx':
            mem = self.b_x[None, :] * mem + self.a_x[None, :] * deriv_value
            corrected = deriv_value + mem
        else:
            mem = self.b_z[:, None] * mem + self.a_z[:, None] * deriv_value
            corrected = deriv_value + mem

        self.memory[mem_key] = mem
        return corrected

    def reset_memory(self):
        """Zera as variáveis de memória. Chamado no início de cada tiro."""
        for key in self.memory:
            self.memory[key].zero_()