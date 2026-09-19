# src/seistomo/physics/boundary/pml.py
import torch
import torch.nn.functional as F
from ..fd.operators import FiniteDifferenceOperators

class CPML:
    """
    Convolutional Perfectly Matched Layer (CPML) para o solver elástico 2D.
    Esta é uma implementação de produção que usa variáveis de memória
    para absorver ondas de forma eficaz nas bordas do modelo.
    """
    def __init__(self, model_shape, dx, dz, width=20, vp_max=5000.0, order=4, device='cpu'):
        self.width = width
        self.nz, self.nx = model_shape
        self.dx = dx
        self.dz = dz
        self.device = device
        
        # Parâmetros da CPML
        # alpha_max: frequência de corte para o amortecimento
        # vp_max: velocidade máxima para calcular o amortecimento
        alpha_max = torch.pi * 100.0  # Frequência típica de 100 Hz
        sigma_max = (order + 1) * vp_max * torch.log(torch.tensor(1e6)) / (2 * width * dx)
        
        # Cria perfis de amortecimento (sigma) e frequência (alpha)
        # Usamos um perfil polinomial de grau 3 para suavidade
        self.sigma_x = torch.zeros(self.nx, device=device)
        self.sigma_z = torch.zeros(self.nz, device=device)
        self.alpha_x = torch.zeros(self.nx, device=device)
        self.alpha_z = torch.zeros(self.nz, device=device)
        
        # Preenche as bordas
        for i in range(width):
            # Perfil para a borda esquerda/inferior
            dist = (width - i) / width
            val_sigma = sigma_max * (dist ** 3)
            val_alpha = alpha_max * (1 - dist)
            
            self.sigma_x[i] = val_sigma
            self.sigma_z[i] = val_sigma
            self.alpha_x[i] = val_alpha
            self.alpha_z[i] = val_alpha
            
            # Perfil para a borda direita/superior
            self.sigma_x[-(i+1)] = val_sigma
            self.sigma_z[-(i+1)] = val_sigma
            self.alpha_x[-(i+1)] = val_alpha
            self.alpha_z[-(i+1)] = val_alpha
        
        # Coeficientes para as equações de atualização da CPML
        # b = exp(-(sigma + alpha) * dt)
        # a = sigma / (sigma + alpha) * (b - 1)
        # Estes são calculados no solver, pois dependem do dt.
        # Aqui armazenamos apenas os perfis.
        
        # Variáveis de memória para a CPML (acumuladores)
        # Precisamos de uma para cada derivada espacial de cada campo.
        # Campos: vx, vz, tau_xx, tau_zz, tau_xz
        # Derivadas: d/dx, d/dz para cada campo
        # Total: 5 campos * 2 derivadas * 2 direções (x e z) = 20 variáveis de memória.
        # Na prática, a CPML é aplicada apenas nas bordas, então podemos armazenar
        # apenas as fatias correspondentes, mas para simplicidade e performance
        # em GPU, armazenamos tensores completos.
        self.memory = {}
        for field in ['vx', 'vz', 'tau_xx', 'tau_zz', 'tau_xz']:
            for deriv in ['dx', 'dz']:
                self.memory[f'{field}_{deriv}'] = torch.zeros(model_shape, device=device)

    def update_coefficients(self, dt):
        """Calcula os coeficientes a e b da CPML para um dado dt."""
        # Coeficientes para a direção x
        self.b_x = torch.exp(-(self.sigma_x + self.alpha_x) * dt)
        self.a_x = torch.where(
            self.sigma_x > 0,
            self.sigma_x / (self.sigma_x + self.alpha_x) * (self.b_x - 1),
            torch.zeros_like(self.sigma_x)
        )
        
        # Coeficientes para a direção z
        self.b_z = torch.exp(-(self.sigma_z + self.alpha_z) * dt)
        self.a_z = torch.where(
            self.sigma_z > 0,
            self.sigma_z / (self.sigma_z + self.alpha_z) * (self.b_z - 1),
            torch.zeros_like(self.sigma_z)
        )

    def apply(self, field_name, deriv_name, deriv_value):
        """
        Aplica a CPML a uma derivada espacial de um campo.
        
        Args:
            field_name: Nome do campo ('vx', 'vz', 'tau_xx', etc.)
            deriv_name: Nome da derivada ('dx' ou 'dz')
            deriv_value: O valor da derivada calculada pelo operador FD.
        
        Returns:
            A derivada corrigida pela CPML.
        """
        mem_key = f'{field_name}_{deriv_name}'
        mem = self.memory[mem_key]
        
        if deriv_name == 'dx':
            # Atualiza a memória
            mem = self.b_x[None, :] * mem + self.a_x[None, :] * deriv_value
            # Aplica a correção
            corrected = deriv_value + mem
        else:  # dz
            mem = self.b_z[:, None] * mem + self.a_z[:, None] * deriv_value
            corrected = deriv_value + mem
        
        self.memory[mem_key] = mem
        return corrected

    def reset_memory(self):
        """Zera as variáveis de memória. Deve ser chamado no início de cada tiro."""
        for key in self.memory:
            self.memory[key].zero_()