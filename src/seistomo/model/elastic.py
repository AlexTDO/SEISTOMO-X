# src/seistomo/physics/elastic/elastic2d.py
import torch
import torch.nn.functional as F
from ...model.elastic import ElasticModel
from ..fd.operators import FiniteDifferenceOperators
from ..boundary.pml import CPML

class Elastic2D:
    """
    Solver de produção para a equação da onda elástica 2D (P-SV).
    Utiliza diferenças finitas de alta ordem, CPML e é totalmente diferenciável.
    Projetado para rodar em GPU com performance otimizada.
    """
    def __init__(self, model: ElasticModel, dt: float, nt: int, 
                 pml_width: int = 20, fd_order: int = 4, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.dt = dt
        self.nt = nt
        self.device = device
        
        # Move o modelo para o dispositivo
        self.model.vp = self.model.vp.to(device)
        self.model.vs = self.model.vs.to(device)
        self.model.rho = self.model.rho.to(device)
        
        # Inicializa os operadores de diferenças finitas
        self.fd = FiniteDifferenceOperators(dx=model.dx, dz=model.dz, order=fd_order, device=device)
        
        # Inicializa a CPML
        vp_max = self.model.vp.max().item()
        self.pml = CPML(
            model_shape=model.shape,
            dx=model.dx,
            dz=model.dz,
            width=pml_width,
            vp_max=vp_max,
            order=fd_order,
            device=device
        )
        self.pml.update_coefficients(dt)
        
        # Pré-calcula os coeficientes elásticos
        self._precompute_coefficients()
        
        # Inicializa os campos de onda
        self._init_fields()

    def _precompute_coefficients(self):
        """Calcula os coeficientes elásticos (Lame parameters)."""
        vp2 = self.model.vp ** 2
        vs2 = self.model.vs ** 2
        rho = self.model.rho
        
        # Parâmetros de Lamé
        self.lambda_ = rho * (vp2 - 2 * vs2)
        self.mu = rho * vs2
        
        # Coeficientes para as equações de diferenças finitas
        self.b = 1.0 / rho
        self.lambda_plus_2mu = self.lambda_ + 2 * self.mu

    def _init_fields(self):
        """Inicializa os campos de onda como tensores no dispositivo correto."""
        shape = self.model.shape
        self.vx = torch.zeros(shape, device=self.device)
        self.vz = torch.zeros(shape, device=self.device)
        self.tau_xx = torch.zeros(shape, device=self.device)
        self.tau_zz = torch.zeros(shape, device=self.device)
        self.tau_xz = torch.zeros(shape, device=self.device)

    def _step(self, source_term_vx=None, source_term_vz=None):
        """
        Avança um passo de tempo na simulação.
        A fonte é injetada diretamente nas velocidades (vx, vz) para simular
        uma fonte de força direcional. Para uma fonte de pressão, injetaríamos
        em tau_xx e tau_zz.
        """
        # --- Atualização das velocidades (vx, vz) ---
        # Calcula as derivadas espaciais
        d_tau_xx_dx = self.fd.d_dx(self.tau_xx)
        d_tau_xz_dz = self.fd.d_dz(self.tau_xz)
        d_tau_xz_dx = self.fd.d_dx(self.tau_xz)
        d_tau_zz_dz = self.fd.d_dz(self.tau_zz)
        
        # Aplica a CPML às derivadas
        d_tau_xx_dx = self.pml.apply('tau_xx', 'dx', d_tau_xx_dx)
        d_tau_xz_dz = self.pml.apply('tau_xz', 'dz', d_tau_xz_dz)
        d_tau_xz_dx = self.pml.apply('tau_xz', 'dx', d_tau_xz_dx)
        d_tau_zz_dz = self.pml.apply('tau_zz', 'dz', d_tau_zz_dz)
        
        # Atualiza as velocidades
        self.vx = self.vx + self.dt * self.b * (d_tau_xx_dx + d_tau_xz_dz)
        self.vz = self.vz + self.dt * self.b * (d_tau_xz_dx + d_tau_zz_dz)
        
        # --- Atualização das tensões (tau_xx, tau_zz, tau_xz) ---
        # Calcula as derivadas espaciais das velocidades
        d_vx_dx = self.fd.d_dx(self.vx)
        d_vz_dz = self.fd.d_dz(self.vz)
        d_vx_dz = self.fd.d_dz(self.vx)
        d_vz_dx = self.fd.d_dx(self.vz)
        
        # Aplica a CPML às derivadas
        d_vx_dx = self.pml.apply('vx', 'dx', d_vx_dx)
        d_vz_dz = self.pml.apply('vz', 'dz', d_vz_dz)
        d_vx_dz = self.pml.apply('vx', 'dz', d_vx_dz)
        d_vz_dx = self.pml.apply('vz', 'dx', d_vz_dx)
        
        div_v = d_vx_dx + d_vz_dz
        
        # Atualiza as tensões
        self.tau_xx = self.tau_xx + self.dt * (self.lambda_ * div_v + 2 * self.mu * d_vx_dx)
        self.tau_zz = self.tau_zz + self.dt * (self.lambda_ * div_v + 2 * self.mu * d_vz_dz)
        self.tau_xz = self.tau_xz + self.dt * self.mu * (d_vx_dz + d_vz_dx)
        
        # Adiciona os termos fonte (se houver)
        if source_term_vx is not None:
            self.vx = self.vx + source_term_vx
        if source_term_vz is not None:
            self.vz = self.vz + source_term_vz

    def _inject_source(self, source_term, coords):
        """
        Injeta um termo fonte em coordenadas arbitrárias usando interpolação bilinear.
        
        Args:
            source_term: Tensor de shape (n_sources,) com a amplitude no tempo atual.
            coords: Tensor de shape (n_sources, 2) com as coordenadas (x, z).
        
        Returns:
            Um tensor de shape (nz, nx) com a fonte distribuída.
        """
        nz, nx = self.model.shape
        field = torch.zeros((nz, nx), device=self.device)
        
        # Coordenadas contínuas
        x = coords[:, 0]
        z = coords[:, 1]
        
        # Índices dos nós vizinhos
        x0 = torch.floor(x).long()
        z0 = torch.floor(z).long()
        x1 = x0 + 1
        z1 = z0 + 1
        
        # Pesos da interpolação bilinear
        wx = x - x0.float()
        wz = z - z0.float()
        w00 = (1 - wx) * (1 - wz)
        w01 = (1 - wx) * wz
        w10 = wx * (1 - wz)
        w11 = wx * wz
        
        # Clipa os índices para não sair da grade
        x0 = x0.clamp(0, nx - 1)
        x1 = x1.clamp(0, nx - 1)
        z0 = z0.clamp(0, nz - 1)
        z1 = z1.clamp(0, nz - 1)
        
        # Distribui a fonte usando index_add_ (eficiente e diferenciável)
        # Precisamos achatar os índices para usar index_add_ em um tensor 1D
        indices = z0 * nx + x0
        field.view(-1).index_add_(0, indices, source_term * w00)
        
        indices = z0 * nx + x1
        field.view(-1).index_add_(0, indices, source_term * w01)
        
        indices = z1 * nx + x0
        field.view(-1).index_add_(0, indices, source_term * w10)
        
        indices = z1 * nx + x1
        field.view(-1).index_add_(0, indices, source_term * w11)
        
        return field

    def _record_receivers(self, field, coords):
        """
        Registra o campo em coordenadas arbitrárias usando interpolação bilinear.
        
        Args:
            field: Tensor de shape (nz, nx) com o campo a ser amostrado.
            coords: Tensor de shape (n_receivers, 2) com as coordenadas (x, z).
        
        Returns:
            Um tensor de shape (n_receivers,) com os valores amostrados.
        """
        nz, nx = self.model.shape
        
        x = coords[:, 0]
        z = coords[:, 1]
        
        x0 = torch.floor(x).long()
        z0 = torch.floor(z).long()
        x1 = x0 + 1
        z1 = z0 + 1
        
        wx = x - x0.float()
        wz = z - z0.float()
        w00 = (1 - wx) * (1 - wz)
        w01 = (1 - wx) * wz
        w10 = wx * (1 - wz)
        w11 = wx * wz
        
        x0 = x0.clamp(0, nx - 1)
        x1 = x1.clamp(0, nx - 1)
        z0 = z0.clamp(0, nz - 1)
        z1 = z1.clamp(0, nz - 1)
        
        # Amostra os valores nos quatro nós vizinhos
        f00 = field[z0, x0]
        f01 = field[z0, x1]
        f10 = field[z1, x0]
        f11 = field[z1, x1]
        
        # Combina com os pesos
        return f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11

    @torch.compile
    def forward(self, survey, wavelet):
        """
        Executa a modelagem direta (forward modeling) para todos os tiros.
        Esta função é compilada com torch.compile para máxima performance.
        
        Retorna:
            data: Tensor de forma (n_sources, n_receivers, nt)
        """
        n_sources = len(survey.sources)
        n_receivers = len(survey.receivers)
        
        # Pré-calcula a wavelet
        t = survey.t.to(self.device)
        w = wavelet(t)  # Shape: (nt,)
        
        # Coordenadas das fontes e receptores
        source_coords = survey.source_coords.to(self.device)
        receiver_coords = survey.receiver_coords.to(self.device)
        
        # Tensor para armazenar os dados sísmicos
        data = torch.zeros((n_sources, n_receivers, self.nt), device=self.device)
        
        # Loop sobre os tiros (pode ser vetorizado com vmap no futuro)
        for i in range(n_sources):
            # Reinicia os campos e a memória da CPML para cada tiro
            self._init_fields()
            self.pml.reset_memory()
            
            # Coordenadas da fonte atual
            src_coord = source_coords[i:i+1]  # Shape: (1, 2)
            
            # Loop de tempo
            for it in range(self.nt):
                # Cria o termo fonte para este instante
                # Injetamos a fonte na pressão (tau_xx e tau_zz) para simular uma fonte de explosão
                source_amplitude = w[it].expand(1)  # Shape: (1,)
                source_field = self._inject_source(source_amplitude, src_coord)
                
                # Avança um passo
                self._step(source_term_vx=source_field, source_term_vz=source_field)
                
                # Registra os dados nos receptores
                # Amostra o campo de pressão (aprox. tau_xx + tau_zz)
                pressure = self.tau_xx + self.tau_zz
                data[i, :, it] = self._record_receivers(pressure, receiver_coords)
                    
        return data