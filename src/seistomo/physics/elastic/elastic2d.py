# src/seistomo/physics/elastic/elastic2d.py
"""
Solver elástico 2D (P-SV) com PML externa ao domínio físico.

Arquitetura:
    - O usuário fornece um modelo físico (nz, nx) e uma Survey com
      coordenadas no sistema do modelo físico (z=0 = superfície).
    - O solver cria um GRID ESTENDIDO (nz + 2*pml_width, nx + 2*pml_width)
      onde o modelo físico ocupa o centro e a PML ocupa as bordas.
    - Todas as coordenadas (fontes, receptores) são convertidas para
      o grid estendido internamente.
    - O wavefield resultante pode ser extraído no sistema do modelo físico
      via get_wavefield_physical().

Referência:
    Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
    velocity-stress finite-difference method. Geophysics, 51(4), 889-901.
"""
import torch
import torch.nn.functional as F
from ...model.elastic import ElasticModel
from ..fd.operators import FiniteDifferenceOperators
from ..boundary.pml import CPML
from ..boundary.extend import extend_with_pml


class Elastic2D:
    def __init__(self, model: ElasticModel, dt: float, nt: int,
                 pml_width: int = 20, fd_order: int = 8,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.dt = dt
        self.nt = nt
        self.device = device
        self.pml_width = pml_width

        # Dimensões do modelo físico
        self.nz_phys, self.nx_phys = model.shape

        # Dimensões do grid estendido
        self.nz_ext = self.nz_phys + 2 * pml_width
        self.nx_ext = self.nx_phys + 2 * pml_width

        # Cria o modelo estendido (replicate nas bordas)
        self.vp_ext = extend_with_pml(model.vp, pml_width).to(device)
        self.vs_ext = extend_with_pml(model.vs, pml_width).to(device)
        self.rho_ext = extend_with_pml(model.rho, pml_width).to(device)

        # Operadores FD no grid estendido
        self.fd = FiniteDifferenceOperators(
            dx=model.dx, dz=model.dz, order=fd_order, device=device
        )

        # CPML no grid estendido
        vp_max = self.vp_ext.max().item()
        self.pml = CPML(
            model_shape=(self.nz_ext, self.nx_ext),
            dx=model.dx,
            dz=model.dz,
            width=pml_width,
            vp_max=vp_max,
            order=fd_order,
            device=device,
        )
        self.pml.update_coefficients(dt)

        # Coeficientes elásticos no grid estendido
        self._precompute_coefficients()

        # Campos de onda no grid estendido
        self._init_fields()

    def _precompute_coefficients(self):
        """Calcula os coeficientes elásticos (Lame parameters) no grid estendido."""
        vp2 = self.vp_ext ** 2
        vs2 = self.vs_ext ** 2
        rho = self.rho_ext
        self.lambda_ = rho * (vp2 - 2 * vs2)
        self.mu = rho * vs2
        self.b = 1.0 / rho
        self.lambda_plus_2mu = self.lambda_ + 2 * self.mu

    def _init_fields(self):
        """Inicializa os campos de onda no grid estendido."""
        shape = (self.nz_ext, self.nx_ext)
        self.vx = torch.zeros(shape, device=self.device)
        self.vz = torch.zeros(shape, device=self.device)
        self.tau_xx = torch.zeros(shape, device=self.device)
        self.tau_zz = torch.zeros(shape, device=self.device)
        self.tau_xz = torch.zeros(shape, device=self.device)

    def _step(self, source_term_vx=None, source_term_vz=None):
        """Avança um passo de tempo no grid estendido."""
        d_tau_xx_dx = self.fd.d_dx(self.tau_xx)
        d_tau_xz_dz = self.fd.d_dz(self.tau_xz)
        d_tau_xz_dx = self.fd.d_dx(self.tau_xz)
        d_tau_zz_dz = self.fd.d_dz(self.tau_zz)

        d_tau_xx_dx = self.pml.apply('tau_xx', 'dx', d_tau_xx_dx)
        d_tau_xz_dz = self.pml.apply('tau_xz', 'dz', d_tau_xz_dz)
        d_tau_xz_dx = self.pml.apply('tau_xz', 'dx', d_tau_xz_dx)
        d_tau_zz_dz = self.pml.apply('tau_zz', 'dz', d_tau_zz_dz)

        self.vx = self.vx + self.dt * self.b * (d_tau_xx_dx + d_tau_xz_dz)
        self.vz = self.vz + self.dt * self.b * (d_tau_xz_dx + d_tau_zz_dz)

        d_vx_dx = self.fd.d_dx(self.vx)
        d_vz_dz = self.fd.d_dz(self.vz)
        d_vx_dz = self.fd.d_dz(self.vx)
        d_vz_dx = self.fd.d_dx(self.vz)

        d_vx_dx = self.pml.apply('vx', 'dx', d_vx_dx)
        d_vz_dz = self.pml.apply('vz', 'dz', d_vz_dz)
        d_vx_dz = self.pml.apply('vx', 'dz', d_vx_dz)
        d_vz_dx = self.pml.apply('vz', 'dx', d_vz_dx)

        div_v = d_vx_dx + d_vz_dz

        self.tau_xx = self.tau_xx + self.dt * (self.lambda_ * div_v + 2 * self.mu * d_vx_dx)
        self.tau_zz = self.tau_zz + self.dt * (self.lambda_ * div_v + 2 * self.mu * d_vz_dz)
        self.tau_xz = self.tau_xz + self.dt * self.mu * (d_vx_dz + d_vz_dx)

        if source_term_vx is not None:
            self.vx = self.vx + source_term_vx
        if source_term_vz is not None:
            self.vz = self.vz + source_term_vz

    def _inject_source(self, source_term, coords):
        """
        Injeta um termo fonte em coordenadas do MODELO FÍSICO.
        Converte internamente para o grid estendido.
        """
        nz_ext, nx_ext = self.nz_ext, self.nx_ext
        field = torch.zeros((nz_ext, nx_ext), device=self.device)

        # Converte coordenadas do modelo físico para o grid estendido
        x = coords[:, 0] + self.pml_width
        z = coords[:, 1] + self.pml_width

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

        x0 = x0.clamp(0, nx_ext - 1)
        x1 = x1.clamp(0, nx_ext - 1)
        z0 = z0.clamp(0, nz_ext - 1)
        z1 = z1.clamp(0, nz_ext - 1)

        idx = z0 * nx_ext + x0
        field.view(-1).index_add_(0, idx, source_term * w00)
        idx = z0 * nx_ext + x1
        field.view(-1).index_add_(0, idx, source_term * w01)
        idx = z1 * nx_ext + x0
        field.view(-1).index_add_(0, idx, source_term * w10)
        idx = z1 * nx_ext + x1
        field.view(-1).index_add_(0, idx, source_term * w11)

        return field

    def _record_receivers(self, field, coords):
        """
        Registra o campo em coordenadas do MODELO FÍSICO.
        Converte internamente para o grid estendido.
        """
        nx_ext = self.nx_ext
        nz_ext = self.nz_ext

        # Coordenadas do modelo físico → grid estendido
        x = coords[:, 0] + self.pml_width
        z = coords[:, 1] + self.pml_width

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

        x0 = x0.clamp(0, nx_ext - 1)
        x1 = x1.clamp(0, nx_ext - 1)
        z0 = z0.clamp(0, nz_ext - 1)
        z1 = z1.clamp(0, nz_ext - 1)

        f00 = field[z0, x0]
        f01 = field[z0, x1]
        f10 = field[z1, x0]
        f11 = field[z1, x1]

        return f00 * w00 + f01 * w01 + f10 * w10 + f11 * w11

    def get_wavefield_physical(self):
        """
        Retorna o wavefield de pressão (tau_xx + tau_zz) restrito
        ao MODELO FÍSICO (sem a PML). Útil para visualização.

        Returns:
            tensor (nz_phys, nx_phys)
        """
        p = self.tau_xx + self.tau_zz
        w = self.pml_width
        return p[w:w + self.nz_phys, w:w + self.nx_phys]

    def forward(self, survey, wavelet):
        """
        Executa a modelagem direta.

        Retorna:
            data: tensor (n_sources, n_receivers, nt)
        """
        n_sources = len(survey.sources)
        n_receivers = len(survey.receivers)

        t = survey.t.to(self.device)
        w = wavelet(t)

        source_coords = survey.source_coords.to(self.device)
        receiver_coords = survey.receiver_coords.to(self.device)

        data = torch.zeros((n_sources, n_receivers, self.nt), device=self.device)

        for i in range(n_sources):
            self._init_fields()
            self.pml.reset_memory()

            src_coord = source_coords[i:i + 1]

            for it in range(self.nt):
                source_amplitude = w[it].expand(1)
                source_field = self._inject_source(source_amplitude, src_coord)
                self._step(source_term_vx=source_field, source_term_vz=source_field)

                pressure = self.tau_xx + self.tau_zz
                rec_values = self._record_receivers(pressure, receiver_coords)
                data[i, :, it] = rec_values

        return data