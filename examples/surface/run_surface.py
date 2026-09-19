# examples/surface/run_surface.py
import torch
import matplotlib.pyplot as plt
from seistomo.model.elastic import ElasticModel
from seistomo.acquisition.survey import SurfaceSurvey
from seistomo.acquisition.source import Ricker
from seistomo.physics.elastic.elastic2d import Elastic2D

# 1. Cria um modelo homogêneo simples
device = 'cuda' if torch.cuda.is_available() else 'cpu'
nz, nx = 200, 400
vp = torch.ones(nz, nx) * 2000.0
vs = torch.ones(nz, nx) * 1000.0
rho = torch.ones(nz, nx) * 2000.0

model = ElasticModel(vp=vp, vs=vs, rho=rho, dx=10.0, dz=10.0)

# 2. Define a aquisição (Surface)
source_x = [50, 150, 250, 350]
receiver_x = list(range(10, 390, 10))
survey = SurfaceSurvey(source_x=source_x, receiver_x=receiver_x, z=0.0, dt=0.001, nt=1000)

# 3. Cria o solver de produção
solver = Elastic2D(
    model=model,
    dt=survey.dt,
    nt=survey.nt,
    pml_width=20,
    fd_order=4,
    device=device
)

# 4. Executa o forward modeling
print("Iniciando forward modeling...")
data = solver.forward(survey=survey, wavelet=Ricker(f0=25))
print(f"Shape dos dados sísmicos: {data.shape}")

# 5. Plota um shot record (para o primeiro tiro)
plt.figure(figsize=(12, 8))
plt.imshow(data[0, :, :].T.cpu().numpy(), aspect='auto', cmap='seismic', 
           extent=[0, len(receiver_x), survey.nt*survey.dt, 0])
plt.title("Shot Record - Surface Survey (Primeiro Tiro)")
plt.xlabel("Receptor Index")
plt.ylabel("Tempo (s)")
plt.colorbar(label="Amplitude")
plt.show()

# 6. Teste de autodiff (gradiente)
print("Testando autodiff...")
vp_tensor = model.vp.clone().detach().requires_grad_(True)
model_test = ElasticModel(vp=vp_tensor, vs=model.vs, rho=model.rho, dx=model.dx, dz=model.dz)
solver_test = Elastic2D(model=model_test, dt=survey.dt, nt=survey.nt, device=device)
data_test = solver_test.forward(survey=survey, wavelet=Ricker(f0=25))

loss = data_test.sum()
loss.backward()

print(f"Gradiente de Vp calculado com sucesso! Shape: {vp_tensor.grad.shape}")
print(f"Norma do gradiente: {vp_tensor.grad.norm():.4f}")