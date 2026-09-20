# diag_two_layer.py
import torch
import matplotlib.pyplot as plt
from seistomo.acquisition import SurfaceSurvey, Ricker
from seistomo.inversion import FWI
from seistomo.misfit import L2Misfit
from seistomo.model import ElasticModel
from seistomo.physics.elastic import Elastic2D

# --- setup igual ao two_layer_setup ---
nz, nx = 40, 40
dx = dz = 10.0
nt = 400
dt = 0.001
pml_width = 10

vp_true = torch.full((nz, nx), 2000.0)
vp_true[nz // 2:, :] = 2500.0
vp_init = torch.full((nz, nx), 2200.0)

vs_true = vp_true / 1.732
rho_true = 1000.0 + 0.3 * vp_true

survey = SurfaceSurvey(
    source_x=[nx / 2],
    receiver_x=list(torch.linspace(2, nx - 3, 15).tolist()),
    z=2.0, dt=dt, nt=nt,
)
wavelet = Ricker(f0=15.0)

# --- d_obs do modelo verdadeiro ---
model_true = ElasticModel(vp_true, vs_true, rho_true, dx, dz)
solver_true = Elastic2D(model_true, dt, nt, pml_width, fd_order=8, device="cpu")
with torch.no_grad():
    d_obs = solver_true.forward(survey, wavelet)

# --- d_syn do modelo inicial ---
model_init = ElasticModel(vp_init, vs_true, rho_true, dx, dz)
solver_init = Elastic2D(model_init, dt, nt, pml_width, fd_order=8, device="cpu")
with torch.no_grad():
    d_syn_init = solver_init.forward(survey, wavelet)

# --- diagnóstico ---
print("d_obs range:", d_obs.min().item(), d_obs.max().item())
print("d_syn range:", d_syn_init.min().item(), d_syn_init.max().item())
print("residual norm:", (d_obs - d_syn_init).norm().item())
print("d_obs norm:", d_obs.norm().item())
print("relative residual:", ((d_obs - d_syn_init).norm() / d_obs.norm()).item())

# Visualização: primeiro receptor
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
t = survey.t.numpy()
rec_idx = 7  # receptor do meio
axes[0].plot(t, d_obs[0, rec_idx].numpy(), label="d_obs (true)")
axes[0].plot(t, d_syn_init[0, rec_idx].numpy(), label="d_syn (init)", alpha=0.7)
axes[0].set_ylabel("Amplitude")
axes[0].legend()
axes[0].set_title(f"Receptor {rec_idx} — traço sísmico")

axes[1].plot(t, (d_obs[0, rec_idx] - d_syn_init[0, rec_idx]).numpy())
axes[1].set_ylabel("Residual")
axes[1].set_title("Residual (d_obs - d_syn)")

# Espectro
freqs = torch.fft.rfftfreq(nt, dt)
spec_obs = torch.fft.rfft(d_obs[0, rec_idx]).abs()
spec_syn = torch.fft.rfft(d_syn_init[0, rec_idx]).abs()
axes[2].semilogy(freqs.numpy(), spec_obs.numpy(), label="d_obs")
axes[2].semilogy(freqs.numpy(), spec_syn.numpy(), label="d_syn", alpha=0.7)
axes[2].set_xlabel("Frequency (Hz)")
axes[2].set_ylabel("Amplitude")
axes[2].legend()
axes[2].set_title("Espectro de amplitude")

plt.tight_layout()
plt.savefig("diag_two_layer.png", dpi=120)
print("\nSalvo: diag_two_layer.png")

# --- FWI com lr=10 (como Deepwave) ---
print("\n--- FWI com lr=10 (padrão Deepwave) ---")
solver_fwi = Elastic2D(
    ElasticModel(vp_init, vs_true, rho_true, dx, dz),
    dt, nt, pml_width, fd_order=8, device="cpu",
)
fwi = FWI(
    solver=solver_fwi, misfit=L2Misfit(),
    optimizer="adam", lr=10.0, max_iter=15, verbose=True, log_every=1,
)
result = fwi.run(
    survey=survey, wavelet=wavelet, data_observed=d_obs,
    vp_initial=vp_init, vs_fixed=vs_true, rho_fixed=rho_true,
)
print(f"\nFinal loss: {result.final_loss:.6f}")
print(f"Initial loss: {result.initial_loss:.6f}")
print(f"Reduction: {(1 - result.final_loss/result.initial_loss)*100:.1f}%")
print(f"Vp range final: [{result.vp.min().item():.1f}, {result.vp.max().item():.1f}]")