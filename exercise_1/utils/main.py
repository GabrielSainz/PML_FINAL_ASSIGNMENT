import time
from pathlib import Path
import torch

# --- set paths
CKPT_DIR = Path("./mnist_diffusion_project/checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# -------------------------
# IMPORTANT: correct GPU timing
# -------------------------
def time_it(fn):
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return out, (time.perf_counter() - t0)

@torch.no_grad()
def batched_sample(sample_fn, n=5000, batch=256):
    outs = []
    for i in range(0, n, batch):
        b = min(batch, n - i)
        outs.append(sample_fn(b).cpu())
    return torch.cat(outs, dim=0)

# -------------------------
# LOAD MODELS (must match your architectures/hparams)
# -------------------------
BASE_T, BASE_B1, BASE_BT = 1000, 1e-4, 2e-2

# Baseline DDPM
payload = torch.load(CKPT_DIR/"baseline_ddpm_ema.pt", map_location="cpu")
unet = ScoreNet(marginal_prob_std=lambda t: torch.ones_like(t).to(DEVICE))
baseline = DDPM(unet, T=BASE_T, beta_1=BASE_B1, beta_T=BASE_BT).to(DEVICE)
baseline.load_state_dict(payload["ema_state_dict"])
baseline.eval()

# Flow matching
payload = torch.load(CKPT_DIR/"flow_matching.pt", map_location="cpu")
unet = ScoreNet(marginal_prob_std=lambda t: torch.ones_like(t).to(DEVICE))
flow = RectifiedFlow(unet).to(DEVICE)
flow.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)  # handles either format
flow.eval()

# Autoencoder (needed for latent decode)
payload = torch.load(CKPT_DIR/"mnist_autoencoder.pt", map_location="cpu")
ae = ConvAutoencoder(latent_c=16).to(DEVICE)
ae.load_state_dict(payload["state_dict"])
ae.eval()

# Latent DDPM
payload = torch.load(CKPT_DIR/"latent_ddpm_ema.pt", map_location="cpu")
net = ScoreNetPad(
    marginal_prob_std=lambda t: torch.ones_like(t).to(DEVICE),
    in_channels=16, channels=(64,128,256,256), embed_dim=256
).to(DEVICE)
latent = DDPM2D(net, C=16, H=4, W=4, T=1000, beta_1=1e-4, beta_T=2e-2).to(DEVICE)
latent.load_state_dict(payload["ema_state_dict"])
latent.eval()

# -------------------------
# TIMED SAMPLING (5000 samples)
# -------------------------
N = 5000
B = 256

# warmup (important for fair timing)
_ = baseline.sample((16, 28*28))
_ = flow.sample((16, 28*28), steps=100, solver="heun")
_ = latent.sample((16, 16*4*4))

# baseline time
_, t_base = time_it(lambda: batched_sample(lambda b: baseline.sample((b, 28*28)), n=N, batch=B))
print(f"Baseline DDPM sampling: {t_base:.3f} s for {N} samples")

# flow time
steps, solver = 100, "heun"
_, t_flow = time_it(lambda: batched_sample(lambda b: flow.sample((b, 28*28), steps=steps, solver=solver), n=N, batch=B))
print(f"Flow matching sampling: {t_flow:.3f} s for {N} samples  (steps={steps}, solver={solver})")

# latent time (includes decode)
D = 16*4*4
def latent_decode_batch(b):
    z = latent.sample((b, D)).view(b, 16, 4, 4)
    x = ae.decode(z)  # [-1,1]
    return x

_, t_lat = time_it(lambda: batched_sample(latent_decode_batch, n=N, batch=B))
print(f"Latent DDPM (+decode) sampling: {t_lat:.3f} s for {N} samples")


