import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t: [B] int64
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, dim]
        return emb


class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim, time_dim=128, hidden=256, depth=3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        layers = []
        in_dim = latent_dim + time_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t):
        # z_t: [B, d], t: [B]
        t_emb = self.time_mlp(t)
        x = torch.cat([z_t, t_emb], dim=1)
        return self.net(x)  # predicted eps


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    acp = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return betas.clamp(1e-4, 0.999)

def extract(a, t, shape):
    # a: [T], t: [B] -> [B, 1, ...] for broadcasting
    out = a.gather(0, t)
    return out.view(t.size(0), *([1] * (len(shape) - 1)))


class LatentDDPM:
    def __init__(self, denoiser, latent_dim, T=200, device="cuda"):
        self.denoiser = denoiser
        self.latent_dim = latent_dim
        self.T = T
        self.device = device

        betas = cosine_beta_schedule(T).to(device)         # [T]
        alphas = 1.0 - betas                               # [T]
        acp = torch.cumprod(alphas, dim=0)                 # [T]
        acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.acp_prev = acp_prev

        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)

        # posterior variance (DDPM)
        self.posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)

    def q_sample(self, z0, t, eps=None):
        # z_t = sqrt(acp_t) z0 + sqrt(1-acp_t) eps
        if eps is None:
            eps = torch.randn_like(z0)
        return extract(self.sqrt_acp, t, z0.shape) * z0 + extract(self.sqrt_one_minus_acp, t, z0.shape) * eps, eps

    def p_mean(self, z_t, t):
        # mu_theta(z_{t-1} | z_t) using eps prediction
        eps_pred = self.denoiser(z_t, t)
        beta_t = extract(self.betas, t, z_t.shape)
        alpha_t = extract(self.alphas, t, z_t.shape)
        acp_t = extract(self.acp, t, z_t.shape)

        mean = (1.0 / torch.sqrt(alpha_t)) * (z_t - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_pred)
        return mean

    @torch.no_grad()
    def p_sample(self, z_t, t):
        mean = self.p_mean(z_t, t)
        var = extract(self.posterior_var, t, z_t.shape)
        noise = torch.randn_like(z_t)

        # no noise at t=0
        nonzero = (t != 0).float().view(-1, *([1] * (z_t.dim() - 1)))
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n):
        z = torch.randn(n, self.latent_dim, device=self.device)  # start from N(0,I)
        for tt in reversed(range(self.T)):
            t = torch.full((n,), tt, device=self.device, dtype=torch.long)
            z = self.p_sample(z, t)
        return z

    def training_loss(self, z0):
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=z0.device, dtype=torch.long)
        z_t, eps = self.q_sample(z0, t)
        eps_pred = self.denoiser(z_t, t)
        return F.mse_loss(eps_pred, eps)

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def train_diffusion(latent_path, vae, device,
                    T=200, epochs=20, batch_size=512, lr=2e-4,
                    hidden=256, depth=3):
    # latent dataset
    ds = LatentDataset(latent_path, scaled=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    d = ds.z.size(1)
    denoiser = LatentDenoiser(latent_dim=d, hidden=hidden, depth=depth).to(device)
    ddpm = LatentDDPM(denoiser, latent_dim=d, T=T, device=device)
    opt = torch.optim.Adam(denoiser.parameters(), lr=lr)

    losses = []
    denoiser.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        for z0, _ in dl:
            z0 = z0.to(device)
            opt.zero_grad()
            loss = ddpm.training_loss(z0)
            loss.backward()
            opt.step()
            running += loss.item()

        avg = running / len(dl)
        losses.append(avg)
        print(f"[Diffusion] epoch {ep:03d}/{epochs}  loss={avg:.6f}")

    # plot diffusion loss curve
    plt.figure()
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("MSE (eps prediction)")
    plt.title("Latent diffusion training loss")
    plt.show()

    return ddpm, ds  # return dataset too (for mean/std)


@torch.no_grad()
def generate_images_from_diffusion(ddpm, vae, latent_ds, device, n=25):
    vae.eval()

    # 1) sample in scaled latent space (mean 0 std 1)
    z_scaled = ddpm.sample(n)  # [n, d], on device

    # 2) unscale back to VAE latent space
    mean = latent_ds.mean.to(device)
    std = latent_ds.std.to(device)
    z = z_scaled * std + mean

    # 3) decode to images
    imgs = vae.decode(z).cpu()  # [n,1,28,28]

    # plot
    cols = int(math.sqrt(n))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
    axes = axes.flatten()
    for i in range(rows*cols):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(imgs[i,0], cmap="gray")
    plt.suptitle("Generated samples (latent diffusion → VAE decode)")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_latent_scatter_2d(latent_ds, ddpm=None, n_samples=2000):
    if latent_ds.z.size(1) != 2:
        print("latent_dim != 2, skipping scatter.")
        return

    z = latent_ds.z  # unscaled stored latents_mu
    y = latent_ds.y

    # subsample
    idx = torch.randperm(z.size(0))[:n_samples]
    z_sub = z[idx]
    y_sub = y[idx]

    plt.figure()
    plt.scatter(z_sub[:,0], z_sub[:,1], s=5, alpha=0.5, c=y_sub, cmap="tab10")
    plt.colorbar()
    plt.title("μ latents (unscaled) colored by label")
    plt.xlabel("z1")
    plt.ylabel("z2")
    plt.show()

    # diffusion samples (in scaled space) projected after unscale
    if ddpm is not None:
        z_scaled = ddpm.sample(1000).cpu()
        z_gen = z_scaled * latent_ds.std + latent_ds.mean
        plt.figure()
        plt.scatter(z_gen[:,0], z_gen[:,1], s=5, alpha=0.5)
        plt.title("Diffusion samples in latent space (unscaled)")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.show()

@torch.no_grad()
def plot_latent_grid_decode_2d(vae, latent_ds, device, grid=20, lim=3.0):
    if latent_ds.z.size(1) != 2:
        print("latent_dim != 2, skipping grid decode.")
        return

    vae.eval()
    # We grid in *scaled* space, then unscale, then decode
    xs = torch.linspace(-lim, lim, grid)
    ys = torch.linspace(-lim, lim, grid)

    canvas = torch.zeros(grid*28, grid*28)

    mean = latent_ds.mean.to(device)
    std = latent_ds.std.to(device)

    for i, y in enumerate(ys.flip(0)):
        for j, x in enumerate(xs):
            z_scaled = torch.tensor([[x, y]], device=device)
            z = z_scaled * std + mean
            img = vae.decode(z)[0,0].cpu()
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = img

    plt.figure(figsize=(8,8))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.title("Latent grid decode (scaled grid → unscale → decode)")
    plt.show()
