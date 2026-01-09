# diffusion.py
"""
Latent diffusion priors for a trained VAE:
- Discrete DDPM prior over z (vector latents)
- Continuous-time VP-SDE prior over z (probability-flow ODE sampling)

Folder layout (per run):
runs_diffusion/<run_name>/
  config.json
  models/
    ddpm.pt
    cont.pt
  plots/
    losses.pdf
    samples_ddpm.pdf
    samples_cont.pdf
    latent_scatter_ddpm.pdf (if latent_dim==2)
    latent_scatter_cont.pdf (if latent_dim==2)
    latent_grid_decode.pdf   (if latent_dim==2)
  latents/
    mnist_latents_mu.pt
"""

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.latent_utils import build_latent_dataset, LatentDataset, load_latent_checkpoint
from utils.VAE import BetaVAE  # model class only (decoder used for plots)


# =========================
# Config + IO
# =========================
@dataclass
class DiffusionConfig:
    # general
    seed: int = 42
    run_root: str = "./runs_diffusion"
    run_name: Optional[str] = None

    # data
    batch_size: int = 512
    num_workers: int = 2
    latent_scaled: bool = True  # use standardized z for training

    # denoiser net
    time_dim: int = 128
    hidden: int = 256
    depth: int = 3

    # training
    epochs: int = 30
    lr: float = 2e-4
    grad_clip_norm: Optional[float] = None
    ema_decay: float = 0.999

    # DDPM (discrete)
    ddpm_T: int = 200  # timesteps

    # Continuous (VP-SDE)
    beta_min: float = 0.1
    beta_max: float = 20.0

    # sampling + plots
    n_plot: int = 25
    plot_every_epochs: int = 0  # 0 = only at end
    sample_steps_cont: int = 200  # Euler steps for probability-flow ODE
    sample_steps_ddpm: Optional[int] = None  # None -> use ddpm_T
    latent_scatter_max_points: int = 5000
    latent_grid_size: int = 20
    latent_grid_lim: float = 3.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_dump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def make_run_dir(run_root: str, run_name: Optional[str] = None) -> str:
    _ensure_dir(run_root)
    if run_name is None:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_root, run_name)
    _ensure_dir(run_dir)
    _ensure_dir(os.path.join(run_dir, "models"))
    _ensure_dir(os.path.join(run_dir, "plots"))
    _ensure_dir(os.path.join(run_dir, "latents"))
    return run_dir


# =========================
# Load VAE from run folder
# =========================
def load_vae_from_run(run_dir: str, device: torch.device) -> BetaVAE:
    """
    Expects run_dir/vae_model.pt.
    Supports:
    - checkpoint dict with 'model_state' and 'latent_dim'
    - raw state_dict (fallback, tries config.json for latent_dim)
    """
    ckpt_path = os.path.join(run_dir, "vae_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        latent_dim = int(ckpt.get("latent_dim", ckpt.get("cfg", {}).get("latent_dim", 16)))
        model = BetaVAE(latent_dim=latent_dim).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model

    # raw state_dict fallback
    cfg_path = os.path.join(run_dir, "config.json")
    latent_dim = 16
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        latent_dim = int(cfg.get("latent_dim", 16))

    model = BetaVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt)
    model.eval()
    return model


# =========================
# EMA helper
# =========================
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in msd.items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=(1 - self.decay))

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)


# =========================
# Time embedding + denoiser
# =========================
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float or int
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb


class LatentDenoiser(nn.Module):
    """
    Predicts eps given (z_t, t).
    Works for both:
    - DDPM with integer t in [0, T-1]
    - continuous with float t in [0,1]
    """
    def __init__(self, latent_dim: int, time_dim: int = 128, hidden: int = 256, depth: int = 3):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
        )

        layers: List[nn.Module] = []
        in_dim = latent_dim + time_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(in_dim, hidden), nn.SiLU()]
            in_dim = hidden
        layers += [nn.Linear(in_dim, latent_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(t)
        x = torch.cat([z_t, t_emb], dim=1)
        return self.net(x)


# =========================
# DDPM (discrete)
# =========================
def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    acp = torch.cos(((x / T) + s) / (1 + s) * math.pi / 2) ** 2
    acp = acp / acp[0]
    betas = 1 - (acp[1:] / acp[:-1])
    return betas.clamp(1e-4, 0.999)


def _extract(a: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    out = a.gather(0, t)
    return out.view(t.size(0), *([1] * (len(shape) - 1)))


class LatentDDPM:
    def __init__(self, denoiser: nn.Module, latent_dim: int, T: int, device: torch.device):
        self.denoiser = denoiser
        self.latent_dim = latent_dim
        self.T = T
        self.device = device

        betas = cosine_beta_schedule(T).to(device)
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, dim=0)
        acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.acp = acp
        self.acp_prev = acp_prev

        self.sqrt_acp = torch.sqrt(acp)
        self.sqrt_one_minus_acp = torch.sqrt(1.0 - acp)
        self.posterior_var = betas * (1.0 - acp_prev) / (1.0 - acp)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(z0)
        zt = _extract(self.sqrt_acp, t, z0.shape) * z0 + _extract(self.sqrt_one_minus_acp, t, z0.shape) * eps
        return zt, eps

    def training_loss(self, z0: torch.Tensor) -> torch.Tensor:
        B = z0.size(0)
        t = torch.randint(0, self.T, (B,), device=z0.device, dtype=torch.long)
        zt, eps = self.q_sample(z0, t)
        eps_pred = self.denoiser(zt, t)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def p_mean(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps_pred = self.denoiser(zt, t)
        beta_t = _extract(self.betas, t, zt.shape)
        alpha_t = _extract(self.alphas, t, zt.shape)
        acp_t = _extract(self.acp, t, zt.shape)
        mean = (1.0 / torch.sqrt(alpha_t)) * (zt - (beta_t / torch.sqrt(1.0 - acp_t)) * eps_pred)
        return mean

    @torch.no_grad()
    def p_sample(self, zt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        mean = self.p_mean(zt, t)
        var = _extract(self.posterior_var, t, zt.shape)
        noise = torch.randn_like(zt)
        nonzero = (t != 0).float().view(-1, *([1] * (zt.dim() - 1)))
        return mean + nonzero * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, n: int, steps: Optional[int] = None) -> torch.Tensor:
        if steps is None:
            steps = self.T
        # If steps < T, we still use schedule indices; simplest: just run full T for now
        z = torch.randn(n, self.latent_dim, device=self.device)
        for tt in reversed(range(self.T)):
            t = torch.full((n,), tt, device=self.device, dtype=torch.long)
            z = self.p_sample(z, t)
        return z


# =========================
# Continuous diffusion (VP-SDE)
# =========================
class VPSDE:
    """
    VP-SDE with linear beta(t) schedule:
      beta(t) = beta_min + t (beta_max - beta_min), t in [0,1]
    Forward marginal:
      z_t = alpha(t) z0 + sigma(t) eps
      alpha(t)=exp(-0.5 * int_0^t beta(s) ds)
      sigma(t)=sqrt(1 - exp(-int_0^t beta(s) ds))
    We train eps-prediction: eps_hat(z_t, t).
    Sampling: probability flow ODE with Euler steps from t=1 -> 0.
    """
    def __init__(self, denoiser: nn.Module, latent_dim: int, beta_min: float, beta_max: float, device: torch.device):
        self.denoiser = denoiser
        self.latent_dim = latent_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.device = device

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def beta_bar(self, t: torch.Tensor) -> torch.Tensor:
        # ∫0^t beta(s) ds for linear beta
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * self.beta_bar(t))

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(1.0 - torch.exp(-self.beta_bar(t)))

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if eps is None:
            eps = torch.randn_like(z0)
        a = self.alpha(t).view(-1, 1)
        s = self.sigma(t).view(-1, 1)
        zt = a * z0 + s * eps
        return zt, eps

    def training_loss(self, z0: torch.Tensor) -> torch.Tensor:
        B = z0.size(0)
        t = torch.rand(B, device=z0.device)  # uniform in [0,1]
        zt, eps = self.q_sample(z0, t)
        eps_pred = self.denoiser(zt, t)
        return F.mse_loss(eps_pred, eps)

    @torch.no_grad()
    def sample(self, n: int, steps: int = 200) -> torch.Tensor:
        # Start at t=1 from N(0,I) in the *training space* (scaled z)
        z = torch.randn(n, self.latent_dim, device=self.device)
        dt = -1.0 / steps
        for i in range(steps):
            t = torch.full((n,), 1.0 + i * dt, device=self.device)  # from 1 -> 0
            t = t.clamp(0.0, 1.0)

            beta_t = self.beta(t).view(-1, 1)
            sigma_t = self.sigma(t).view(-1, 1).clamp(min=1e-5)

            eps_pred = self.denoiser(z, t)  # eps_hat
            # probability flow ODE drift for VP-SDE:
            # dz = [-0.5 beta z - 0.5 beta * score] dt
            # score = -eps/sigma  => -0.5 beta * score = +0.5 beta * eps/sigma
            drift = -0.5 * beta_t * z + 0.5 * beta_t * (eps_pred / sigma_t)
            z = z + drift * dt

        return z


# =========================
# Plot helpers
# =========================
def _save_pdf(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_image_grid_pdf(images: torch.Tensor, path: str, title: str = "", n: int = 25) -> None:
    """
    images: [N,1,28,28] on CPU in [0,1]
    """
    n = min(n, images.size(0))
    cols = int(math.sqrt(n))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(images[i, 0], cmap="gray")
    fig.suptitle(title)
    fig.tight_layout()
    _save_pdf(fig, path)


@torch.no_grad()
def save_latent_scatter_2d_pdf(
    z_real_unscaled: torch.Tensor,
    z_gen_unscaled: torch.Tensor,
    path: str,
    title: str,
    max_points: int = 5000,
) -> None:
    z_real = z_real_unscaled[:max_points].cpu()
    z_gen = z_gen_unscaled[:max_points].cpu()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(z_real[:, 0], z_real[:, 1], s=6, alpha=0.35, label="real μ")
    ax.scatter(z_gen[:, 0], z_gen[:, 1], s=6, alpha=0.35, label="generated")
    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.legend()
    fig.tight_layout()
    _save_pdf(fig, path)


@torch.no_grad()
def save_latent_grid_decode_2d_pdf(
    vae: BetaVAE,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    path: str,
    grid_size: int = 20,
    lim: float = 3.0,
    title: str = "Latent grid decode (scaled grid -> unscale -> decode)",
) -> None:
    if vae.latent_dim != 2:
        return

    vae.eval()
    xs = torch.linspace(-lim, lim, grid_size)
    ys = torch.linspace(-lim, lim, grid_size)

    canvas = torch.zeros(grid_size * 28, grid_size * 28)

    mean = mean.to(device)
    std = std.to(device)

    for i, y in enumerate(ys.flip(0)):
        for j, x in enumerate(xs):
            z_scaled = torch.tensor([[x, y]], device=device)
            z = z_scaled * std + mean
            img = vae.decode(z)[0, 0].detach().cpu()
            canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = img

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    _save_pdf(fig, path)


# =========================
# Training loops
# =========================
def _train_epoch_ddpm(ddpm: LatentDDPM, loader: torch.utils.data.DataLoader, opt: torch.optim.Optimizer,
                      ema: EMA, device: torch.device, grad_clip_norm: Optional[float] = None) -> float:
    ddpm.denoiser.train()
    running = 0.0
    for z0, _ in loader:
        z0 = z0.to(device)
        opt.zero_grad(set_to_none=True)
        loss = ddpm.training_loss(z0)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(ddpm.denoiser.parameters(), grad_clip_norm)
        opt.step()
        ema.update(ddpm.denoiser)
        running += float(loss.item())
    return running / len(loader)


def _train_epoch_cont(sde: VPSDE, loader: torch.utils.data.DataLoader, opt: torch.optim.Optimizer,
                      ema: EMA, device: torch.device, grad_clip_norm: Optional[float] = None) -> float:
    sde.denoiser.train()
    running = 0.0
    for z0, _ in loader:
        z0 = z0.to(device)
        opt.zero_grad(set_to_none=True)
        loss = sde.training_loss(z0)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(sde.denoiser.parameters(), grad_clip_norm)
        opt.step()
        ema.update(sde.denoiser)
        running += float(loss.item())
    return running / len(loader)


@torch.no_grad()
def _decode_generated_latents(
    vae: BetaVAE,
    z_scaled: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    z_scaled: [N,d] in standardized latent space.
    Return decoded images [N,1,28,28] on CPU in [0,1].
    """
    mean = mean.to(device)
    std = std.to(device)
    z = z_scaled * std + mean
    imgs = vae.decode(z).detach().cpu()
    return imgs


def train_latent_priors(
    vae_run_dir: str,
    train_loader_for_latents: torch.utils.data.DataLoader,
    device: torch.device,
    cfg: DiffusionConfig,
) -> Tuple[str, Dict[str, Any]]:
    """
    End-to-end:
    - load VAE
    - build latent dataset (mu + mean/std)
    - train DDPM + continuous VP-SDE prior
    - save models + plots
    Returns (run_dir, metrics dict)
    """
    set_seed(cfg.seed)
    run_dir = make_run_dir(cfg.run_root, cfg.run_name)

    _json_dump(os.path.join(run_dir, "config.json"), {**asdict(cfg), "vae_run_dir": vae_run_dir})

    plots_dir = os.path.join(run_dir, "plots")
    models_dir = os.path.join(run_dir, "models")
    latents_dir = os.path.join(run_dir, "latents")

    # ---- load VAE (frozen)
    vae = load_vae_from_run(vae_run_dir, device=device)
    vae.eval()

    # ---- build latent dataset inside this run folder
    latent_path = os.path.join(latents_dir, "mnist_latents_mu.pt")
    build_latent_dataset(vae, train_loader_for_latents, device, save_path=latent_path)
    latent_ckpt = load_latent_checkpoint(latent_path)
    mean = latent_ckpt["mean"]
    std = latent_ckpt["std"]
    latent_dim = int(latent_ckpt["meta"]["latent_dim"])

    # ---- dataset for training priors
    latent_ds = LatentDataset(latent_path, scaled=cfg.latent_scaled)
    latent_dl = torch.utils.data.DataLoader(
        latent_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ---- DDPM
    denoiser_ddpm = LatentDenoiser(latent_dim, cfg.time_dim, cfg.hidden, cfg.depth).to(device)
    ddpm = LatentDDPM(denoiser_ddpm, latent_dim=latent_dim, T=cfg.ddpm_T, device=device)
    opt_ddpm = torch.optim.Adam(denoiser_ddpm.parameters(), lr=cfg.lr)
    ema_ddpm = EMA(denoiser_ddpm, decay=cfg.ema_decay)

    # ---- Continuous VP-SDE
    denoiser_cont = LatentDenoiser(latent_dim, cfg.time_dim, cfg.hidden, cfg.depth).to(device)
    cont = VPSDE(denoiser_cont, latent_dim=latent_dim, beta_min=cfg.beta_min, beta_max=cfg.beta_max, device=device)
    opt_cont = torch.optim.Adam(denoiser_cont.parameters(), lr=cfg.lr)
    ema_cont = EMA(denoiser_cont, decay=cfg.ema_decay)

    # ---- train both
    hist = {"ddpm_loss": [], "cont_loss": []}

    for ep in range(1, cfg.epochs + 1):
        ddpm_loss = _train_epoch_ddpm(ddpm, latent_dl, opt_ddpm, ema_ddpm, device, cfg.grad_clip_norm)
        cont_loss = _train_epoch_cont(cont, latent_dl, opt_cont, ema_cont, device, cfg.grad_clip_norm)

        hist["ddpm_loss"].append(ddpm_loss)
        hist["cont_loss"].append(cont_loss)

        print(f"[Latent priors] epoch {ep:03d}/{cfg.epochs} | ddpm_loss={ddpm_loss:.6f} | cont_loss={cont_loss:.6f}")

        # optional intermediate sample plots
        if cfg.plot_every_epochs and (ep % cfg.plot_every_epochs == 0):
            # use EMA weights for sampling
            tmp_ddpm = LatentDenoiser(latent_dim, cfg.time_dim, cfg.hidden, cfg.depth).to(device)
            tmp_ddpm.load_state_dict(denoiser_ddpm.state_dict())
            ema_ddpm.copy_to(tmp_ddpm)
            ddpm_ema = LatentDDPM(tmp_ddpm, latent_dim, cfg.ddpm_T, device)

            tmp_cont = LatentDenoiser(latent_dim, cfg.time_dim, cfg.hidden, cfg.depth).to(device)
            tmp_cont.load_state_dict(denoiser_cont.state_dict())
            ema_cont.copy_to(tmp_cont)
            cont_ema = VPSDE(tmp_cont, latent_dim, cfg.beta_min, cfg.beta_max, device)

            z_ddpm = ddpm_ema.sample(cfg.n_plot)
            z_cont = cont_ema.sample(cfg.n_plot, steps=cfg.sample_steps_cont)

            imgs_ddpm = _decode_generated_latents(vae, z_ddpm, mean, std, device)
            imgs_cont = _decode_generated_latents(vae, z_cont, mean, std, device)

            save_image_grid_pdf(imgs_ddpm, os.path.join(plots_dir, f"samples_ddpm_ep{ep:03d}.pdf"),
                                title=f"DDPM samples (epoch {ep})", n=cfg.n_plot)
            save_image_grid_pdf(imgs_cont, os.path.join(plots_dir, f"samples_cont_ep{ep:03d}.pdf"),
                                title=f"Continuous samples (epoch {ep})", n=cfg.n_plot)

    # ---- save losses plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["ddpm_loss"], label="ddpm loss")
    ax.plot(hist["cont_loss"], label="cont loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE(eps)")
    ax.set_title("Latent prior training losses")
    ax.legend()
    fig.tight_layout()
    _save_pdf(fig, os.path.join(plots_dir, "losses.pdf"))

    # ---- save EMA checkpoints
    ema_ddpm.copy_to(denoiser_ddpm)
    ema_cont.copy_to(denoiser_cont)

    torch.save(
        {"state_dict": denoiser_ddpm.state_dict(), "cfg": asdict(cfg), "latent_dim": latent_dim},
        os.path.join(models_dir, "ddpm.pt"),
    )
    torch.save(
        {"state_dict": denoiser_cont.state_dict(), "cfg": asdict(cfg), "latent_dim": latent_dim},
        os.path.join(models_dir, "cont.pt"),
    )

    # ---- final samples
    z_ddpm = ddpm.sample(cfg.n_plot)
    z_cont = cont.sample(cfg.n_plot, steps=cfg.sample_steps_cont)

    imgs_ddpm = _decode_generated_latents(vae, z_ddpm, mean, std, device)
    imgs_cont = _decode_generated_latents(vae, z_cont, mean, std, device)

    save_image_grid_pdf(imgs_ddpm, os.path.join(plots_dir, "samples_ddpm.pdf"),
                        title="Generated samples (DDPM latent prior -> VAE decode)", n=cfg.n_plot)
    save_image_grid_pdf(imgs_cont, os.path.join(plots_dir, "samples_cont.pdf"),
                        title="Generated samples (Continuous VP-SDE prior -> VAE decode)", n=cfg.n_plot)

    # ---- 2D-specific plots
    if latent_dim == 2:
        z_real_unscaled = latent_ckpt["z_mu"]  # [N,2] (unscaled mu)
        # generated in scaled space -> unscale
        z_ddpm_unscaled = (z_ddpm.detach().cpu() * std) + mean
        z_cont_unscaled = (z_cont.detach().cpu() * std) + mean

        save_latent_scatter_2d_pdf(
            z_real_unscaled, z_ddpm_unscaled,
            os.path.join(plots_dir, "latent_scatter_ddpm.pdf"),
            title="Latent scatter (real μ vs DDPM samples)",
            max_points=cfg.latent_scatter_max_points,
        )
        save_latent_scatter_2d_pdf(
            z_real_unscaled, z_cont_unscaled,
            os.path.join(plots_dir, "latent_scatter_cont.pdf"),
            title="Latent scatter (real μ vs Continuous samples)",
            max_points=cfg.latent_scatter_max_points,
        )

        save_latent_grid_decode_2d_pdf(
            vae, mean, std, device,
            os.path.join(plots_dir, "latent_grid_decode.pdf"),
            grid_size=cfg.latent_grid_size,
            lim=cfg.latent_grid_lim,
        )

    # ---- save history
    _json_dump(os.path.join(run_dir, "history.json"), hist)

    metrics = {
        "run_dir": run_dir,
        "latent_dim": latent_dim,
        "latent_path": latent_path,
    }
    return run_dir, metrics
