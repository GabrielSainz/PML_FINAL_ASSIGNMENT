# vae.py
"""
Beta-VAE utilities for MNIST (PyTorch).

What this module provides:
- BetaVAE model (conv encoder/decoder)
- beta-VAE loss (BCE-with-logits recon + beta * KL)
- training loop with checkpoint + config/history saving
- plot utilities saved as PDF (loss curves, reconstructions, prior samples, latent viz)

Typical usage (in a notebook):
    from vae import VAEConfig, BetaVAE, train_vae

    cfg = VAEConfig(latent_dim=16, beta=0.01, epochs=15, lr=2e-3, run_root="./runs_vae")
    model, history, run_dir = train_vae(cfg, dataloader_train, dataloader_val, device)
"""

from __future__ import annotations

import os
import json
import time
import math
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# =========================
# Config / IO helpers
# =========================
@dataclass
class VAEConfig:
    # training
    latent_dim: int = 16
    beta: float = 0.01
    epochs: int = 15
    lr: float = 2e-3
    seed: int = 42
    grad_clip_norm: Optional[float] = None  # e.g. 1.0

    # logging / saving
    run_root: str = "./runs_vae"
    run_name: Optional[str] = None          # if None -> auto timestamp
    save_best_only: bool = True

    # plots
    max_images_plot: int = 16
    prior_n_plot: int = 25
    latent_grid_size: int = 20
    latent_grid_lim: float = 3.0
    latent_scatter_max_points: int = 5000


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _json_dump(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def _text_dump(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def make_run_dir(run_root: str, run_name: Optional[str] = None) -> str:
    _ensure_dir(run_root)
    if run_name is None:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_root, run_name)
    plots_dir = os.path.join(run_dir, "plots")
    _ensure_dir(run_dir)
    _ensure_dir(plots_dir)
    return run_dir


# =========================
# Model
# =========================
class BetaVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> feature map -> flatten
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),      # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),     # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 1, 1),    # 7 -> 7
            nn.ReLU(inplace=True),
        )
        self.enc_out_dim = 128 * 7 * 7
        self.fc_mu = nn.Linear(self.enc_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_out_dim, latent_dim)

        # Decoder: latent -> feature map -> 1x28x28 logits
        self.fc_dec = nn.Linear(latent_dim, self.enc_out_dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 1, 1),    # 7 -> 7
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),     # 7 -> 14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),      # 14 -> 28
            # No sigmoid here because we use BCEWithLogitsLoss
        )

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.view(z.size(0), 128, 7, 7)
        logits = self.dec(h)
        return logits

    def forward(self, x: torch.Tensor, sample_latent: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample_latent else mu
        logits = self.decode_logits(z)
        return logits, mu, logvar, z

    @torch.no_grad()
    def encode_mu(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        mu, _ = self.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.decode_logits(z)
        return torch.sigmoid(logits)  # in [0,1]


# =========================
# Loss
# =========================
_bce_logits = nn.BCEWithLogitsLoss(reduction="none")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # KL(q(z|x)||N(0,I)) for diagonal Gaussian q
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1)  # per-sample


def beta_vae_loss(
    x: torch.Tensor,
    logits: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon = _bce_logits(logits, x)                 # [B,1,28,28]
    recon = recon.view(x.size(0), -1).sum(dim=1)   # per-sample
    kl = kl_divergence(mu, logvar)                 # per-sample
    loss = (recon + beta * kl).mean()
    return loss, recon.mean(), kl.mean()


# =========================
# Training loop
# =========================
def _run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    beta: float,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None,
) -> Tuple[float, float, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
    n_batches = 0

    for x, _ in loader:
        x = x.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits, mu, logvar, _ = model(x, sample_latent=True)
        loss, recon, kl = beta_vae_loss(x, logits, mu, logvar, beta=beta)

        if is_train:
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        total_loss += float(loss.item())
        total_recon += float(recon.item())
        total_kl += float(kl.item())
        n_batches += 1

    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def save_checkpoint(
    run_dir: str,
    model: BetaVAE,
    optimizer: torch.optim.Optimizer,
    cfg: VAEConfig,
    history: Dict[str, List[float]],
    tag: str = "last",
) -> str:
    ckpt = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "cfg": asdict(cfg),
        "history": history,
        "latent_dim": model.latent_dim,
    }
    path = os.path.join(run_dir, f"vae_{tag}.pt")
    torch.save(ckpt, path)
    return path


def load_checkpoint(path: str, device: torch.device) -> Tuple[BetaVAE, Dict[str, Any]]:
    ckpt = torch.load(path, map_location=device)
    latent_dim = int(ckpt.get("latent_dim", ckpt["cfg"]["latent_dim"]))
    model = BetaVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


# =========================
# Plotting (saved as PDF)
# =========================
def _save_fig_pdf(fig: plt.Figure, outpath: str) -> None:
    fig.savefig(outpath, format="pdf", bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_reconstructions_pdf(
    model: BetaVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    outpath: str,
    max_images: int = 16,
    title: str = "Reconstructions",
) -> None:
    model.eval()
    x, _ = next(iter(loader))
    x = x.to(device)[:max_images]
    logits, _, _, _ = model(x, sample_latent=True)
    x_hat = torch.sigmoid(logits)

    n = x.size(0)
    fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 2.8))
    for i in range(n):
        axes[0, i].imshow(x[i, 0].detach().cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i, 0].detach().cpu(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstruction", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    _save_fig_pdf(fig, outpath)


@torch.no_grad()
def save_prior_samples_pdf(
    model: BetaVAE,
    device: torch.device,
    outpath: str,
    n: int = 25,
    title: str = "Samples from prior N(0,I)",
) -> None:
    model.eval()
    z = torch.randn(n, model.latent_dim, device=device)
    x_hat = model.decode(z).detach().cpu()

    cols = int(math.sqrt(n))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.8 * rows))
    axes = axes.flatten()
    for i in range(rows * cols):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(x_hat[i, 0], cmap="gray")
    fig.suptitle(title)
    fig.tight_layout()
    _save_fig_pdf(fig, outpath)


def save_curves_pdf(history: Dict[str, List[float]], outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("β-VAE loss")
    fig.tight_layout()
    _save_fig_pdf(fig, outpath)


def save_recon_kl_curves_pdf(history: Dict[str, List[float]], outdir: str) -> None:
    # recon
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_recon"], label="train_recon")
    ax.plot(history["val_recon"], label="val_recon")
    ax.set_xlabel("epoch")
    ax.set_ylabel("recon")
    ax.legend()
    ax.set_title("Reconstruction term")
    fig.tight_layout()
    _save_fig_pdf(fig, os.path.join(outdir, "recon_curve.pdf"))

    # kl
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["train_kl"], label="train_kl")
    ax.plot(history["val_kl"], label="val_kl")
    ax.set_xlabel("epoch")
    ax.set_ylabel("kl")
    ax.legend()
    ax.set_title("KL term")
    fig.tight_layout()
    _save_fig_pdf(fig, os.path.join(outdir, "kl_curve.pdf"))


@torch.no_grad()
def save_latent_grid_pdf(
    model: BetaVAE,
    device: torch.device,
    outpath: str,
    grid_size: int = 20,
    lim: float = 3.0,
    title: str = "Latent grid (latent_dim=2)",
) -> None:
    if model.latent_dim != 2:
        return
    model.eval()

    zs = torch.linspace(-lim, lim, grid_size)
    canvas = torch.zeros(grid_size * 28, grid_size * 28)

    for i, yi in enumerate(zs.flip(0)):
        for j, xj in enumerate(zs):
            z = torch.tensor([[xj, yi]], device=device, dtype=torch.float32)
            img = model.decode(z)[0, 0].detach().cpu()
            canvas[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = img

    fig = plt.figure(figsize=(8, 8))
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    _save_fig_pdf(fig, outpath)


def _pca_project_2d(z: torch.Tensor) -> torch.Tensor:
    """
    Simple PCA-to-2D using torch SVD.
    z: [N, d] on CPU float32.
    returns: [N, 2]
    """
    zc = z - z.mean(dim=0, keepdim=True)
    # covariance-ish via SVD on centered data
    # zc = U S V^T, PCs are columns of V
    _, _, Vt = torch.linalg.svd(zc, full_matrices=False)
    W = Vt[:2].T  # [d,2]
    return zc @ W  # [N,2]


@torch.no_grad()
def save_latent_scatter_pdf(
    model: BetaVAE,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    outpath: str,
    max_points: int = 5000,
    title: str = "Latent scatter (mu)",
) -> None:
    model.eval()

    zs = []
    ys = []
    n_collected = 0

    for x, y in loader:
        x = x.to(device)
        mu = model.encode_mu(x).detach().cpu()
        zs.append(mu)
        ys.append(y.detach().cpu())
        n_collected += mu.size(0)
        if n_collected >= max_points:
            break

    z = torch.cat(zs, dim=0)[:max_points].float()
    y = torch.cat(ys, dim=0)[:max_points].long()

    if model.latent_dim == 2:
        z2 = z
    else:
        z2 = _pca_project_2d(z)

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(z2[:, 0].numpy(), z2[:, 1].numpy(), s=6, alpha=0.6, c=y.numpy(), cmap="tab10")
    ax.set_title(title + ("" if model.latent_dim == 2 else " (PCA projection)"))
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    fig.colorbar(sc, ax=ax, label="label")
    fig.tight_layout()
    _save_fig_pdf(fig, outpath)


# =========================
# Main training entry point
# =========================
def train_vae(
    cfg: VAEConfig,
    dataloader_train: torch.utils.data.DataLoader,
    dataloader_val: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[BetaVAE, Dict[str, List[float]], str]:
    """
    Trains a BetaVAE and saves artifacts to:
        run_dir/
          config.json
          model.txt
          history.json
          vae_last.pt (or vae_best.pt)
          plots/*.pdf
    Returns: (model, history, run_dir)
    """
    set_seed(cfg.seed)

    run_dir = make_run_dir(cfg.run_root, cfg.run_name)
    plots_dir = os.path.join(run_dir, "plots")

    # init model/opt
    model = BetaVAE(latent_dim=cfg.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # save config + model text
    _json_dump(os.path.join(run_dir, "config.json"), asdict(cfg))
    _text_dump(os.path.join(run_dir, "model.txt"), repr(model))

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "train_recon": [], "val_recon": [],
        "train_kl": [], "val_kl": [],
    }

    best_val = float("inf")
    best_path = None

    for epoch in range(1, cfg.epochs + 1):
        tr = _run_epoch(model, dataloader_train, device, beta=cfg.beta,
                        optimizer=optimizer, grad_clip_norm=cfg.grad_clip_norm)
        va = _run_epoch(model, dataloader_val, device, beta=cfg.beta,
                        optimizer=None, grad_clip_norm=None)

        train_loss, train_recon, train_kl = tr
        val_loss, val_recon, val_kl = va

        history["train_loss"].append(train_loss)
        history["train_recon"].append(train_recon)
        history["train_kl"].append(train_kl)
        history["val_loss"].append(val_loss)
        history["val_recon"].append(val_recon)
        history["val_kl"].append(val_kl)

        print(
            f"[VAE] epoch {epoch:03d}/{cfg.epochs} | "
            f"train loss={train_loss:.3f} recon={train_recon:.3f} kl={train_kl:.3f} | "
            f"val loss={val_loss:.3f} recon={val_recon:.3f} kl={val_kl:.3f}"
        )

        # best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            best_path = save_checkpoint(run_dir, model, optimizer, cfg, history, tag="best")

    # save final checkpoint
    last_path = save_checkpoint(run_dir, model, optimizer, cfg, history, tag="last")
    if cfg.save_best_only and best_path is not None:
        # keep best as the main model file too
        main_model_path = os.path.join(run_dir, "vae_model.pt")
        os.replace(best_path, main_model_path)
    else:
        main_model_path = os.path.join(run_dir, "vae_model.pt")
        os.replace(last_path, main_model_path)

    # save history
    _json_dump(os.path.join(run_dir, "history.json"), history)

    # plots (PDF)
    save_curves_pdf(history, os.path.join(plots_dir, "loss_curve.pdf"))
    save_recon_kl_curves_pdf(history, plots_dir)

    save_reconstructions_pdf(
        model, dataloader_val, device,
        outpath=os.path.join(plots_dir, "reconstructions.pdf"),
        max_images=cfg.max_images_plot,
        title=f"Reconstructions (beta={cfg.beta}, zdim={cfg.latent_dim})",
    )

    save_prior_samples_pdf(
        model, device,
        outpath=os.path.join(plots_dir, "prior_samples.pdf"),
        n=cfg.prior_n_plot,
        title=f"Prior samples (beta={cfg.beta}, zdim={cfg.latent_dim})",
    )

    save_latent_scatter_pdf(
        model, dataloader_val, device,
        outpath=os.path.join(plots_dir, "latent_scatter.pdf"),
        max_points=cfg.latent_scatter_max_points,
        title=f"Latent scatter (beta={cfg.beta}, zdim={cfg.latent_dim})",
    )

    save_latent_grid_pdf(
        model, device,
        outpath=os.path.join(plots_dir, "latent_grid.pdf"),
        grid_size=cfg.latent_grid_size,
        lim=cfg.latent_grid_lim,
        title=f"Latent grid (beta={cfg.beta}, zdim={cfg.latent_dim})",
    )

    return model, history, run_dir
