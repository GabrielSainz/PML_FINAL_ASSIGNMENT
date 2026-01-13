# utils/evaluate_diffusion_trained2.py
"""
Evaluate:
1) Latent priors trained in run_root (VAE prior / latent DDPM / continuous VP-SDE)
2) Pixel-space baseline DDPM + Flow Matching (checkpoints in pixel_ckpt_dir)

Uses a FIXED, PRETRAINED MNISTFeatureNet for all metrics.

Outputs in out_dir:
- summary.csv
- sample_validity.pdf
- fid_scores.pdf
- sampling_time.pdf
- label_histograms.pdf (multi-page)
- quickcheck_gen__*.pdf (1 row, 12 cols) for each evaluated model
- quickcheck_recon__*.pdf (2 rows original/recon, 12 cols) for each latent run's VAE (optional)

Important:
- Pixel models in your setup were trained on FLAT 784 tensors in [-1,1] (dequantized),
  so we MUST map generated samples [-1,1] -> [0,1] before plotting / feature net eval.
- Real MNIST features are computed from ToTensor() only (images in [0,1], shape [B,1,28,28]).
"""

from __future__ import annotations

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---- Your project imports (latent side)
from utils.latent_utils import load_latent_checkpoint

# We try to use your exact training-time diffusion code (recommended).
# If these imports fail, you should point them to your actual module path.
try:
    from utils.diffusion import LatentDenoiser, LatentDDPM, VPSDE, load_vae_from_run
except Exception as e:
    raise ImportError(
        "Could not import LatentDenoiser/LatentDDPM/VPSDE/load_vae_from_run from utils.diffusion.\n"
        "Fix the import path in utils/evaluate_diffusion_trained2.py to match your project.\n"
        f"Original error: {repr(e)}"
    )

# ============================================================
# Feature net (fixed evaluator)
# ============================================================
class MNISTFeatureNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, feat_dim)
        self.fc2 = nn.Linear(feat_dim, 10)

    def forward(self, x, return_features=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        feats = F.relu(self.fc1(x))
        logits = self.fc2(feats)
        if return_features:
            return logits, feats
        return logits


def load_feature_net(ckpt_path: str, device: torch.device) -> MNISTFeatureNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = MNISTFeatureNet(feat_dim=128)
    net.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
    net.to(device).eval()
    print(f"[OK] Loaded MNISTFeatureNet: {ckpt_path}")
    return net


# ============================================================
# Pixel-space model defs (baseline DDPM + Flow Matching)
# (must match the architectures used to TRAIN those checkpoints)
# ============================================================
class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        if x.dim() != 1:
            x = x.view(-1)
        x_proj = x[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=(32, 64, 128, 256), embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(
            channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1
        )
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(
            channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1
        )
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

    def forward(self, x, t):
        if t.dim() != 1:
            t = t.view(-1)
        embed = self.act(self.embed(t))

        h1 = self.conv1(x)
        h1 = h1 + self.dense1(embed)
        h1 = self.act(self.gnorm1(h1))

        h2 = self.conv2(h1)
        h2 = h2 + self.dense2(embed)
        h2 = self.act(self.gnorm2(h2))

        h3 = self.conv3(h2)
        h3 = h3 + self.dense3(embed)
        h3 = self.act(self.gnorm3(h3))

        h4 = self.conv4(h3)
        h4 = h4 + self.dense4(embed)
        h4 = self.act(self.gnorm4(h4))

        h = self.tconv4(h4)
        h = h + self.dense5(embed)
        h = self.act(self.tgnorm4(h))

        h = self.tconv3(torch.cat([h, h3], dim=1))
        h = h + self.dense6(embed)
        h = self.act(self.tgnorm3(h))

        h = self.tconv2(torch.cat([h, h2], dim=1))
        h = h + self.dense7(embed)
        h = self.act(self.tgnorm2(h))

        h = self.tconv1(torch.cat([h, h1], dim=1))
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h


class DDPM(nn.Module):
    def __init__(self, network, T=1000, beta_1=1e-4, beta_T=2e-2):
        super().__init__()
        self._network = network
        self.T = T

        self.network = lambda x, t: self._network(
            x.reshape(-1, 1, 28, 28),
            (t.squeeze() / T)
        ).reshape(-1, 28 * 28)

        self.register_buffer("beta", torch.linspace(beta_1, beta_T, T + 1))
        self.register_buffer("alpha", 1.0 - self.beta)
        self.register_buffer("alpha_bar", self.alpha.cumprod(dim=0))

    def forward_diffusion(self, x0, t, epsilon):
        mean = torch.sqrt(self.alpha_bar[t]) * x0
        std = torch.sqrt(1.0 - self.alpha_bar[t])
        return mean + std * epsilon

    def reverse_diffusion(self, xt, t, epsilon):
        mean = (1.0 / torch.sqrt(self.alpha[t])) * (
            xt - (self.beta[t] / torch.sqrt(1.0 - self.alpha_bar[t])) * self.network(xt, t)
        )
        std = torch.where(
            t > 0,
            torch.sqrt(((1.0 - self.alpha_bar[t - 1]) / (1.0 - self.alpha_bar[t])) * self.beta[t]),
            torch.zeros_like(t, dtype=xt.dtype, device=xt.device),
        )
        return mean + std * epsilon

    @torch.no_grad()
    def sample(self, shape):
        x = torch.randn(shape, device=self.beta.device)
        for step in range(self.T, 0, -1):
            t = torch.full((x.size(0), 1), step, device=x.device, dtype=torch.long)
            eps = torch.randn_like(x) if step > 1 else torch.zeros_like(x)
            x = self.reverse_diffusion(x, t, eps)
        return x


class RectifiedFlow(nn.Module):
    def __init__(self, unet_backbone):
        super().__init__()
        self._net = unet_backbone
        self.net = lambda x, t: self._net(
            x.view(-1, 1, 28, 28),
            t.view(-1)
        ).view(-1, 28 * 28)

    @torch.no_grad()
    def sample(self, shape, steps=100, solver="heun"):
        B = shape[0]
        x = torch.randn(shape, device=next(self.parameters()).device)
        dt = 1.0 / steps

        if solver == "euler":
            for i in range(steps, 0, -1):
                t = torch.full((B,), i / steps, device=x.device)
                v = self.net(x, t)
                x = x - dt * v
            return x

        if solver == "heun":
            for i in range(steps, 0, -1):
                t1 = torch.full((B,), i / steps, device=x.device)
                v1 = self.net(x, t1)
                x_euler = x - dt * v1

                t0 = torch.full((B,), (i - 1) / steps, device=x.device)
                v0 = self.net(x_euler, t0)
                x = x - 0.5 * dt * (v1 + v0)
            return x

        raise ValueError("solver must be 'euler' or 'heun'")


# ============================================================
# Helpers
# ============================================================
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _read_json(p: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def list_run_dirs(run_root: str) -> List[str]:
    if not os.path.exists(run_root):
        return []
    out = []
    for name in sorted(os.listdir(run_root)):
        d = os.path.join(run_root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")):
            out.append(d)
    return out


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_it(device: torch.device, fn):
    _cuda_sync(device)
    t0 = time.perf_counter()
    out = fn()
    _cuda_sync(device)
    return out, (time.perf_counter() - t0)


# ============================================================
# Postprocessing for pixel outputs (CRITICAL FIX)
# ============================================================
@torch.no_grad()
def flat_to_img01(x_flat: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """
    x_flat: [B,784] (often in [-1,1] for your pixel models)
    return : [B,1,28,28] on CPU in [0,1]
    mode:
      - "minus1_1": assume x in [-1,1]
      - "0_1": assume x in [0,1]
      - "auto": detect by min/max
    """
    x = x_flat.view(-1, 1, 28, 28).float()

    if mode == "minus1_1":
        x = (x + 1.0) * 0.5
    elif mode == "0_1":
        x = x
    else:
        xmin = float(x.min().item())
        xmax = float(x.max().item())
        if xmin < -0.05:
            x = (x + 1.0) * 0.5

    return x.clamp(0.0, 1.0).cpu()


# ============================================================
# Metrics: classifier + FID(feature)
# ============================================================
@torch.no_grad()
def classifier_metrics(
    clf: MNISTFeatureNet,
    images01: torch.Tensor,  # [N,1,28,28] CPU [0,1]
    device: torch.device,
    validity_thresh: float = 0.9,
    batch_size: int = 512,
) -> Dict[str, Any]:
    clf.eval()
    N = images01.size(0)

    preds, confs, feats = [], [], []
    for i in range(0, N, batch_size):
        x = images01[i:i + batch_size].to(device)
        logits, feat = clf(x, return_features=True)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        preds.append(pred.detach().cpu())
        confs.append(conf.detach().cpu())
        feats.append(feat.detach().cpu())

    preds = torch.cat(preds, dim=0)
    confs = torch.cat(confs, dim=0)
    feats = torch.cat(feats, dim=0)

    counts = torch.bincount(preds, minlength=10).float()
    unique_digits = int((counts > 0).sum().item())

    p = counts / counts.sum().clamp(min=1.0)
    entropy = float(-(p * (p + 1e-12).log()).sum().item())

    return {
        "validity": float((confs >= validity_thresh).float().mean().item()),
        "mean_conf": float(confs.mean().item()),
        "unique_digits": unique_digits,
        "label_entropy_nats": entropy,
        "counts": counts.numpy(),
        "features": feats,  # CPU
    }


def _cov(x: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    return (x.T @ x) / max(1, (n - 1))


def fid_from_features(feat_real: torch.Tensor, feat_gen: torch.Tensor, eps: float = 1e-6) -> float:
    feat_real = feat_real.float()
    feat_gen = feat_gen.float()

    m1 = feat_real.mean(dim=0)
    m2 = feat_gen.mean(dim=0)
    x1 = feat_real - m1
    x2 = feat_gen - m2

    C1 = _cov(x1) + eps * torch.eye(x1.size(1))
    C2 = _cov(x2) + eps * torch.eye(x2.size(1))

    s1, U1 = torch.linalg.eigh(C1)
    s1 = torch.clamp(s1, min=0.0)
    sqrtC1 = (U1 * torch.sqrt(s1).unsqueeze(0)) @ U1.T

    M = sqrtC1 @ C2 @ sqrtC1
    sM, _ = torch.linalg.eigh(M)
    sM = torch.clamp(sM, min=0.0)
    trace_sqrt = torch.sum(torch.sqrt(sM))

    fid = torch.sum((m1 - m2) ** 2) + torch.trace(C1) + torch.trace(C2) - 2.0 * trace_sqrt
    return float(fid.item())


# ============================================================
# Plot + quickcheck writers
# ============================================================
def save_gen_quickcheck_pdf(images01: torch.Tensor, pdf_path: str, title: str, cols: int = 12) -> None:
    n = min(cols, images01.size(0))
    fig, axes = plt.subplots(1, n, figsize=(1.3 * n, 1.6))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(images01[i, 0].cpu(), cmap="gray")
        axes[i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_recon_quickcheck_pdf(x01: torch.Tensor, xhat01: torch.Tensor, pdf_path: str, title: str, cols: int = 12) -> None:
    n = min(cols, x01.size(0), xhat01.size(0))
    fig, axes = plt.subplots(2, n, figsize=(1.3 * n, 3.2))
    for i in range(n):
        axes[0, i].imshow(x01[i, 0].cpu(), cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(xhat01[i, 0].cpu(), cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstruction", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def save_summary_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"[OK] Saved: {path}")


def save_label_histograms_pdf(label_hists: List[Tuple[np.ndarray, str]], path: str) -> None:
    with PdfPages(path) as pdf:
        for counts, title in label_hists:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(list(range(10)), counts)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Count")
            ax.set_title(title)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"[OK] Saved: {path}")


def save_fid_bar_pdf(fid_points: List[Tuple[float, str]], path: str) -> None:
    fid_sorted = sorted(fid_points, key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([n for _, n in fid_sorted], [v for v, _ in fid_sorted])
    ax.set_ylabel("MNIST-FID (feature space)")
    ax.set_title("Lower is better")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def save_scatter_pdf(val_points: List[Tuple[float, float, str]], path: str, validity_thresh: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, c, label in val_points:
        ax.scatter(v, c, s=35)
        ax.annotate(label, (v, c), fontsize=7, alpha=0.9)
    ax.set_xlabel(f"Validity (P(max) ≥ {validity_thresh})")
    ax.set_ylabel("Mean confidence")
    ax.set_title("Sample quality (classifier-based)")
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


def save_sampling_time_pdf(time_points: List[Tuple[float, str]], path: str) -> None:
    t_sorted = sorted(time_points, key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([n for _, n in t_sorted], [v for v, _ in t_sorted])
    ax.set_ylabel("Sampling time (s) for n_gen samples")
    ax.set_title("Lower is faster")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved: {path}")


# ============================================================
# Real features (CRITICAL: use [0,1] MNIST, not your dequant/flatten)
# ============================================================
@torch.no_grad()
def compute_real_features(
    clf: MNISTFeatureNet,
    data_dir: str,
    device: torch.device,
    n_val: int = 10_000,
    batch_size: int = 256,
    num_workers: int = 2,
) -> torch.Tensor:
    eval_tf = transforms.ToTensor()  # [0,1], [1,28,28]
    full = datasets.MNIST(data_dir, train=True, download=True, transform=eval_tf)
    n_train = len(full) - n_val
    _, ds_val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(123))
    dl = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    feats = []
    clf.eval()
    for x, _ in dl:
        x = x.to(device)
        _, f = clf(x, return_features=True)
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)


# ============================================================
# Main: evaluate everything
# ============================================================
def evaluate_all_models(
    run_root: str,
    out_dir: str,
    data_dir: str,
    device: torch.device,
    feature_net_ckpt: str,
    pixel_ckpt_dir: Optional[str] = None,
    baseline_ckpt_name: str = "baseline_ddpm_ema.pt",
    flow_ckpt_name: str = "flow_matching.pt",
    pixel_output_mode: str = "auto",  # auto | 0_1 | minus1_1
    n_gen: int = 5000,
    validity_thresh: float = 0.9,
    batch_size: int = 256,
    num_workers: int = 2,
    quickcheck: bool = True,
    quickcheck_cols: int = 12,
) -> List[Dict[str, Any]]:
    """
    Returns:
      rows: list of dicts (also saved to out_dir/summary.csv)

    Notes on "Sampling time (s)":
      This is the TOTAL wall time to generate n_gen samples (not per-sample).
    """
    _ensure_dir(out_dir)

    clf = load_feature_net(feature_net_ckpt, device)
    feat_real = compute_real_features(clf, data_dir, device, n_val=10_000, batch_size=256, num_workers=num_workers)
    print("[OK] Real features:", tuple(feat_real.shape))

    rows: List[Dict[str, Any]] = []
    label_hists: List[Tuple[np.ndarray, str]] = []
    fid_points: List[Tuple[float, str]] = []
    val_points: List[Tuple[float, float, str]] = []
    time_points: List[Tuple[float, str]] = []

    # ------------------------------------------------------------
    # A) Evaluate latent diffusion runs in run_root
    # ------------------------------------------------------------
    run_dirs = list_run_dirs(run_root)
    if not run_dirs:
        print(f"[WARN] No diffusion runs found in: {run_root}")

    # fixed images for recon quickcheck (VAE side)
    recon_x01 = None
    if quickcheck:
        eval_tf = transforms.ToTensor()
        ds = datasets.MNIST(data_dir, train=True, download=True, transform=eval_tf)
        dl_tmp = DataLoader(ds, batch_size=64, shuffle=True, num_workers=num_workers)
        x01, _ = next(iter(dl_tmp))
        recon_x01 = x01[:quickcheck_cols].to(device)

    for ddir in run_dirs:
        run_name = os.path.basename(ddir)
        cfg = _read_json(os.path.join(ddir, "config.json")) or {}
        vae_run_dir = cfg.get("vae_run_dir", None)
        if vae_run_dir is None:
            print(f"[Skip] {run_name}: missing vae_run_dir in config.json")
            continue

        latent_path = os.path.join(ddir, "latents", "mnist_latents_mu.pt")
        if not os.path.exists(latent_path):
            print(f"[Skip] {run_name}: missing {latent_path}")
            continue

        ddpm_path = os.path.join(ddir, "models", "ddpm.pt")
        cont_path = os.path.join(ddir, "models", "cont.pt")

        # Load VAE and latent stats
        vae = load_vae_from_run(vae_run_dir, device=device)  # your exact loader
        latent_ckpt = load_latent_checkpoint(latent_path)
        mean = latent_ckpt["mean"]
        std = latent_ckpt["std"]
        latent_dim = int(latent_ckpt["meta"]["latent_dim"])

        # ----- recon quickcheck (once per run: VAE recon)
        if quickcheck and recon_x01 is not None:
            with torch.no_grad():
                # assume VAE expects [0,1]
                logits, mu, logvar, z = vae(recon_x01, sample_latent=False)
                xhat01 = torch.sigmoid(logits).clamp(0, 1).detach().cpu()
            pdf_path = os.path.join(out_dir, f"quickcheck_recon__{run_name}.pdf")
            save_recon_quickcheck_pdf(
                recon_x01.detach().cpu(), xhat01, pdf_path,
                title=f"VAE reconstruction — {run_name}", cols=quickcheck_cols
            )

        # ============ 1) VAE prior ============
        def sample_vae_prior() -> torch.Tensor:
            outs = []
            remaining = n_gen
            while remaining > 0:
                b = min(batch_size, remaining)
                z = torch.randn(b, latent_dim, device=device)
                imgs = vae.decode(z).detach().cpu()  # [0,1]
                outs.append(imgs)
                remaining -= b
            return torch.cat(outs, dim=0)

        imgs_vae, t_vae = time_it(device, sample_vae_prior)
        m = classifier_metrics(clf, imgs_vae, device=device, validity_thresh=validity_thresh)
        fid = fid_from_features(feat_real, m["features"])

        label = f"{run_name}::vae_prior"
        rows.append({
            "run_name": run_name,
            "model_type": "vae_prior",
            "latent_dim": latent_dim,
            "MNIST-FID": fid,
            "Mean conf.": m["mean_conf"],
            "Unique digits": m["unique_digits"],
            "Sampling time (s)": t_vae,
            "validity@thr": m["validity"],
            "label_entropy_nats": m["label_entropy_nats"],
        })
        label_hists.append((m["counts"], label))
        fid_points.append((fid, label))
        val_points.append((m["validity"], m["mean_conf"], label))
        time_points.append((t_vae, label))

        if quickcheck:
            save_gen_quickcheck_pdf(
                imgs_vae, os.path.join(out_dir, f"quickcheck_gen__{run_name}__vae_prior.pdf"),
                title=f"Generated — {label}", cols=quickcheck_cols
            )

        print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={t_vae:.2f}s")

        # ============ 2) Latent DDPM ============
        if os.path.exists(ddpm_path):
            ddpm_ckpt = torch.load(ddpm_path, map_location=device)
            den = LatentDenoiser(
                latent_dim=latent_dim,
                time_dim=int(cfg.get("time_dim", ddpm_ckpt.get("cfg", {}).get("time_dim", 128))),
                hidden=int(cfg.get("hidden", ddpm_ckpt.get("cfg", {}).get("hidden", 256))),
                depth=int(cfg.get("depth", ddpm_ckpt.get("cfg", {}).get("depth", 3))),
            ).to(device)
            den.load_state_dict(ddpm_ckpt["state_dict"])
            den.eval()

            T = int(cfg.get("ddpm_T", ddpm_ckpt.get("cfg", {}).get("ddpm_T", 200)))
            ddpm = LatentDDPM(den, latent_dim=latent_dim, T=T, device=device)

            def sample_latent_ddpm() -> torch.Tensor:
                # sample in "scaled space" (like your training), then unscale and decode
                z_scaled = ddpm.sample(n_gen)  # [N,d] on device
                z = z_scaled * std.to(device) + mean.to(device)
                imgs = vae.decode(z).detach().cpu()
                return imgs

            imgs, t_ddpm = time_it(device, sample_latent_ddpm)
            m = classifier_metrics(clf, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            label = f"{run_name}::ddpm"
            rows.append({
                "run_name": run_name,
                "model_type": "ddpm",
                "latent_dim": latent_dim,
                "MNIST-FID": fid,
                "Mean conf.": m["mean_conf"],
                "Unique digits": m["unique_digits"],
                "Sampling time (s)": t_ddpm,
                "validity@thr": m["validity"],
                "label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))
            time_points.append((t_ddpm, label))

            if quickcheck:
                save_gen_quickcheck_pdf(
                    imgs, os.path.join(out_dir, f"quickcheck_gen__{run_name}__ddpm.pdf"),
                    title=f"Generated — {label}", cols=quickcheck_cols
                )

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={t_ddpm:.2f}s")
        else:
            print(f"[Warn] {run_name}: missing models/ddpm.pt")

        # ============ 3) Continuous VP-SDE ============
        if os.path.exists(cont_path):
            cont_ckpt = torch.load(cont_path, map_location=device)
            den = LatentDenoiser(
                latent_dim=latent_dim,
                time_dim=int(cfg.get("time_dim", cont_ckpt.get("cfg", {}).get("time_dim", 128))),
                hidden=int(cfg.get("hidden", cont_ckpt.get("cfg", {}).get("hidden", 256))),
                depth=int(cfg.get("depth", cont_ckpt.get("cfg", {}).get("depth", 3))),
            ).to(device)
            den.load_state_dict(cont_ckpt["state_dict"])
            den.eval()

            beta_min = float(cfg.get("beta_min", cont_ckpt.get("cfg", {}).get("beta_min", 0.1)))
            beta_max = float(cfg.get("beta_max", cont_ckpt.get("cfg", {}).get("beta_max", 20.0)))
            cont = VPSDE(den, latent_dim=latent_dim, beta_min=beta_min, beta_max=beta_max, device=device)

            steps = int(cfg.get("sample_steps_cont", cont_ckpt.get("cfg", {}).get("sample_steps_cont", 200)))

            def sample_latent_cont() -> torch.Tensor:
                z_scaled = cont.sample(n_gen, steps=steps)  # [N,d]
                z = z_scaled * std.to(device) + mean.to(device)
                imgs = vae.decode(z).detach().cpu()
                return imgs

            imgs, t_cont = time_it(device, sample_latent_cont)
            m = classifier_metrics(clf, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            label = f"{run_name}::cont"
            rows.append({
                "run_name": run_name,
                "model_type": "cont",
                "latent_dim": latent_dim,
                "MNIST-FID": fid,
                "Mean conf.": m["mean_conf"],
                "Unique digits": m["unique_digits"],
                "Sampling time (s)": t_cont,
                "validity@thr": m["validity"],
                "label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))
            time_points.append((t_cont, label))

            if quickcheck:
                save_gen_quickcheck_pdf(
                    imgs, os.path.join(out_dir, f"quickcheck_gen__{run_name}__cont.pdf"),
                    title=f"Generated — {label}", cols=quickcheck_cols
                )

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={t_cont:.2f}s")
        else:
            print(f"[Warn] {run_name}: missing models/cont.pt")

    # ------------------------------------------------------------
    # B) Evaluate pixel-space baseline DDPM + Flow Matching
    # ------------------------------------------------------------
    if pixel_ckpt_dir is not None:
        base_path = os.path.join(pixel_ckpt_dir, baseline_ckpt_name)
        flow_path = os.path.join(pixel_ckpt_dir, flow_ckpt_name)

        # Baseline DDPM
        if os.path.exists(base_path):
            payload = torch.load(base_path, map_location="cpu")
            # Use cfg if saved, else fall back
            cfg = payload.get("cfg", {})
            T = int(cfg.get("T", 1000))
            b1 = float(cfg.get("beta_1", 1e-4))
            bT = float(cfg.get("beta_T", 2e-2))

            unet = ScoreNet(marginal_prob_std=lambda t: torch.ones_like(t).to(device))
            baseline = DDPM(unet, T=T, beta_1=b1, beta_T=bT).to(device)
            baseline.load_state_dict(payload["ema_state_dict"] if "ema_state_dict" in payload else payload["state_dict"])
            baseline.eval()

            def sample_pixel_baseline() -> torch.Tensor:
                outs = []
                remaining = n_gen
                while remaining > 0:
                    b = min(batch_size, remaining)
                    x_flat = baseline.sample((b, 28 * 28))
                    outs.append(flat_to_img01(x_flat, mode=pixel_output_mode))
                    remaining -= b
                return torch.cat(outs, dim=0)

            imgs, t_base = time_it(device, sample_pixel_baseline)
            m = classifier_metrics(clf, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            run_name = f"pixel::{baseline_ckpt_name}"
            label = run_name
            rows.append({
                "run_name": run_name,
                "model_type": "pixel_ddpm",
                "latent_dim": "",
                "MNIST-FID": fid,
                "Mean conf.": m["mean_conf"],
                "Unique digits": m["unique_digits"],
                "Sampling time (s)": t_base,
                "validity@thr": m["validity"],
                "label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))
            time_points.append((t_base, label))

            if quickcheck:
                save_gen_quickcheck_pdf(
                    imgs, os.path.join(out_dir, f"quickcheck_gen__pixel__baseline_ddpm.pdf"),
                    title=f"Generated — {label}", cols=quickcheck_cols
                )

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={t_base:.2f}s")
        else:
            print(f"[Warn] Missing pixel baseline ckpt: {base_path}")

        # Flow Matching
        if os.path.exists(flow_path):
            payload = torch.load(flow_path, map_location="cpu")
            unet = ScoreNet(marginal_prob_std=lambda t: torch.ones_like(t).to(device))
            flow = RectifiedFlow(unet).to(device)
            flow.load_state_dict(payload["state_dict"] if "state_dict" in payload else payload)
            flow.eval()

            # you can tune these if you want
            steps = 100
            solver = "heun"

            def sample_pixel_flow() -> torch.Tensor:
                outs = []
                remaining = n_gen
                while remaining > 0:
                    b = min(batch_size, remaining)
                    x_flat = flow.sample((b, 28 * 28), steps=steps, solver=solver)
                    outs.append(flat_to_img01(x_flat, mode=pixel_output_mode))
                    remaining -= b
                return torch.cat(outs, dim=0)

            imgs, t_flow = time_it(device, sample_pixel_flow)
            m = classifier_metrics(clf, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            run_name = f"pixel::{flow_ckpt_name}"
            label = run_name
            rows.append({
                "run_name": run_name,
                "model_type": f"pixel_flow_{steps}_{solver}",
                "latent_dim": "",
                "MNIST-FID": fid,
                "Mean conf.": m["mean_conf"],
                "Unique digits": m["unique_digits"],
                "Sampling time (s)": t_flow,
                "validity@thr": m["validity"],
                "label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))
            time_points.append((t_flow, label))

            if quickcheck:
                save_gen_quickcheck_pdf(
                    imgs, os.path.join(out_dir, f"quickcheck_gen__pixel__flow_matching.pdf"),
                    title=f"Generated — {label}", cols=quickcheck_cols
                )

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={t_flow:.2f}s")
        else:
            print(f"[Warn] Missing pixel flow ckpt: {flow_path}")

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------
    if rows:
        save_summary_csv(rows, os.path.join(out_dir, "summary.csv"))
        save_scatter_pdf(val_points, os.path.join(out_dir, "sample_validity.pdf"), validity_thresh)
        save_fid_bar_pdf(fid_points, os.path.join(out_dir, "fid_scores.pdf"))
        save_sampling_time_pdf(time_points, os.path.join(out_dir, "sampling_time.pdf"))
        save_label_histograms_pdf(label_hists, os.path.join(out_dir, "label_histograms.pdf"))
    else:
        print("[WARN] No rows produced; nothing to save.")

    return rows
