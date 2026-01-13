# evaluate_diffusion.py
"""
Evaluate:
1) Latent diffusion priors (DDPM + Continuous VP-SDE) in runs_diffusion/
2) Pixel-space models (baseline DDPM + Flow Matching) from a checkpoint folder

Metrics per model:
- mnist_fid ↓          (FID computed in MNISTFeatureNet feature space)
- mean_conf ↑          (avg max softmax prob)
- unique_digits ↑      (#unique predicted digits among VALID samples only)
- sampling_time_s      (TOTAL time for n_gen samples, includes sampling + any decoding)

Quickcheck PDFs (saved into out_dir/quickcheck/):
- Latent runs: recon 2 rows × 12 cols (original + recon) -> one PDF per latent run
- All models (latent priors + pixel models): generated samples 1 row × 12 cols -> one PDF per model

Outputs in out_dir:
- summary.csv
- sample_validity.pdf
- fid_scores.pdf
- label_histograms.pdf
- quickcheck/*.pdf

Latent diffusion run structure assumed:
  config.json (contains "vae_run_dir")
  models/ddpm.pt
  models/cont.pt
  latents/mnist_latents_mu.pt

Pixel checkpoints assumed:
  pixel_ckpt_dir / baseline_ckpt_name
  pixel_ckpt_dir / flow_ckpt_name
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

# Project imports
from utils.VAE import BetaVAE
from utils.latent_utils import load_latent_checkpoint
from utils.diffusion import LatentDenoiser, LatentDDPM, VPSDE, load_vae_from_run


# =============================================================================
# IO helpers
# =============================================================================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_diffusion_runs(run_root: str) -> List[str]:
    if not run_root or (not os.path.exists(run_root)):
        return []
    out = []
    for name in sorted(os.listdir(run_root)):
        p = os.path.join(run_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            out.append(p)
    return out


def _basename_no_ext(path: str) -> str:
    b = os.path.basename(path)
    return os.path.splitext(b)[0]


def _cuda_sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# =============================================================================
# Fixed evaluator: MNISTFeatureNet (your pretrained)
# =============================================================================
class MNISTFeatureNet(nn.Module):
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, feat_dim)
        self.fc2 = nn.Linear(feat_dim, 10)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 28->14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 14->7
        x = x.view(x.size(0), -1)
        feats = F.relu(self.fc1(x))
        logits = self.fc2(feats)
        if return_features:
            return logits, feats
        return logits


def load_feature_net(ckpt_path: str, device: torch.device, feat_dim: int = 128) -> MNISTFeatureNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = MNISTFeatureNet(feat_dim=feat_dim)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    net.load_state_dict(state)
    net.to(device).eval()
    print(f"Loaded feature net from: {ckpt_path}")
    return net


@torch.no_grad()
def compute_or_load_real_features(
    feat_net: MNISTFeatureNet,
    data_dir: str,
    device: torch.device,
    out_dir: str,
    n_val: int = 10_000,
    seed: int = 123,
    batch_size: int = 256,
    num_workers: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      feat_real: [n_val, feat_dim] CPU
      fixed_x12: [12,1,28,28] CPU (deterministic from the fixed val split)
    """
    _ensure_dir(out_dir)
    cache_path = os.path.join(out_dir, "mnist_real_features.pt")

    transform = transforms.ToTensor()
    full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    n_train = len(full) - n_val
    _, ds_val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    dl_val12 = DataLoader(ds_val, batch_size=12, shuffle=False, num_workers=num_workers, pin_memory=True)
    fixed_x12, _ = next(iter(dl_val12))
    fixed_x12 = fixed_x12.cpu()

    if os.path.exists(cache_path):
        ck = torch.load(cache_path, map_location="cpu")
        return ck["feat_real"], fixed_x12

    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = []
    feat_net.eval()
    for x, _ in dl_val:
        x = x.to(device)
        _, f = feat_net(x, return_features=True)
        feats.append(f.detach().cpu())
    feat_real = torch.cat(feats, dim=0)

    torch.save({"feat_real": feat_real, "n_val": n_val, "seed": seed}, cache_path)
    print(f"Saved real MNIST features cache to: {cache_path}")
    return feat_real, fixed_x12


# =============================================================================
# Metrics
# =============================================================================
@torch.no_grad()
def classifier_metrics(
    feat_net: MNISTFeatureNet,
    images: torch.Tensor,  # [N,1,28,28] CPU in [0,1]
    device: torch.device,
    validity_thresh: float = 0.9,
    batch_size: int = 512,
) -> Dict[str, Any]:
    feat_net.eval()
    N = images.size(0)

    preds, confs, feats = [], [], []
    for i in range(0, N, batch_size):
        x = images[i:i + batch_size].to(device)
        logits, feat = feat_net(x, return_features=True)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        preds.append(pred.detach().cpu())
        confs.append(conf.detach().cpu())
        feats.append(feat.detach().cpu())

    preds = torch.cat(preds, dim=0)
    confs = torch.cat(confs, dim=0)
    feats = torch.cat(feats, dim=0)

    counts = torch.bincount(preds, minlength=10).float()
    p = counts / counts.sum().clamp(min=1.0)
    entropy = float(-(p * (p + 1e-12).log()).sum().item())  # nats

    valid_mask = confs >= validity_thresh
    validity = float(valid_mask.float().mean().item())
    unique_digits = int(torch.unique(preds[valid_mask]).numel()) if valid_mask.any() else 0

    return {
        "validity": validity,
        "mean_conf": float(confs.mean().item()),
        "counts": counts.numpy(),
        "label_entropy_nats": entropy,
        "unique_digits": unique_digits,
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


# =============================================================================
# Latent sampling helpers
# =============================================================================
@torch.no_grad()
def generate_vae_prior_samples(vae: BetaVAE, device: torch.device, n: int, batch_size: int = 512) -> torch.Tensor:
    vae.eval()
    out = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        z = torch.randn(b, vae.latent_dim, device=device)
        imgs = vae.decode(z).detach().cpu()
        out.append(imgs)
        remaining -= b
    return torch.cat(out, dim=0)


@torch.no_grad()
def decode_scaled_latents(
    vae: BetaVAE,
    z_scaled: torch.Tensor,  # [N,d] on device
    mean: torch.Tensor,      # [d] CPU
    std: torch.Tensor,       # [d] CPU
    device: torch.device,
) -> torch.Tensor:
    vae.eval()
    mean = mean.to(device)
    std = std.to(device)
    z = z_scaled * std + mean
    return vae.decode(z).detach().cpu()


# =============================================================================
# Quickcheck figures (separate PDFs)
# =============================================================================
@torch.no_grad()
def fig_recon_2xN(vae: BetaVAE, x_cpu: torch.Tensor, device: torch.device, title: str) -> plt.Figure:
    """
    2 rows × N cols: top originals, bottom reconstructions (deterministic via mu).
    """
    vae.eval()
    x = x_cpu.to(device)
    logits, mu, logvar, z = vae(x, sample_latent=False)
    x_hat = torch.sigmoid(logits).detach().cpu()

    n = x_cpu.size(0)
    fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 2.8))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for i in range(n):
        axes[0, i].imshow(x_cpu[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i, 0], cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstruction", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def fig_images_1xN(imgs_cpu: torch.Tensor, title: str) -> plt.Figure:
    """
    1 row × N cols grid.
    """
    n = imgs_cpu.size(0)
    fig, axes = plt.subplots(1, n, figsize=(1.2 * n, 1.4))
    if n == 1:
        axes = [axes]
    for i in range(n):
        axes[i].imshow(imgs_cpu[i, 0], cmap="gray")
        axes[i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


# =============================================================================
# Pixel-space models (baseline DDPM + Flow Matching)
# =============================================================================
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        if x.dim() != 1:
            x = x.view(-1)
        x_proj = x[:, None] * self.W[None, :] * 2.0 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    """Time-dependent score model built upon U-Net architecture."""
    def __init__(self, marginal_prob_std, channels=(32, 64, 128, 256), embed_dim=256):
        super().__init__()
        self.marginal_prob_std = marginal_prob_std
        self.act = lambda x: x * torch.sigmoid(x)

        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Encoder
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

        # Decoder
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
    """Baseline DDPM wrapper (flat 784)."""
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
    """Rectified Flow / Flow Matching (flat 784)."""
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


def _postprocess_pixel_samples(x_flat: torch.Tensor, mode: str = "auto") -> torch.Tensor:
    """
    x_flat: [N,784] on device
    returns: [N,1,28,28] CPU in [0,1]
    mode:
      - "auto": if values look like [-1,1] -> map to [0,1], else clamp [0,1]
      - "0_1": clamp to [0,1]
      - "minus1_1": map (x+1)/2 then clamp
    """
    x = x_flat.detach()
    if mode == "minus1_1":
        x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)
    elif mode == "0_1":
        x = x.clamp(0.0, 1.0)
    else:
        # auto
        xmin = float(x.min().item())
        xmax = float(x.max().item())
        if (xmin < -0.1) and (xmax <= 1.2) and (xmin >= -1.2):
            x = (x + 1.0) * 0.5
        x = x.clamp(0.0, 1.0)

    x = x.view(-1, 1, 28, 28).cpu()
    return x


def load_pixel_baseline_ddpm(
    ckpt_path: str,
    device: torch.device,
    T: Optional[int] = None,
    beta_1: Optional[float] = None,
    beta_T: Optional[float] = None,
) -> DDPM:
    payload = torch.load(ckpt_path, map_location="cpu")

    # try to get config from checkpoint if present
    cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
    T = int(T if T is not None else cfg.get("T", 1000))
    beta_1 = float(beta_1 if beta_1 is not None else cfg.get("beta_1", 1e-4))
    beta_T = float(beta_T if beta_T is not None else cfg.get("beta_T", 2e-2))

    unet = ScoreNet(marginal_prob_std=lambda t: torch.ones_like(t).to(device))
    model = DDPM(unet, T=T, beta_1=beta_1, beta_T=beta_T).to(device)

    # load EMA if available
    if isinstance(payload, dict) and "ema_state_dict" in payload:
        model.load_state_dict(payload["ema_state_dict"])
    elif isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_state_dict(payload)

    model.eval()
    print(f"Loaded pixel baseline DDPM: {ckpt_path} (T={T}, beta_1={beta_1}, beta_T={beta_T})")
    return model


def load_pixel_flow_matching(
    ckpt_path: str,
    device: torch.device,
) -> RectifiedFlow:
    payload = torch.load(ckpt_path, map_location="cpu")
    unet = ScoreNet(marginal_prob_std=lambda t: torch.ones_like(t).to(device))
    model = RectifiedFlow(unet).to(device)

    if isinstance(payload, dict) and "state_dict" in payload:
        model.load_state_dict(payload["state_dict"])
    else:
        model.load_state_dict(payload)

    model.eval()
    print(f"Loaded pixel Flow Matching: {ckpt_path}")
    return model


# =============================================================================
# Main evaluator: latent runs + optional pixel models
# =============================================================================
def evaluate_all_models(
    run_root: str,
    out_dir: str,
    data_dir: str,
    device: torch.device,
    feature_net_ckpt: str = "mnist_feature_net.pt",
    # Pixel models (optional)
    pixel_ckpt_dir: Optional[str] = None,
    baseline_ckpt_name: Optional[str] = None,
    flow_ckpt_name: Optional[str] = None,
    pixel_output_mode: str = "auto",  # "auto" | "0_1" | "minus1_1"
    baseline_T: Optional[int] = None,
    baseline_beta_1: Optional[float] = None,
    baseline_beta_T: Optional[float] = None,
    flow_steps: int = 100,
    flow_solver: str = "heun",
    # common
    n_gen: int = 5000,
    validity_thresh: float = 0.9,
    num_workers: int = 2,
    seed_sampling: int = 0,
    quickcheck: bool = True,
    quickcheck_cols: int = 12,
) -> List[Dict[str, Any]]:
    _ensure_dir(out_dir)
    qc_dir = os.path.join(out_dir, "quickcheck")
    _ensure_dir(qc_dir)

    feat_net = load_feature_net(feature_net_ckpt, device=device, feat_dim=128)
    feat_real, fixed_x12 = compute_or_load_real_features(
        feat_net=feat_net,
        data_dir=data_dir,
        device=device,
        out_dir=out_dir,
        n_val=10_000,
        seed=123,
        batch_size=256,
        num_workers=num_workers,
    )

    # deterministic sampling
    torch.manual_seed(seed_sampling)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_sampling)

    rows: List[Dict[str, Any]] = []
    label_hists: List[Tuple[np.ndarray, str]] = []
    fid_points: List[Tuple[float, str]] = []
    val_points: List[Tuple[float, float, str]] = []

    # -------------------------------------------------------------------------
    # (A) Latent diffusion runs
    # -------------------------------------------------------------------------
    run_dirs = list_diffusion_runs(run_root)
    for ddir in run_dirs:
        name = os.path.basename(ddir)
        cfg = _read_json(os.path.join(ddir, "config.json")) or {}
        vae_run_dir = cfg.get("vae_run_dir", None)
        if vae_run_dir is None:
            print(f"[Skip] {name}: missing vae_run_dir in config.json")
            continue

        ddpm_path = os.path.join(ddir, "models", "ddpm.pt")
        cont_path = os.path.join(ddir, "models", "cont.pt")
        latent_path = os.path.join(ddir, "latents", "mnist_latents_mu.pt")
        if not os.path.exists(latent_path):
            print(f"[Skip] {name}: missing latents/mnist_latents_mu.pt")
            continue

        vae = load_vae_from_run(vae_run_dir, device=device)
        latent_ckpt = load_latent_checkpoint(latent_path)
        mean = latent_ckpt["mean"]
        std = latent_ckpt["std"]
        latent_dim = int(latent_ckpt["meta"]["latent_dim"])

        # quickcheck recon: 2 rows × 12
        if quickcheck:
            xN = fixed_x12[:quickcheck_cols].cpu()
            recon_pdf = os.path.join(qc_dir, f"{name}__recon_2x{quickcheck_cols}.pdf")
            fig = fig_recon_2xN(vae, xN, device=device, title=f"{name} — Recon (original + recon)")
            fig.savefig(recon_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)

        # ---- vae_prior
        _cuda_sync_if_needed(device)
        t0 = time.perf_counter()
        imgs_vae = generate_vae_prior_samples(vae, device=device, n=n_gen)
        _cuda_sync_if_needed(device)
        sampling_time = time.perf_counter() - t0

        m = classifier_metrics(feat_net, imgs_vae, device=device, validity_thresh=validity_thresh)
        fid = fid_from_features(feat_real, m["features"])

        label = f"{name} :: vae_prior"
        rows.append({
            "run_name": name,
            "space": "latent",
            "model_type": "vae_prior",
            "latent_dim": latent_dim,
            "mnist_fid": fid,
            "mean_conf": m["mean_conf"],
            "unique_digits": m["unique_digits"],
            "sampling_time_s": float(sampling_time),
            "gen_validity": m["validity"],
            "gen_label_entropy_nats": m["label_entropy_nats"],
        })
        label_hists.append((m["counts"], label))
        fid_points.append((fid, label))
        val_points.append((m["validity"], m["mean_conf"], label))

        if quickcheck:
            gen_pdf = os.path.join(qc_dir, f"{name}__gen__vae_prior_1x{quickcheck_cols}.pdf")
            fig = fig_images_1xN(imgs_vae[:quickcheck_cols].cpu(), title=f"{name} — Generated (vae_prior)")
            fig.savefig(gen_pdf, format="pdf", bbox_inches="tight")
            plt.close(fig)

        print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")

        # ---- ddpm
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

            _cuda_sync_if_needed(device)
            t0 = time.perf_counter()
            z_scaled = ddpm.sample(n_gen)
            imgs = decode_scaled_latents(vae, z_scaled, mean, std, device=device)
            _cuda_sync_if_needed(device)
            sampling_time = time.perf_counter() - t0

            m = classifier_metrics(feat_net, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])
            label = f"{name} :: ddpm"

            rows.append({
                "run_name": name,
                "space": "latent",
                "model_type": "ddpm",
                "latent_dim": latent_dim,
                "mnist_fid": fid,
                "mean_conf": m["mean_conf"],
                "unique_digits": m["unique_digits"],
                "sampling_time_s": float(sampling_time),
                "gen_validity": m["validity"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            if quickcheck:
                gen_pdf = os.path.join(qc_dir, f"{name}__gen__ddpm_1x{quickcheck_cols}.pdf")
                fig = fig_images_1xN(imgs[:quickcheck_cols].cpu(), title=f"{name} — Generated (ddpm)")
                fig.savefig(gen_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")
        else:
            print(f"[Warn] {name}: missing models/ddpm.pt")

        # ---- cont
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

            _cuda_sync_if_needed(device)
            t0 = time.perf_counter()
            z_scaled = cont.sample(n_gen, steps=steps)
            imgs = decode_scaled_latents(vae, z_scaled, mean, std, device=device)
            _cuda_sync_if_needed(device)
            sampling_time = time.perf_counter() - t0

            m = classifier_metrics(feat_net, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])
            label = f"{name} :: cont"

            rows.append({
                "run_name": name,
                "space": "latent",
                "model_type": "cont",
                "latent_dim": latent_dim,
                "mnist_fid": fid,
                "mean_conf": m["mean_conf"],
                "unique_digits": m["unique_digits"],
                "sampling_time_s": float(sampling_time),
                "gen_validity": m["validity"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            if quickcheck:
                gen_pdf = os.path.join(qc_dir, f"{name}__gen__cont_1x{quickcheck_cols}.pdf")
                fig = fig_images_1xN(imgs[:quickcheck_cols].cpu(), title=f"{name} — Generated (cont)")
                fig.savefig(gen_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")
        else:
            print(f"[Warn] {name}: missing models/cont.pt")

    # -------------------------------------------------------------------------
    # (B) Pixel models (optional)
    # -------------------------------------------------------------------------
    if pixel_ckpt_dir and (baseline_ckpt_name or flow_ckpt_name):
        pixel_ckpt_dir = os.path.abspath(pixel_ckpt_dir)

        # Baseline DDPM
        if baseline_ckpt_name:
            ckpt_path = os.path.join(pixel_ckpt_dir, baseline_ckpt_name)
            model_name = _basename_no_ext(baseline_ckpt_name)
            baseline = load_pixel_baseline_ddpm(
                ckpt_path=ckpt_path,
                device=device,
                T=baseline_T,
                beta_1=baseline_beta_1,
                beta_T=baseline_beta_T,
            )

            _cuda_sync_if_needed(device)
            t0 = time.perf_counter()
            x_flat = baseline.sample((n_gen, 28 * 28))
            _cuda_sync_if_needed(device)
            sampling_time = time.perf_counter() - t0

            imgs = _postprocess_pixel_samples(x_flat, mode=pixel_output_mode)
            m = classifier_metrics(feat_net, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])
            label = f"{model_name} :: pixel_ddpm"

            rows.append({
                "run_name": model_name,
                "space": "pixel",
                "model_type": "pixel_ddpm",
                "latent_dim": 28 * 28,
                "mnist_fid": fid,
                "mean_conf": m["mean_conf"],
                "unique_digits": m["unique_digits"],
                "sampling_time_s": float(sampling_time),
                "gen_validity": m["validity"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            if quickcheck:
                gen_pdf = os.path.join(qc_dir, f"{model_name}__gen__pixel_ddpm_1x{quickcheck_cols}.pdf")
                fig = fig_images_1xN(imgs[:quickcheck_cols].cpu(), title=f"{model_name} — Generated (pixel_ddpm)")
                fig.savefig(gen_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")

        # Flow Matching
        if flow_ckpt_name:
            ckpt_path = os.path.join(pixel_ckpt_dir, flow_ckpt_name)
            model_name = _basename_no_ext(flow_ckpt_name)
            flow = load_pixel_flow_matching(ckpt_path=ckpt_path, device=device)

            _cuda_sync_if_needed(device)
            t0 = time.perf_counter()
            x_flat = flow.sample((n_gen, 28 * 28), steps=flow_steps, solver=flow_solver)
            _cuda_sync_if_needed(device)
            sampling_time = time.perf_counter() - t0

            imgs = _postprocess_pixel_samples(x_flat, mode=pixel_output_mode)
            m = classifier_metrics(feat_net, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])
            label = f"{model_name} :: flow_matching"

            rows.append({
                "run_name": model_name,
                "space": "pixel",
                "model_type": "flow_matching",
                "latent_dim": 28 * 28,
                "mnist_fid": fid,
                "mean_conf": m["mean_conf"],
                "unique_digits": m["unique_digits"],
                "sampling_time_s": float(sampling_time),
                "gen_validity": m["validity"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            if quickcheck:
                gen_pdf = os.path.join(qc_dir, f"{model_name}__gen__flow_matching_1x{quickcheck_cols}.pdf")
                fig = fig_images_1xN(imgs[:quickcheck_cols].cpu(), title=f"{model_name} — Generated (flow_matching)")
                fig.savefig(gen_pdf, format="pdf", bbox_inches="tight")
                plt.close(fig)

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")

    # -------------------------------------------------------------------------
    # Save summary.csv
    # -------------------------------------------------------------------------
    if not rows:
        print("No models evaluated; nothing to save.")
        return []

    csv_path = os.path.join(out_dir, "summary.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"Saved: {csv_path}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    # Validity/Confidence scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, c, label in val_points:
        ax.scatter(v, c, s=35)
        ax.annotate(label, (v, c), fontsize=7, alpha=0.9)
    ax.set_xlabel(f"Validity (P(max) ≥ {validity_thresh})")
    ax.set_ylabel("Mean confidence")
    ax.set_title("Sample quality (fixed MNISTFeatureNet)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_validity.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # FID bar plot (sorted)
    fid_sorted = sorted(fid_points, key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([n for _, n in fid_sorted], [v for v, _ in fid_sorted])
    ax.set_ylabel("MNIST-FID (feature space)")
    ax.set_title("MNIST-FID using MNISTFeatureNet (lower is better)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fid_scores.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Label histograms multi-page
    hist_pdf_path = os.path.join(out_dir, "label_histograms.pdf")
    with PdfPages(hist_pdf_path) as pdf:
        for counts, label in label_hists:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(list(range(10)), counts)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Count")
            ax.set_title(f"Label histogram — {label}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved: {hist_pdf_path}")

    print(f"Saved quickcheck PDFs in: {qc_dir}")
    return rows
