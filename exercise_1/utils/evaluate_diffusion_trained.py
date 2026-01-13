# evaluate_diffusion.py
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

from utils.VAE import BetaVAE
from utils.latent_utils import load_latent_checkpoint
from utils.diffusion import LatentDenoiser, LatentDDPM, VPSDE, load_vae_from_run


# =========================
# IO helpers
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_diffusion_runs(run_root: str) -> List[str]:
    if not os.path.exists(run_root):
        return []
    out = []
    for name in sorted(os.listdir(run_root)):
        p = os.path.join(run_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            out.append(p)
    return out


# =========================
# Fixed evaluator: MNISTFeatureNet
# =========================
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


# =========================
# Real features cache (for MNIST-FID)
# =========================
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
      fixed_x12: [12,1,28,28] CPU (first 12 from the same fixed val split) for quickcheck recon
    """
    _ensure_dir(out_dir)
    cache_path = os.path.join(out_dir, "mnist_real_features.pt")

    transform = transforms.ToTensor()
    full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    n_train = len(full) - n_val
    _, ds_val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_val12 = DataLoader(ds_val, batch_size=12, shuffle=False, num_workers=num_workers, pin_memory=True)

    fixed_x12, _ = next(iter(dl_val12))
    fixed_x12 = fixed_x12.cpu()

    if os.path.exists(cache_path):
        ck = torch.load(cache_path, map_location="cpu")
        return ck["feat_real"], fixed_x12

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


# =========================
# Metrics
# =========================
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
        x = images[i:i+batch_size].to(device)
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
    if valid_mask.any():
        unique_digits = int(torch.unique(preds[valid_mask]).numel())
    else:
        unique_digits = 0

    return {
        "validity": float(valid_mask.float().mean().item()),
        "mean_conf": float(confs.mean().item()),
        "counts": counts.numpy(),
        "label_entropy_nats": entropy,
        "unique_digits": unique_digits,
        "features": feats,       # CPU
        "preds": preds,          # CPU
        "confs": confs,          # CPU
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


# =========================
# Sampling helpers
# =========================
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


# =========================
# Quickcheck plots
# =========================
@torch.no_grad()
def fig_recon_grid(vae: BetaVAE, x12_cpu: torch.Tensor, device: torch.device, title: str) -> plt.Figure:
    vae.eval()
    x = x12_cpu.to(device)
    # deterministic reconstruction via mu (avoid sampling noise)
    logits, mu, logvar, z = vae(x, sample_latent=False)
    x_hat = torch.sigmoid(logits).detach().cpu()

    n = x12_cpu.size(0)
    fig, axes = plt.subplots(2, n, figsize=(1.2 * n, 3.0))
    for i in range(n):
        axes[0, i].imshow(x12_cpu[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i, 0], cmap="gray")
        axes[1, i].axis("off")
    axes[0, 0].set_title("Original", fontsize=10)
    axes[1, 0].set_title("Reconstruction", fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def fig_samples_grid(samples_by_type: Dict[str, torch.Tensor], title: str) -> plt.Figure:
    """
    samples_by_type: dict model_type -> images [12,1,28,28] CPU
    Creates a grid: rows = model types, cols = 12
    """
    types = list(samples_by_type.keys())
    rows = len(types)
    cols = samples_by_type[types[0]].size(0)

    fig, axes = plt.subplots(rows, cols, figsize=(1.2 * cols, 1.4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, t in enumerate(types):
        imgs = samples_by_type[t]
        for c in range(cols):
            axes[r, c].imshow(imgs[c, 0], cmap="gray")
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(t, rotation=0, labelpad=20, fontsize=10, va="center")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


# =========================
# Main evaluator
# =========================
def evaluate_diffusion_runs(
    run_root: str,
    out_dir: str,
    data_dir: str,
    device: torch.device,
    feature_net_ckpt: str = "mnist_feature_net.pt",
    n_gen: int = 5000,
    validity_thresh: float = 0.9,
    num_workers: int = 2,
    seed_sampling: int = 0,
    quickcheck: bool = True,
    quickcheck_cols: int = 12,
) -> List[Dict[str, Any]]:
    _ensure_dir(out_dir)

    # fixed evaluator
    feat_net = load_feature_net(feature_net_ckpt, device=device, feat_dim=128)

    # fixed real features + fixed 12 images for recon
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

    run_dirs = list_diffusion_runs(run_root)
    if not run_dirs:
        print(f"No diffusion runs found in: {run_root}")
        return []

    # deterministic sampling if desired
    torch.manual_seed(seed_sampling)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_sampling)

    rows: List[Dict[str, Any]] = []
    label_hists: List[Tuple[np.ndarray, str]] = []
    fid_points: List[Tuple[float, str]] = []
    val_points: List[Tuple[float, float, str]] = []

    qc_pdf = os.path.join(out_dir, "quickcheck.pdf")
    qc_writer = PdfPages(qc_pdf) if quickcheck else None

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

        # quickcheck: recon page (VAE-only, but attached to diffusion run)
        if qc_writer is not None:
            fig = fig_recon_grid(vae, fixed_x12[:quickcheck_cols], device=device,
                                 title=f"{name} — Reconstructions (VAE)")
            qc_writer.savefig(fig)
            plt.close(fig)

        # For sample-grid quickcheck
        qc_samples: Dict[str, torch.Tensor] = {}

        # -------- vae_prior (baseline) with timing
        t0 = time.perf_counter()
        imgs_vae = generate_vae_prior_samples(vae, device=device, n=n_gen)
        sampling_time = time.perf_counter() - t0

        m = classifier_metrics(feat_net, imgs_vae, device=device, validity_thresh=validity_thresh)
        fid = fid_from_features(feat_real, m["features"])

        label = f"{name} :: vae_prior"
        rows.append({
            "run_name": name,
            "model_type": "vae_prior",
            "latent_dim": latent_dim,
            "mnist_fid": fid,
            "mean_conf": m["mean_conf"],
            "unique_digits": m["unique_digits"],
            "sampling_time_s": float(sampling_time),
            "gen_validity": m["validity"],
            "gen_label_entropy_nats": m["label_entropy_nats"],
            "diffusion_run_dir": ddir,
            "vae_run_dir": vae_run_dir,
        })
        label_hists.append((m["counts"], label))
        fid_points.append((fid, label))
        val_points.append((m["validity"], m["mean_conf"], label))

        print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")

        if qc_writer is not None:
            qc_samples["vae_prior"] = imgs_vae[:quickcheck_cols].cpu()

        # -------- DDPM
        ddpm = None
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

            t0 = time.perf_counter()
            z_scaled = ddpm.sample(n_gen)
            imgs = decode_scaled_latents(vae, z_scaled, mean, std, device=device)
            sampling_time = time.perf_counter() - t0

            m = classifier_metrics(feat_net, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            label = f"{name} :: ddpm"
            rows.append({
                "run_name": name,
                "model_type": "ddpm",
                "latent_dim": latent_dim,
                "mnist_fid": fid,
                "mean_conf": m["mean_conf"],
                "unique_digits": m["unique_digits"],
                "sampling_time_s": float(sampling_time),
                "gen_validity": m["validity"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
                "diffusion_run_dir": ddir,
                "vae_run_dir": vae_run_dir,
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")

            if qc_writer is not None:
                qc_samples["ddpm"] = imgs[:quickcheck_cols].cpu()
        else:
            print(f"[Warn] {name}: missing models/ddpm.pt")

        # -------- Continuous VP-SDE
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

            t0 = time.perf_counter()
            z_scaled = cont.sample(n_gen, steps=steps)
            imgs = decode_scaled_latents(vae, z_scaled, mean, std, device=device)
            sampling_time = time.perf_counter() - t0

            m = classifier_metrics(feat_net, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            label = f"{name} :: cont"
            rows.append({
                "run_name": name,
                "model_type": "cont",
                "latent_dim": latent_dim,
                "mnist_fid": fid,
                "mean_conf": m["mean_conf"],
                "unique_digits": m["unique_digits"],
                "sampling_time_s": float(sampling_time),
                "gen_validity": m["validity"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
                "diffusion_run_dir": ddir,
                "vae_run_dir": vae_run_dir,
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            print(f"[Eval] {label} | fid={fid:.2f} conf={m['mean_conf']:.3f} uniq={m['unique_digits']} time={sampling_time:.2f}s")

            if qc_writer is not None:
                qc_samples["cont"] = imgs[:quickcheck_cols].cpu()
        else:
            print(f"[Warn] {name}: missing models/cont.pt")

        # quickcheck: generated samples page
        if qc_writer is not None and len(qc_samples) > 0:
            fig = fig_samples_grid(qc_samples, title=f"{name} — Generated samples")
            qc_writer.savefig(fig)
            plt.close(fig)

    if qc_writer is not None:
        qc_writer.close()
        print(f"Saved: {qc_pdf}")

    # =========================
    # Save summary.csv
    # =========================
    csv_path = os.path.join(out_dir, "summary.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"Saved: {csv_path}")

    # =========================
    # Plots
    # =========================
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

    fid_sorted = sorted(fid_points, key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([n for _, n in fid_sorted], [v for v, _ in fid_sorted])
    ax.set_ylabel("MNIST-FID (feature space)")
    ax.set_title("MNIST-FID using MNISTFeatureNet (lower is better)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fid_scores.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

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

    return rows
