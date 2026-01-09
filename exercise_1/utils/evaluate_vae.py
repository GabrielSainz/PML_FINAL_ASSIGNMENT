# evaluate_vae.py
"""
Evaluate and compare saved BetaVAE runs under a run_root folder.

Outputs (in out_dir):
- summary.csv
- rate_distortion.pdf
- sample_validity.pdf
- fid_scores.pdf
- label_histograms.pdf

Assumes each run_dir contains:
- vae_model.pt  (checkpoint dict with model_state and latent_dim, or raw state_dict)
- config.json   (optional)
- history.json  (optional)

Usage (notebook):
    from evaluate_vae import evaluate_all_runs

    df = evaluate_all_runs(
        run_root="./runs_vae",
        out_dir="./runs_vae/summary",
        data_dir="./mnist_data",
        device=device,
        n_gen=5000,
        validity_thresh=0.9,
        clf_epochs=3,
    )
"""

from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# -------------------------
# Load your VAE model class
# -------------------------
# This expects vae.py in the same directory / PYTHONPATH
from vae import BetaVAE, beta_vae_loss


# =========================
# Small utilities
# =========================
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_run_dirs(run_root: str) -> List[str]:
    if not os.path.exists(run_root):
        return []
    dirs = []
    for name in sorted(os.listdir(run_root)):
        p = os.path.join(run_root, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "vae_model.pt")):
            dirs.append(p)
    return dirs


def load_vae_from_run(run_dir: str, device: torch.device) -> BetaVAE:
    """
    Loads VAE from run_dir/vae_model.pt.
    Supports two formats:
    - checkpoint dict with keys: model_state, latent_dim (preferred)
    - raw state_dict (fallback, requires latent_dim from config.json)
    """
    ckpt_path = os.path.join(run_dir, "vae_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        latent_dim = int(ckpt.get("latent_dim", ckpt.get("cfg", {}).get("latent_dim", 16)))
        model = BetaVAE(latent_dim=latent_dim).to(device)
        model.load_state_dict(ckpt["model_state"])
        return model

    # fallback: raw state_dict
    cfg = _read_json(os.path.join(run_dir, "config.json")) or {}
    latent_dim = int(cfg.get("latent_dim", 16))
    model = BetaVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt)
    return model


# =========================
# MNIST classifier for evaluation
# =========================
class MNISTClassifier(nn.Module):
    """
    Simple CNN classifier.
    forward(x) returns (logits, features) where features are penultimate embeddings.
    """
    def __init__(self, feat_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),   # 28->14
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),  # 14->7
            nn.ReLU(inplace=True),
        )
        self.fc_feat = nn.Linear(128 * 7 * 7, feat_dim)
        self.fc_out = nn.Linear(feat_dim, 10)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x)
        h = h.view(x.size(0), -1)
        feat = F.relu(self.fc_feat(h))
        logits = self.fc_out(feat)
        return logits, feat


@torch.no_grad()
def _clf_accuracy(model: MNISTClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return correct / max(1, total)


def train_or_load_classifier(
    data_dir: str,
    device: torch.device,
    out_dir: str,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 1e-3,
    num_workers: int = 2,
) -> MNISTClassifier:
    """
    Trains a small MNIST classifier (or loads if checkpoint exists).
    Saves to out_dir/mnist_classifier.pt
    """
    _ensure_dir(out_dir)
    ckpt_path = os.path.join(out_dir, "mnist_classifier.pt")

    transform = transforms.ToTensor()
    full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    n_val = 10_000
    n_train = len(full) - n_val
    ds_train, ds_val = random_split(full, [n_train, n_val])

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = MNISTClassifier().to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        return model

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for ep in range(1, epochs + 1):
        running = 0.0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item())

        acc = _clf_accuracy(model, dl_val, device)
        print(f"[Classifier] epoch {ep}/{epochs} loss={running/len(dl_train):.4f} val_acc={acc:.4f}")

    torch.save({"state_dict": model.state_dict()}, ckpt_path)
    return model


# =========================
# FID (feature space)
# =========================
def _cov(x: torch.Tensor) -> torch.Tensor:
    # x: [N, d], centered assumed
    n = x.size(0)
    return (x.T @ x) / max(1, (n - 1))


def fid_from_features(feat_real: torch.Tensor, feat_gen: torch.Tensor, eps: float = 1e-6) -> float:
    """
    FID between two sets of features using symmetric PSD trick:
        sqrtm( sqrt(C1) C2 sqrt(C1) )
    """
    feat_real = feat_real.float()
    feat_gen = feat_gen.float()

    m1 = feat_real.mean(dim=0)
    m2 = feat_gen.mean(dim=0)

    x1 = feat_real - m1
    x2 = feat_gen - m2

    C1 = _cov(x1) + eps * torch.eye(x1.size(1))
    C2 = _cov(x2) + eps * torch.eye(x2.size(1))

    # sqrt(C1) via eig
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
# VAE evaluation primitives
# =========================
@torch.no_grad()
def eval_recon_kl(
    vae: BetaVAE,
    loader: DataLoader,
    device: torch.device,
    beta_for_loss: float = 1.0,
    deterministic_recon: bool = True,
) -> Tuple[float, float]:
    """
    Returns (mean_recon, mean_kl) per image on loader.
    recon uses BCE sum over pixels (as in your training).
    """
    vae.eval()
    recon_sum = 0.0
    kl_sum = 0.0
    n = 0

    for x, _ in loader:
        x = x.to(device)
        logits, mu, logvar, _ = vae(x, sample_latent=not deterministic_recon)
        _, recon, kl = beta_vae_loss(x, logits, mu, logvar, beta=beta_for_loss)
        b = x.size(0)
        recon_sum += float(recon.item()) * b
        kl_sum += float(kl.item()) * b
        n += b

    return recon_sum / max(1, n), kl_sum / max(1, n)


@torch.no_grad()
def generate_prior_samples(vae: BetaVAE, device: torch.device, n: int, batch_size: int = 512) -> torch.Tensor:
    """
    Generates samples from z ~ N(0,I), returns images in [0,1] as [N,1,28,28] on CPU.
    """
    vae.eval()
    imgs = []
    remaining = n
    while remaining > 0:
        b = min(batch_size, remaining)
        z = torch.randn(b, vae.latent_dim, device=device)
        x = vae.decode(z).detach().cpu()
        imgs.append(x)
        remaining -= b
    return torch.cat(imgs, dim=0)


@torch.no_grad()
def classifier_metrics(
    clf: MNISTClassifier,
    images: torch.Tensor,
    device: torch.device,
    validity_thresh: float = 0.9,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """
    images: [N,1,28,28] on CPU
    Returns metrics + features for FID.
    """
    clf.eval()
    N = images.size(0)

    preds = []
    confs = []
    feats = []

    for i in range(0, N, batch_size):
        x = images[i:i+batch_size].to(device)
        logits, feat = clf(x)
        prob = F.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        preds.append(pred.detach().cpu())
        confs.append(conf.detach().cpu())
        feats.append(feat.detach().cpu())

    preds = torch.cat(preds, dim=0)
    confs = torch.cat(confs, dim=0)
    feats = torch.cat(feats, dim=0)

    validity = float((confs >= validity_thresh).float().mean().item())
    mean_conf = float(confs.mean().item())

    counts = torch.bincount(preds, minlength=10).float()
    p = counts / counts.sum().clamp(min=1.0)
    label_entropy = float(-(p * (p + 1e-12).log()).sum().item())  # nats

    return {
        "validity": validity,
        "mean_conf": mean_conf,
        "counts": counts.numpy(),
        "label_entropy": label_entropy,
        "features": feats,  # torch tensor CPU
    }


# =========================
# Main evaluator (multiple runs)
# =========================
def evaluate_all_runs(
    run_root: str,
    out_dir: str,
    data_dir: str,
    device: torch.device,
    n_gen: int = 5000,
    validity_thresh: float = 0.9,
    clf_epochs: int = 3,
    batch_size_val: int = 256,
    num_workers: int = 2,
) -> "np.ndarray":
    """
    Evaluates all run dirs in run_root and writes summary + plots to out_dir.
    Returns a list of dicts (also saved to CSV).
    """
    _ensure_dir(out_dir)

    # Val loader (fixed split for comparability)
    transform = transforms.ToTensor()
    full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    n_val = 10_000
    n_train = len(full) - n_val
    _, ds_val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(123))

    dl_val = DataLoader(ds_val, batch_size=batch_size_val, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    # Classifier (cache in out_dir)
    clf = train_or_load_classifier(
        data_dir=data_dir,
        device=device,
        out_dir=out_dir,
        epochs=clf_epochs,
    )

    # Real feature stats (for FID baseline)
    feat_real = []
    with torch.no_grad():
        for x, _ in dl_val:
            x = x.to(device)
            _, feat = clf(x)
            feat_real.append(feat.detach().cpu())
    feat_real = torch.cat(feat_real, dim=0)

    run_dirs = list_run_dirs(run_root)
    if len(run_dirs) == 0:
        print(f"No runs found in: {run_root}")
        return []

    rows: List[Dict[str, Any]] = []

    # For plotting
    rd_points = []  # (kl, recon, label)
    fid_points = []
    val_points = []  # (validity, conf, label)
    label_hists = []

    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)

        cfg = _read_json(os.path.join(run_dir, "config.json")) or {}
        history = _read_json(os.path.join(run_dir, "history.json")) or {}

        vae = load_vae_from_run(run_dir, device=device)

        # recon/kl on val (deterministic recon by default)
        recon, kl = eval_recon_kl(
            vae, dl_val, device=device,
            beta_for_loss=1.0,
            deterministic_recon=True,
        )

        # prior samples + classifier metrics
        imgs = generate_prior_samples(vae, device=device, n=n_gen, batch_size=512)
        cm = classifier_metrics(clf, imgs, device=device, validity_thresh=validity_thresh)

        fid = fid_from_features(feat_real, cm["features"])

        row = {
            "run_name": run_name,
            "run_dir": run_dir,
            "latent_dim": int(cfg.get("latent_dim", getattr(vae, "latent_dim", -1))),
            "beta": float(cfg.get("beta", float("nan"))),
            "val_recon": recon,
            "val_kl": kl,
            "gen_validity": cm["validity"],
            "gen_mean_conf": cm["mean_conf"],
            "gen_label_entropy_nats": cm["label_entropy"],
            "fid_feat": fid,
        }
        rows.append(row)

        rd_points.append((kl, recon, run_name))
        fid_points.append((fid, run_name))
        val_points.append((cm["validity"], cm["mean_conf"], run_name))
        label_hists.append((cm["counts"], run_name))

        print(f"[Eval] {run_name} | recon={recon:.2f} kl={kl:.2f} "
              f"valid={cm['validity']:.3f} conf={cm['mean_conf']:.3f} fid={fid:.2f}")

    # -------------------------
    # Save summary.csv
    # -------------------------
    csv_path = os.path.join(out_dir, "summary.csv")
    cols = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"Saved: {csv_path}")

    # -------------------------
    # Plots (PDF)
    # -------------------------
    # Rate-distortion: KL vs Recon
    fig, ax = plt.subplots(figsize=(7, 5))
    for (kl, recon, name) in rd_points:
        ax.scatter(kl, recon, s=40)
        ax.annotate(name, (kl, recon), fontsize=7, alpha=0.9)
    ax.set_xlabel("Rate (KL, val)")
    ax.set_ylabel("Distortion (Recon BCE-sum, val)")
    ax.set_title("Rate–Distortion (VAE comparison)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rate_distortion.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Validity/conf plot
    fig, ax = plt.subplots(figsize=(7, 5))
    for (v, c, name) in val_points:
        ax.scatter(v, c, s=40)
        ax.annotate(name, (v, c), fontsize=7, alpha=0.9)
    ax.set_xlabel(f"Validity (P(max) ≥ {validity_thresh})")
    ax.set_ylabel("Mean confidence")
    ax.set_title("Prior sample quality (classifier-based)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_validity.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # FID plot
    fid_points_sorted = sorted(fid_points, key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([n for _, n in fid_points_sorted], [v for v, _ in fid_points_sorted])
    ax.set_ylabel("FID (feature space)")
    ax.set_title("FID-like score (lower is better)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fid_scores.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # Label histograms (multi-page PDF)
    hist_pdf_path = os.path.join(out_dir, "label_histograms.pdf")
    with PdfPages(hist_pdf_path) as pdf:
        for counts, name in label_hists:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(list(range(10)), counts)
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Count")
            ax.set_title(f"Label histogram (prior samples) — {name}")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Saved: {hist_pdf_path}")

    return rows
