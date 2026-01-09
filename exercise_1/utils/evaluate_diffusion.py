# evaluate_diffusion.py
"""
Evaluate latent diffusion priors (DDPM + Continuous VP-SDE) trained in runs_diffusion/.

For each diffusion run dir, we evaluate:
- vae_prior: decode z ~ N(0, I) (baseline)
- ddpm: sample z from latent DDPM prior -> unscale -> VAE decode
- cont: sample z from continuous VP-SDE prior -> unscale -> VAE decode

Outputs in out_dir:
- summary.csv
- sample_validity.pdf
- fid_scores.pdf
- label_histograms.pdf (multi-page)

Assumes each diffusion run has:
  config.json (contains "vae_run_dir" and training params)
  models/ddpm.pt
  models/cont.pt
  latents/mnist_latents_mu.pt  (mean/std for unscaling)
"""

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# From your project files
from utils.VAE import BetaVAE
from utils.latent_utils import load_latent_checkpoint

# Import diffusion model classes (same ones you trained with)
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
# MNIST classifier (same idea as evaluate_vae.py)
# =========================
class MNISTClassifier(nn.Module):
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

    for ep in range(1, epochs + 1):
        model.train()
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
# Metrics: validity/conf/entropy + FID(feature)
# =========================
@torch.no_grad()
def classifier_metrics(
    clf: MNISTClassifier,
    images: torch.Tensor,  # [N,1,28,28] CPU in [0,1]
    device: torch.device,
    validity_thresh: float = 0.9,
    batch_size: int = 512,
) -> Dict[str, Any]:
    clf.eval()
    N = images.size(0)

    preds, confs, feats = [], [], []
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

    counts = torch.bincount(preds, minlength=10).float()
    p = counts / counts.sum().clamp(min=1.0)
    entropy = float(-(p * (p + 1e-12).log()).sum().item())  # nats

    return {
        "validity": float((confs >= validity_thresh).float().mean().item()),
        "mean_conf": float(confs.mean().item()),
        "counts": counts.numpy(),
        "label_entropy_nats": entropy,
        "features": feats,  # CPU tensor
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
# Main evaluator
# =========================
def evaluate_diffusion_runs(
    run_root: str,
    out_dir: str,
    data_dir: str,
    device: torch.device,
    n_gen: int = 5000,
    validity_thresh: float = 0.9,
    clf_epochs: int = 3,
    num_workers: int = 2,
) -> List[Dict[str, Any]]:
    _ensure_dir(out_dir)

    # fixed MNIST val split for "real" features
    transform = transforms.ToTensor()
    full = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    n_val = 10_000
    n_train = len(full) - n_val
    _, ds_val = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(123))

    dl_val = DataLoader(ds_val, batch_size=256, shuffle=False, num_workers=num_workers, pin_memory=True)

    # classifier (cache in out_dir)
    clf = train_or_load_classifier(data_dir=data_dir, device=device, out_dir=out_dir, epochs=clf_epochs)

    # real features
    feat_real = []
    with torch.no_grad():
        clf.eval()
        for x, _ in dl_val:
            x = x.to(device)
            _, feat = clf(x)
            feat_real.append(feat.detach().cpu())
    feat_real = torch.cat(feat_real, dim=0)

    rows: List[Dict[str, Any]] = []
    label_hists: List[Tuple[np.ndarray, str]] = []
    fid_points: List[Tuple[float, str]] = []
    val_points: List[Tuple[float, float, str]] = []

    run_dirs = list_diffusion_runs(run_root)
    if not run_dirs:
        print(f"No diffusion runs found in: {run_root}")
        return []

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

        # Load VAE + latent stats
        vae = load_vae_from_run(vae_run_dir, device=device)
        latent_ckpt = load_latent_checkpoint(latent_path)
        mean = latent_ckpt["mean"]
        std = latent_ckpt["std"]
        latent_dim = int(latent_ckpt["meta"]["latent_dim"])

        # ---------- baseline: VAE prior ----------
        imgs_vae = generate_vae_prior_samples(vae, device=device, n=n_gen)
        m = classifier_metrics(clf, imgs_vae, device=device, validity_thresh=validity_thresh)
        fid = fid_from_features(feat_real, m["features"])

        label = f"{name} :: vae_prior"
        rows.append({
            "run_name": name,
            "model_type": "vae_prior",
            "latent_dim": latent_dim,
            "gen_validity": m["validity"],
            "gen_mean_conf": m["mean_conf"],
            "gen_label_entropy_nats": m["label_entropy_nats"],
            "fid_feat": fid,
            "diffusion_run_dir": ddir,
            "vae_run_dir": vae_run_dir,
        })
        label_hists.append((m["counts"], label))
        fid_points.append((fid, label))
        val_points.append((m["validity"], m["mean_conf"], label))

        print(f"[Eval] {label} | valid={m['validity']:.3f} conf={m['mean_conf']:.3f} fid={fid:.2f}")

        # ---------- DDPM prior ----------
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

            z_scaled = ddpm.sample(n_gen)  # [N,d] on device (scaled space)
            imgs = decode_scaled_latents(vae, z_scaled, mean, std, device=device)
            m = classifier_metrics(clf, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            label = f"{name} :: ddpm"
            rows.append({
                "run_name": name,
                "model_type": "ddpm",
                "latent_dim": latent_dim,
                "gen_validity": m["validity"],
                "gen_mean_conf": m["mean_conf"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
                "fid_feat": fid,
                "diffusion_run_dir": ddir,
                "vae_run_dir": vae_run_dir,
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            print(f"[Eval] {label} | valid={m['validity']:.3f} conf={m['mean_conf']:.3f} fid={fid:.2f}")
        else:
            print(f"[Warn] {name}: missing models/ddpm.pt")

        # ---------- Continuous prior ----------
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
            z_scaled = cont.sample(n_gen, steps=steps)
            imgs = decode_scaled_latents(vae, z_scaled, mean, std, device=device)
            m = classifier_metrics(clf, imgs, device=device, validity_thresh=validity_thresh)
            fid = fid_from_features(feat_real, m["features"])

            label = f"{name} :: cont"
            rows.append({
                "run_name": name,
                "model_type": "cont",
                "latent_dim": latent_dim,
                "gen_validity": m["validity"],
                "gen_mean_conf": m["mean_conf"],
                "gen_label_entropy_nats": m["label_entropy_nats"],
                "fid_feat": fid,
                "diffusion_run_dir": ddir,
                "vae_run_dir": vae_run_dir,
            })
            label_hists.append((m["counts"], label))
            fid_points.append((fid, label))
            val_points.append((m["validity"], m["mean_conf"], label))

            print(f"[Eval] {label} | valid={m['validity']:.3f} conf={m['mean_conf']:.3f} fid={fid:.2f}")
        else:
            print(f"[Warn] {name}: missing models/cont.pt")

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
    # Validity/Confidence scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for v, c, label in val_points:
        ax.scatter(v, c, s=35)
        ax.annotate(label, (v, c), fontsize=7, alpha=0.9)
    ax.set_xlabel(f"Validity (P(max) ≥ {validity_thresh})")
    ax.set_ylabel("Mean confidence")
    ax.set_title("Sample quality (classifier-based)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sample_validity.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)

    # FID bar plot (sorted)
    fid_sorted = sorted(fid_points, key=lambda t: t[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar([n for _, n in fid_sorted], [v for v, _ in fid_sorted])
    ax.set_ylabel("FID (feature space)")
    ax.set_title("FID-like score (lower is better)")
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

    return rows
