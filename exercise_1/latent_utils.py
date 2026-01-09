# latent_utils.py
"""
Utilities to build and load latent datasets produced by a trained VAE.
These are used for training a latent diffusion model.

Saves:
- z_mu: [N, d] deterministic latents (mu)
- y:    [N]    labels
- mean: [d]    dataset mean of z_mu
- std:  [d]    dataset std of z_mu
"""

from __future__ import annotations

import os
from typing import Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset


@torch.no_grad()
def build_latent_dataset(
    vae,
    loader,
    device: torch.device,
    save_path: str = "mnist_latents_mu.pt",
    max_items: Optional[int] = None,
) -> str:
    """
    Encode a dataset into deterministic latents z=mu(x), compute mean/std (Welford),
    and save to a .pt file.

    Args:
        vae: trained VAE model (must implement encode_mu or encode returning mu)
        loader: DataLoader yielding (x,y)
        device: torch device for encoding
        save_path: output path to .pt
        max_items: optionally cap how many items to encode

    Returns:
        save_path
    """
    vae.eval()

    zs, ys = [], []

    n = 0
    mean = None
    M2 = None  # Welford accumulator

    latent_dim = getattr(vae, "latent_dim", None)

    for x, y in loader:
        x = x.to(device)

        # Prefer deterministic latents
        if hasattr(vae, "encode_mu"):
            mu = vae.encode_mu(x)  # [B, d]
        else:
            mu, _ = vae.encode(x)  # fallback

        mu = mu.detach().cpu()
        y = y.detach().cpu()

        if latent_dim is None:
            latent_dim = mu.size(1)

        # Apply max_items cap (optional)
        if max_items is not None:
            remaining = max_items - n
            if remaining <= 0:
                break
            if mu.size(0) > remaining:
                mu = mu[:remaining]
                y = y[:remaining]

        zs.append(mu)
        ys.append(y)

        # Welford update
        b = mu.size(0)
        if b == 0:
            continue

        if mean is None:
            mean = mu.mean(dim=0)
            M2 = ((mu - mean) ** 2).sum(dim=0)
            n = b
        else:
            n_new = n + b
            batch_mean = mu.mean(dim=0)
            batch_M2 = ((mu - batch_mean) ** 2).sum(dim=0)

            delta = batch_mean - mean
            mean = mean + delta * (b / n_new)
            M2 = M2 + batch_M2 + delta**2 * (n * b / n_new)
            n = n_new

    z = torch.cat(zs, dim=0) if zs else torch.empty((0, latent_dim or 0))
    y = torch.cat(ys, dim=0) if ys else torch.empty((0,), dtype=torch.long)

    if n >= 2:
        var = M2 / (n - 1)
        std = torch.sqrt(var + 1e-8)
    else:
        # degenerate fallback
        std = torch.ones(latent_dim)
        mean = torch.zeros(latent_dim)

    payload: Dict[str, Any] = {
        "z_mu": z.float(),              # [N, d]
        "y": y.long(),                  # [N]
        "mean": mean.float(),           # [d]
        "std": std.float(),             # [d]
        "meta": {
            "latent_type": "mu",
            "N": int(z.size(0)),
            "latent_dim": int(z.size(1)) if z.numel() else int(latent_dim or 0),
        },
    }

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(payload, save_path)
    print(f"Saved latent dataset to: {save_path}")
    print(f"z_mu: {tuple(z.shape)} | mean/std: {tuple(mean.shape)} / {tuple(std.shape)}")
    return save_path


def load_latent_checkpoint(path: str) -> Dict[str, Any]:
    """Load latent dataset checkpoint on CPU (safe default)."""
    return torch.load(path, map_location="cpu")


class LatentDataset(Dataset):
    """
    Dataset yielding (z, y) where z is either raw mu-latent or standardized.
    Standardization uses per-dimension mean/std saved in the checkpoint.
    """
    def __init__(self, path: str, scaled: bool = True):
        ckpt = load_latent_checkpoint(path)
        self.z = ckpt["z_mu"].float()
        self.y = ckpt["y"].long()
        self.mean = ckpt["mean"].float()
        self.std = ckpt["std"].float()
        self.scaled = scaled

    def __len__(self) -> int:
        return int(self.z.size(0))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.z[idx]
        if self.scaled:
            z = (z - self.mean) / self.std
        return z, self.y
