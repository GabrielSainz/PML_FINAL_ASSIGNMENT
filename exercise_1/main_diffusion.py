# scripts/run_latent_diffusion.py
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from utils.train_diffusion import train_diffusion_on_vae_latents


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help='e.g. "z16_beta0.01"')
    ap.add_argument(
        "--path_root",
        type=str,
        default="/content/drive/MyDrive/University_of_Copenhagen/block6/PML/final_assignment/exercise_1",
    )
    # Keep these as args only if you want flexibility; otherwise hardcode them.
    ap.add_argument("--e", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[info] device = {device}")

    path_root = Path(args.path_root)
    vae_run_dir = path_root / "runs_vae" / f"mnist_betaVAE_{args.model}_e{args.e}_lr{args.lr}_seed{args.seed}"

    #if not vae_run_dir.exists():
    #    raise FileNotFoundError(
    #        f"VAE run dir not found:\n  {vae_run_dir}\n"
    #        f"Check --model / --e / --lr / --seed or pass the correct path_root."
    #    )

    train_diffusion_on_vae_latents(args.model, str(path_root), str(vae_run_dir))


if __name__ == "__main__":
    main()
