import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from utils.VAE import VAEConfig, train_vae

def main():
    # -------------------------
    # Device
    # -------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    root_path="/content/drive/MyDrive/University_of_Copenhagen/block6/PML/final_assignment/exercise_1" 

    batch_size = 128
    val_split = 0.1
    num_workers = 2

    transform = transforms.ToTensor()  # required for BCE loss in vae.py

    dataset_full = datasets.MNIST(f"{root_path}/mnist_data", download=True, train=True, transform=transform)
    n_val = int(len(dataset_full) * val_split)
    n_train = len(dataset_full) - n_val
    dataset_train, dataset_val = random_split(dataset_full, [n_train, n_val])

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True)

    # -------------------------
    # Experiment grid
    # -------------------------
    latent_dims = [2, 16]
    betas = [0.01, 0.1, 1.0]

    # Training knobs (shared)
    epochs = 15
    lr = 2e-3
    seed = 42

    run_root = f"{root_path}/runs_vae"  # parent folder

    # optional: keep a summary list
    runs_summary = []

    for zdim in latent_dims:
        for beta in betas:
            run_name = f"mnist_betaVAE_z{zdim}_beta{beta:g}_e{epochs}_lr{lr:g}_seed{seed}"
            print("\n" + "="*90)
            print("RUN:", run_name)
            print("="*90)

            cfg = VAEConfig(
                latent_dim=zdim,
                beta=float(beta),
                epochs=epochs,
                lr=lr,
                seed=seed,
                run_root=run_root,
                run_name=run_name,        # ensures separate folder per config
                save_best_only=True,

                # plot params
                max_images_plot=16,
                prior_n_plot=25,
                latent_grid_size=20,
                latent_grid_lim=3.0,
                latent_scatter_max_points=5000,
            )

            model, history, run_dir = train_vae(cfg, dataloader_train, dataloader_val, device)

            runs_summary.append({
                "run_name": run_name,
                "run_dir": run_dir,
                "latent_dim": zdim,
                "beta": beta,
                "final_val_loss": history["val_loss"][-1],
                "final_val_recon": history["val_recon"][-1],
                "final_val_kl": history["val_kl"][-1],
            })

    print("\nAll runs finished.")


if __name__ == "__main__":
    main()