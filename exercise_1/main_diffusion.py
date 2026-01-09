import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.diffusion import DiffusionConfig, train_latent_priors

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1) Point to the VAE run you selected
path_root="/content/drive/MyDrive/University_of_Copenhagen/block6/PML/final_assignment/exercise_1"
vae_run_dir = f"{path_root}/runs_vae/mnist_betaVAE_z16_beta0.01_e15_lr0.002_seed42"

# 2) Data loader used ONLY to build latent dataset (shuffle=False recommended)
transform = transforms.ToTensor()
mnist_train = datasets.MNIST(f"{path_root}/mnist_data", download=True, train=True, transform=transform)

loader_for_latents = DataLoader(
    mnist_train,
    batch_size=256,
    shuffle=False,   # stable ordering
    num_workers=2,
    pin_memory=True
)

# 3) Diffusion config (tweak as needed)
cfg = DiffusionConfig(
    run_root=f"{path_root}/runs_diffusion",
    run_name="latent_priors_z16_beta1",  # name your run
    epochs=30,
    lr=2e-4,
    batch_size=512,
    ddpm_T=200,
    sample_steps_cont=200,
    hidden=256,
    depth=3,
    ema_decay=0.999,
    n_plot=25,
    plot_every_epochs=0,  # set e.g. 10 if you want intermediate sample PDFs
)

run_dir, metrics = train_latent_priors(
    vae_run_dir=vae_run_dir,
    train_loader_for_latents=loader_for_latents,
    device=device,
    cfg=cfg,
)

print("Done. Outputs in:", run_dir)
print(metrics)
