from utils.evaluate_vae import evaluate_all_runs
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rows = evaluate_all_runs(
    run_root="./runs_vae",
    out_dir="./runs_vae/summary",
    data_dir="./mnist_data",
    device=device,
    n_gen=5000,            # increase to 10000 if you want
    validity_thresh=0.9,
    clf_epochs=3,          # trains once and caches in summary/
)
