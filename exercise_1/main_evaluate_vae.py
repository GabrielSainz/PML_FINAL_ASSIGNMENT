from utils.evaluate_vae import evaluate_all_runs
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path="/content/drive/MyDrive/University_of_Copenhagen/block6/PML/final_assignment/exercise_1" 
rows = evaluate_all_runs(
    run_root=f"{root_path}/runs_vae",
    out_dir=f"{root_path}/runs_vae/summary",
    data_dir=f"{root_path}/mnist_data",
    device=device,
    n_gen=5000,            # increase to 10000 if you want
    validity_thresh=0.9,
    clf_epochs=3,          # trains once and caches in summary/
)
