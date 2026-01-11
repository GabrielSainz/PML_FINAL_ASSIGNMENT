import torch
from utils.evaluate_diffusion_trained import evaluate_diffusion_runs

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

root_path="/content/drive/MyDrive/University_of_Copenhagen/block6/PML/final_assignment/exercise_1" 
rows = evaluate_diffusion_runs(
    run_root=f"{root_path}/runs_diffusion",
    out_dir=f"{root_path}/runs_diffusion/summary_fixed_net",
    data_dir=f"{root_path}/mnist_data",
    device=device,
    feature_net_ckpt="utils/mnist_feature_net.pt",
    n_gen=5000,           # 10k if you want tighter estimates
    validity_thresh=0.9,
)

print(f"Done. See {root_path}/runs_diffusion/summary_fixed_net/")
