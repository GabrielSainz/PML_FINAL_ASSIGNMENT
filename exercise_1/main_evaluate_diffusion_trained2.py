import torch
from utils.evaluate_diffusion_trained2 import evaluate_all_models

root_path="/content/drive/MyDrive/University_of_Copenhagen/block6/PML/final_assignment/exercise_1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

rows = evaluate_all_models(
    run_root=f"{root_path}/runs_diffusion",
    out_dir=f"{root_path}/runs_diffusion/summary_fixed_net_3",
    data_dir=f"{root_path}/mnist_data",
    device=device,
    feature_net_ckpt="/content/PML_FINAL_ASSIGNMENT/exercise_1/utils/models/mnist_feature_net.pt",
    pixel_ckpt_dir="/content/PML_FINAL_ASSIGNMENT/exercise_1/utils/models",
    baseline_ckpt_name="baseline_ddpm_ema.pt",
    flow_ckpt_name="flow_matching.pt",
    pixel_output_mode="minus1_1",   # auto | 0_1 | minus1_1
    n_gen=5000,
    quickcheck=True,
    quickcheck_cols=12,
)

print(f"Done. See {root_path}/runs_diffusion/summary_fixed_net_3/")