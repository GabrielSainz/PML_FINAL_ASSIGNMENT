# Probabilistic Machine Learning — Final Assignment

Brief repository for a final assignment in probabilistic machine learning. Contains two independent exercises: diffusion / VAE work (exercise_1) and Gaussian process experiments (exercise_2), with code, pretrained models, results and utility scripts.

## Project structure
- exercise_1/  
    - main_*.py — entry points for training and evaluation (VAE and diffusion).  
    - mnist_ddpm_solution.ipynb — notebook exploring DDPM on MNIST.  
    - utils/ — diffusion, VAE, training and evaluation utilities.  
    - models/ — pretrained weights (baseline_ddpm_ema.pt, flow_matching.pt, mnist_feature_net.pt).
- exercise_2/  
    - gp_*.py, MVN.py, sampling_pyro.py — GP experiments, prior sampling and Pyro examples.  
    - data_part_B.csv — dataset for GP tasks.  
    - results_*/ — saved experiment outputs: grids, plots and summary tables.

## Key scripts
- exercise_1/main_vae.py — train/evaluate VAE.
- exercise_1/main_diffusion.py — train/evaluate diffusion/score-based models.
- exercise_1/main_evaluate_*.py — various evaluation utilities using pretrained models.
- exercise_2/gp_p2_q*.py, gp_b1_baseline.py — GP experiment drivers.
- exercise_2/sampling_pyro.py — Pyro-based sampling examples.

## Quick start
1. Create environment and install dependencies (example):
     pip install torch torchvision numpy scipy matplotlib pandas seaborn arviz pyro-ppl
2. Run a script:
     - Train/evaluate diffusion: python exercise_1/main_diffusion.py
     - Train/evaluate VAE: python exercise_1/main_vae.py
     - Run GP experiment: python exercise_2/gp_p2_q1.py

Adjust command-line flags in each main_*.py as needed (see header / argparsing in scripts).

## Results and reproducibility
- Precomputed models are in exercise_1/models/.
- experiment outputs and summaries are under exercise_2/results_*/ and exercise_1 (notebook outputs).
- Use the provided utils/ and results/ grids for reproducing published figures and tables.

## Notes
- Scripts assume a typical ML Python stack (PyTorch, NumPy, SciPy). Some GP experiments use Pyro and ArviZ.
- Inspect individual main_*.py and utils/*.py for hyperparameters and exact command-line options.
