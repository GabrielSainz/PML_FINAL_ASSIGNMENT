from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide.initialization import init_to_value

#python exercise_2/gp_p2_q3.py --data exercise_2/data_part_B.csv --summary exercise_2/results_b2_gp/summary_table_b2.csv --kernel best --outdir exercise_2/results_b2_3_9_10 --jitter 1e-4 --num_chains 4


# -------------------------
# Ground truth function (for plotting only)
# -------------------------
def f_true_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return -x**2 + 2.0 / (1.0 + np.exp(-10.0 * x))


# -------------------------
# Torch kernel utilities (1D)
# -------------------------
def pairwise_diff_t(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return x1[:, None] - x2[None, :]


# ---- RBF blocks: K, K1=∂/∂x_i k(x_i,x_j), K2=∂^2/(∂x_i∂x_j)k
def rbf_blocks_t(X: torch.Tensor, ell: torch.Tensor, sf2: torch.Tensor):
    d = pairwise_diff_t(X, X)
    K = sf2 * torch.exp(-0.5 * (d**2) / (ell**2))
    K1 = -(d / (ell**2)) * K
    K2 = (1.0 / (ell**2) - (d**2) / (ell**4)) * K
    return K, K1, K2


def rbf_k_t(x1: torch.Tensor, x2: torch.Tensor, ell: torch.Tensor, sf2: torch.Tensor):
    d = pairwise_diff_t(x1, x2)
    return sf2 * torch.exp(-0.5 * (d**2) / (ell**2))


def rbf_dk_dxtrain_t(x_train: torch.Tensor, x_test: torch.Tensor, ell: torch.Tensor, sf2: torch.Tensor):
    # ∂/∂x_train k(x_train, x_test)  -> shape [n_train, n_test]
    d = pairwise_diff_t(x_train, x_test)
    K = sf2 * torch.exp(-0.5 * (d**2) / (ell**2))
    return -(d / (ell**2)) * K


# ---- Matern 5/2 blocks
def matern52_blocks_t(X: torch.Tensor, ell: torch.Tensor, sf2: torch.Tensor):
    d = pairwise_diff_t(X, X)
    u = d.abs()
    a = math.sqrt(5.0)
    t = u / ell

    K = sf2 * (1.0 + a * t + (5.0 / 3.0) * t**2) * torch.exp(-a * t)
    K1 = -(5.0 * sf2 / (3.0 * ell**2)) * torch.exp(-a * t) * d * (1.0 + a * t)
    K2 = (5.0 * sf2 / (3.0 * ell**2)) * torch.exp(-a * t) * (1.0 + a * t - 5.0 * t**2)
    return K, K1, K2


def matern52_k_t(x1: torch.Tensor, x2: torch.Tensor, ell: torch.Tensor, sf2: torch.Tensor):
    d = pairwise_diff_t(x1, x2)
    u = d.abs()
    a = math.sqrt(5.0)
    t = u / ell
    return sf2 * (1.0 + a * t + (5.0 / 3.0) * t**2) * torch.exp(-a * t)


def matern52_dk_dxtrain_t(x_train: torch.Tensor, x_test: torch.Tensor, ell: torch.Tensor, sf2: torch.Tensor):
    # ∂/∂x_train k(x_train, x_test)  -> shape [n_train, n_test]
    d = pairwise_diff_t(x_train, x_test)
    u = d.abs()
    a = math.sqrt(5.0)
    t = u / ell
    return -(5.0 * sf2 / (3.0 * ell**2)) * torch.exp(-a * t) * d * (1.0 + a * t)


# -------------------------
# CΔ construction
# CΔ = K - K1^T D - D K1 + D K2 D + σy^2 I
# implemented without explicit diag matrices.
# -------------------------
def build_C_delta_t(K, K1, K2, Delta, sigma_y2, jitter=1e-6):
    Dcol = Delta[None, :]     # scale columns by Δ
    Drow = Delta[:, None]     # scale rows by Δ
    C = K - (K1.T * Dcol) - (Drow * K1) + ((Drow * K2) * Dcol)
    n = K.shape[0]
    C = 0.5 * (C + C.T)  # symmetrize for numerical safety
    C = C + (sigma_y2 + jitter) * torch.eye(n, dtype=K.dtype, device=K.device)
    return C


# -------------------------
# Conditional posterior of f* given Delta (Δ_* = 0)
# f*|y,Δ ~ N( μ*(Δ), Σ*(Δ) )
# using Cov(f*,y)=K_*X - Cov(f*,f'_X) D
# -------------------------
@torch.no_grad()
def posterior_f_given_delta(
    X_train: torch.Tensor,
    y: torch.Tensor,
    Delta: torch.Tensor,
    X_test: torch.Tensor,
    *,
    kernel: str,
    ell: torch.Tensor,
    sf2: torch.Tensor,
    sy2: torch.Tensor,
    jitter: float,
):
    if kernel == "RBF":
        blocks_fn = rbf_blocks_t
        k_fn = rbf_k_t
        dk_dxtrain_fn = rbf_dk_dxtrain_t
    elif kernel == "Matern52":
        blocks_fn = matern52_blocks_t
        k_fn = matern52_k_t
        dk_dxtrain_fn = matern52_dk_dxtrain_t
    else:
        raise ValueError("kernel must be RBF or Matern52")

    K, K1, K2 = blocks_fn(X_train, ell, sf2)
    C = build_C_delta_t(K, K1, K2, Delta, sy2, jitter=jitter)

    L = torch.linalg.cholesky(C)
    alpha = torch.cholesky_solve(y[:, None], L)[:, 0]  # C^{-1} y

    Kxs = k_fn(X_test, X_train, ell, sf2)  # [m,n]
    dK_train_test = dk_dxtrain_fn(X_train, X_test, ell, sf2)  # [n,m] = ∂/∂x_train k(x_train,x_test)
    Cov_fstar_fprime = dK_train_test.T  # [m,n] = Cov(f*, f'_X)

    Cov_fy = Kxs - (Cov_fstar_fprime * Delta[None, :])  # scale columns by Δ_i

    mean_f = Cov_fy @ alpha  # [m]

    # diag variance
    Cov_yf = Cov_fy.T  # [n,m]
    v = torch.linalg.solve_triangular(L, Cov_yf, upper=False)  # [n,m]
    Kss_diag = torch.diagonal(k_fn(X_test, X_test, ell, sf2), 0)
    var_f = torch.clamp(Kss_diag - (v**2).sum(dim=0), min=0.0)
    return mean_f, var_f


# -------------------------
# Read best params from summary_table_b2
# -------------------------
def load_best_params(summary_path: str, kernel_choice: str):
    df = pd.read_csv(summary_path)
    # expected columns: kernel, variant, ell, sigma_f2, sigma_y2, log_marginal_likelihood
    for col in ["ell", "sigma_f2", "sigma_y2", "log_marginal_likelihood"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ell", "sigma_f2", "sigma_y2", "log_marginal_likelihood"])

    if kernel_choice in ("RBF", "Matern52"):
        dfk = df[df["kernel"].astype(str) == kernel_choice].copy()
        if dfk.empty:
            raise ValueError(f"No rows found for kernel={kernel_choice} in {summary_path}")
        best = dfk.loc[dfk["log_marginal_likelihood"].idxmax()]
    elif kernel_choice == "best":
        best = df.loc[df["log_marginal_likelihood"].idxmax()]
    else:
        raise ValueError("kernel_choice must be 'RBF', 'Matern52', or 'best'")

    return {
        "kernel": str(best["kernel"]),
        "variant": str(best.get("variant", "")),
        "ell": float(best["ell"]),
        "sigma_f2": float(best["sigma_f2"]),
        "sigma_y2": float(best["sigma_y2"]),
        "log_marginal_likelihood": float(best["log_marginal_likelihood"]),
    }


# -------------------------
# Pyro model: Δ ~ N(0, σx^2 I), y|Δ ~ N(0, CΔ)
# -------------------------
class GPDeltaModel:
    def __init__(self, X, y, *, kernel, ell, sf2, sy2, sx2, jitter):
        self.X = X
        self.y = y
        self.kernel = kernel
        self.ell = ell
        self.sf2 = sf2
        self.sy2 = sy2
        self.sx2 = sx2
        self.jitter = jitter

        self.ell_t = torch.tensor(ell, dtype=X.dtype, device=X.device)
        self.sf2_t = torch.tensor(sf2, dtype=X.dtype, device=X.device)
        self.sy2_t = torch.tensor(sy2, dtype=X.dtype, device=X.device)

        if kernel == "RBF":
            self.blocks_fn = rbf_blocks_t
        elif kernel == "Matern52":
            self.blocks_fn = matern52_blocks_t
        else:
            raise ValueError("kernel must be RBF or Matern52")

    def __call__(self):
        Delta = pyro.sample(
            "Delta",
            dist.Normal(0.0, math.sqrt(self.sx2))
                .expand([self.X.shape[0]])
                .to_event(1)
        )
        K, K1, K2 = self.blocks_fn(self.X, self.ell_t, self.sf2_t)
        C = build_C_delta_t(K, K1, K2, Delta, self.sy2_t, jitter=self.jitter)
        pyro.sample(
            "y",
            dist.MultivariateNormal(loc=torch.zeros_like(self.y), covariance_matrix=C),
            obs=self.y
        )



# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_part_B.csv")
    ap.add_argument("--summary", type=str, default="summary_table_b2.csv")
    ap.add_argument("--kernel", type=str, default="best", choices=["best", "RBF", "Matern52"])

    ap.add_argument("--sigma_x2", type=float, default=0.01)
    ap.add_argument("--jitter", type=float, default=1e-6)

    ap.add_argument("--outdir", type=str, default="results_b2_3")
    ap.add_argument("--grid_n", type=int, default=200)

    # NUTS
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--num_chains", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)

    # marginal f computation
    ap.add_argument("--thin_for_f", type=int, default=5, help="use every k-th Delta draw for marginal f to reduce cost")

    args = ap.parse_args()

    pyro.set_rng_seed(args.seed)
    torch.manual_seed(args.seed)

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    tables_dir = outdir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load best B2.2 params
    best = load_best_params(args.summary, args.kernel)
    kernel = best["kernel"]
    ell = best["ell"]
    sf2 = best["sigma_f2"]
    sy2 = best["sigma_y2"]

    # Save chosen params
    pd.DataFrame([best]).to_csv(tables_dir / "chosen_b2_2_params.csv", index=False)
    print("Using B2.2 params:", best)

    # Load data
    df = pd.read_csv(args.data, header=None)
    df.columns = ["x", "y", "Delta_true"]
    x = df["x"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    delta_true = df["Delta_true"].to_numpy(dtype=np.float64)

    X_t = torch.tensor(x, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)

    init = init_to_value(values={"Delta": torch.zeros_like(X_t)})

    # NUTS
    model = GPDeltaModel(
    X_t, y_t,
    kernel=kernel,
    ell=ell,
    sf2=sf2,
    sy2=sy2,
    sx2=args.sigma_x2,
    jitter=args.jitter,
    )
    nuts = NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        full_mass=True,          # important
        target_accept_prob=0.9,  # try 0.9 then 0.95 if needed
        max_tree_depth=12,        # try 12, then 15 if needed
        init_strategy=init
    )
    mcmc = MCMC(nuts, num_samples=args.num_samples, warmup_steps=args.warmup, num_chains=args.num_chains)
    mcmc.run()
    print(mcmc.diagnostics())

    #extra = mcmc.get_extra_fields()
    #if "divergences" in extra:
    #    div = extra["divergences"].detach().cpu().numpy()
    #    print("Total divergences:", div.sum())

    samples = mcmc.get_samples(group_by_chain=True)
    Delta_samps = samples["Delta"].detach().cpu().numpy()  # [C,S,N]
    Delta_flat = Delta_samps.reshape(-1, Delta_samps.shape[-1])  # [C*S, N]

    # Basic posterior stats for Δ
    delta_mean = Delta_flat.mean(axis=0)
    delta_sd = Delta_flat.std(axis=0, ddof=0)
    delta_q025 = np.quantile(Delta_flat, 0.025, axis=0)
    delta_q975 = np.quantile(Delta_flat, 0.975, axis=0)

    delta_table = pd.DataFrame({
        "i": np.arange(1, Delta_flat.shape[1] + 1),
        "Delta_post_mean": delta_mean,
        "Delta_post_sd": delta_sd,
        "Delta_post_q025": delta_q025,
        "Delta_post_q975": delta_q975,
        "Delta_true": delta_true,
    })
    delta_table.to_csv(tables_dir / "delta_posterior_summary.csv", index=False)

    # ArviZ convergence checks (optional, but requested)
    try:
        import arviz as az
        idata = az.from_dict(posterior={"Delta": Delta_samps})
        azsum = az.summary(idata, var_names=["Delta"])
        azsum.to_csv(tables_dir / "arviz_summary_delta.csv")
        # Trace plots for Δ9 and Δ10 if ArviZ exists
        az.plot_trace(idata, var_names=["Delta"], coords={"Delta_dim_0": [9, 10]})
        plt.tight_layout()
        plt.savefig(plots_dir / "arviz_trace_delta9_delta10.pdf", bbox_inches="tight")
        plt.close()
    except Exception as e:
        (tables_dir / "arviz_not_available.txt").write_text(f"ArviZ failed or not installed: {repr(e)}\n")

    # Exercise-required scatter of (Δ9, Δ10) (1-indexed -> indices 8 and 9)
    idx9, idx10 = 9, 10
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(Delta_flat[:, idx9], Delta_flat[:, idx10], s=10, alpha=0.5, label="posterior samples")
    ax.scatter([delta_true[idx9]], [delta_true[idx10]], s=90, marker="x", label="true (Δ9, Δ10)")
    # The prompt mentions (-0.25, 0.25); add it as reference too:
    ax.scatter([-0.25], [0.25], s=90, marker="+", label="prompt ref (-0.25, 0.25)")
    ax.set_title(f"Posterior samples (Δ9, Δ10) | {kernel}")
    ax.set_xlabel("Δ9")
    ax.set_ylabel("Δ10")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(plots_dir / "scatter_delta9_delta10.pdf", bbox_inches="tight")
    plt.close(fig)

    # ----- Marginal posterior of f on a grid by integrating over Δ
    Xg = np.linspace(-1.0, 1.0, args.grid_n)
    Xg_t = torch.tensor(Xg, dtype=torch.float64)

    ell_t = torch.tensor(ell, dtype=torch.float64)
    sf2_t = torch.tensor(sf2, dtype=torch.float64)
    sy2_t = torch.tensor(sy2, dtype=torch.float64)

    # Thin Δ samples for speed
    thin = max(1, args.thin_for_f)
    Delta_use = Delta_flat[::thin, :]
    S = Delta_use.shape[0]

    means = []
    vars_ = []
    for s in range(S):
        Delta_s = torch.tensor(Delta_use[s], dtype=torch.float64)
        m_s, v_s = posterior_f_given_delta(
            X_train=X_t,
            y=y_t,
            Delta=Delta_s,
            X_test=Xg_t,
            kernel=kernel,
            ell=ell_t,
            sf2=sf2_t,
            sy2=sy2_t,
            jitter=args.jitter,
        )
        means.append(m_s.cpu().numpy())
        vars_.append(v_s.cpu().numpy())

    means = np.stack(means, axis=0)  # [S, grid_n]
    vars_ = np.stack(vars_, axis=0)  # [S, grid_n]

    # Law of total variance
    mean_marg = means.mean(axis=0)
    var_marg = vars_.mean(axis=0) + means.var(axis=0, ddof=0)

    lo = mean_marg - 1.96 * np.sqrt(var_marg)
    hi = mean_marg + 1.96 * np.sqrt(var_marg)

    # Plot marginal posterior of f
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.scatter(x, y, s=25, alpha=0.8, label="data (x,y)")
    ax.plot(Xg, f_true_np(Xg), linewidth=2, label="true f(x)")
    ax.plot(Xg, mean_marg, linewidth=2, label="marginal posterior mean of f")
    ax.fill_between(Xg, lo, hi, alpha=0.25, label="95% marginal CI (f)")
    ax.set_title(f"Marginal posterior of f (Δ integrated) | {kernel}")
    ax.set_xlabel("x")
    ax.set_ylabel("y / f(x)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(plots_dir / "marginal_posterior_f.pdf", bbox_inches="tight")
    plt.close(fig)

    # Save grid table
    f_table = pd.DataFrame({
        "x_grid": Xg,
        "f_post_mean": mean_marg,
        "f_post_var": var_marg,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "f_true": f_true_np(Xg),
    })
    f_table.to_csv(tables_dir / "marginal_posterior_f_grid.csv", index=False)

    print("\nDone.")
    print(f"Kernel used: {kernel}")
    print(f"Saved plots to:  {plots_dir}")
    print(f"Saved tables to: {tables_dir}")
    print(f"\nΔ9 mean={delta_mean[idx9]:.4f} (true {delta_true[idx9]:.4f}), Δ10 mean={delta_mean[idx10]:.4f} (true {delta_true[idx10]:.4f})")


if __name__ == "__main__":
    main()
