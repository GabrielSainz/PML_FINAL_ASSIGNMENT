# gp_b2_1_joint_prior.py
# B.2.1 — Joint GP prior over (f, f') on a grid, sampling + finite-difference verification.
#
# Inputs:
#   data_part_B.csv with columns: x, y, Delta (no header)
# Outputs (PDF):
#   plots/prior_f_<KERNEL>.pdf
#   plots/prior_fp_<KERNEL>.pdf
#   plots/derivative_check_<KERNEL>.pdf
#   plots/derivative_scatter_<KERNEL>.pdf
# Tables:
#   tables/b2_1_derivative_check_summary.csv
#
# Usage:
#   python exercise_2/gp_p2_q1.py --data exercise_2/data_part_B.csv --outdir exercise_2/results_b2_1 --kernel all --ell 0.7 --sf2 1.0

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Kernel blocks: K, K1, K2  (1D)
# Convention:
#   K_ij   = k(x_i, x_j)
#   K1_ij  = ∂/∂x_i k(x_i, x_j)     (derivative wrt FIRST arg)
#   K2_ij  = ∂^2/(∂x_i ∂x_j) k(x_i, x_j)  (mixed derivative)
# Joint prior:
#   [f; f'] ~ N(0, [[K, K1^T],[K1,K2]])
# =========================
def _pairwise_diff(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X).reshape(-1)
    return X[:, None] - X[None, :]


def rbf_blocks(X: np.ndarray, ell: float, sf2: float):
    d = _pairwise_diff(X)
    K = sf2 * np.exp(-0.5 * (d**2) / (ell**2))
    K1 = -(d / (ell**2)) * K
    K2 = (1.0 / (ell**2) - (d**2) / (ell**4)) * K
    return K, K1, K2


def matern52_blocks(X: np.ndarray, ell: float, sf2: float):
    d = _pairwise_diff(X)
    u = np.abs(d)
    a = np.sqrt(5.0)
    t = u / ell

    K = sf2 * (1.0 + a * t + (5.0 / 3.0) * t**2) * np.exp(-a * t)
    K1 = -(5.0 * sf2 / (3.0 * ell**2)) * np.exp(-a * t) * d * (1.0 + a * t)
    K2 = (5.0 * sf2 / (3.0 * ell**2)) * np.exp(-a * t) * np.exp(-a * t * 0.0)  # no-op, keeps style
    K2 = (5.0 * sf2 / (3.0 * ell**2)) * np.exp(-a * t) * (1.0 + a * t - 5.0 * t**2)
    return K, K1, K2


def joint_covariance(K: np.ndarray, K1: np.ndarray, K2: np.ndarray, jitter: float = 1e-8):
    n = K.shape[0]
    Sigma = np.block([[K, K1.T],
                      [K1, K2]])
    # Symmetrize (tiny numerical asymmetries can break Cholesky)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + jitter * np.eye(2 * n)
    return Sigma


def sample_joint_prior(Sigma: np.ndarray, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(Sigma)
    z = rng.standard_normal(size=(Sigma.shape[0], n_samples))
    return (L @ z).T  # [n_samples, 2n]


def finite_diff_central(f: np.ndarray, x: np.ndarray):
    """
    Central differences on a grid:
      fp_fd[i] = (f[i+1] - f[i-1]) / (x[i+1] - x[i-1]) for i=1..n-2
    """
    fp_fd = np.full_like(f, np.nan, dtype=float)
    fp_fd[1:-1] = (f[2:] - f[:-2]) / (x[2:] - x[:-2])
    return fp_fd


def rmse_and_corr(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return np.nan, np.nan
    rmse = float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))
    corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
    return rmse, corr


# =========================
# Plotting helpers
# =========================
def rugplot_x(ax, x: np.ndarray, ymin: float, height: float = 0.04, alpha: float = 0.35):
    """
    Small rug marks to show where training inputs are (for context).
    """
    x = np.asarray(x).reshape(-1)
    for xi in x:
        ax.plot([xi, xi], [ymin, ymin + height], alpha=alpha, linewidth=1.0)


def plot_prior_samples(
    *,
    Xg: np.ndarray,
    samples: np.ndarray,
    n: int,
    title: str,
    ylabel: str,
    x_train: np.ndarray,
    outpath: Path,
):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    for s in range(samples.shape[0]):
        ax.plot(Xg, samples[s, :n], linewidth=1.6, alpha=0.9)

    ymin, ymax = ax.get_ylim()
    rugplot_x(ax, x_train, ymin=ymin, height=(ymax - ymin) * 0.04)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_derivative_check(
    *,
    Xg: np.ndarray,
    f: np.ndarray,
    fp: np.ndarray,
    fp_fd: np.ndarray,
    title: str,
    outpath: Path,
    rmse: float,
    corr: float,
):
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    ax.plot(Xg, fp, linewidth=2.0, label="Sampled f'(x) from joint prior")
    ax.plot(Xg, fp_fd, linewidth=2.0, linestyle="--", label="Central diff of sampled f(x)")
    ax.set_title(f"{title} | RMSE={rmse:.3g}, corr={corr:.3g}")
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_derivative_scatter(
    *,
    fp: np.ndarray,
    fp_fd: np.ndarray,
    title: str,
    outpath: Path,
):
    mask = np.isfinite(fp_fd)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(fp_fd[mask], fp[mask], s=14, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("finite-diff f'(x)")
    ax.set_ylabel("sampled f'(x)")
    ax.grid(True, alpha=0.3)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# =========================
# Runner per kernel
# =========================
def run_b21_for_kernel(
    *,
    kernel_name: str,
    blocks_fn,
    Xg: np.ndarray,
    x_train: np.ndarray,
    ell: float,
    sf2: float,
    n_samples: int,
    seed: int,
    jitter: float,
    plots_dir: Path,
):
    K, K1, K2 = blocks_fn(Xg, ell, sf2)
    Sigma = joint_covariance(K, K1, K2, jitter=jitter)
    draws = sample_joint_prior(Sigma, n_samples=n_samples, seed=seed)  # [S, 2n]
    n = Xg.size
    f_samps = draws[:, :n]
    fp_samps = draws[:, n:]

    # Prior sample plots (f and f')
    plot_prior_samples(
        Xg=Xg,
        samples=draws,
        n=n,
        title=f"B.2.1 Prior samples of f on grid ({kernel_name})\nell={ell}, sf2={sf2}",
        ylabel="f(x)",
        x_train=x_train,
        outpath=plots_dir / f"prior_f_{kernel_name}.pdf",
    )

    # For f' samples, we plot fp_samps explicitly
    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for s in range(n_samples):
        ax.plot(Xg, fp_samps[s], linewidth=1.6, alpha=0.9)
    ymin, ymax = ax.get_ylim()
    rugplot_x(ax, x_train, ymin=ymin, height=(ymax - ymin) * 0.04)
    ax.set_title(f"B.2.1 Prior samples of f' on grid ({kernel_name})\nell={ell}, sf2={sf2}")
    ax.set_xlabel("x")
    ax.set_ylabel("f'(x)")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / f"prior_fp_{kernel_name}.pdf", bbox_inches="tight")
    plt.close(fig)

    # Verification using one representative sample (sample 0)
    f0 = f_samps[0]
    fp0 = fp_samps[0]
    fp_fd0 = finite_diff_central(f0, Xg)
    rmse0, corr0 = rmse_and_corr(fp0, fp_fd0)

    plot_derivative_check(
        Xg=Xg,
        f=f0,
        fp=fp0,
        fp_fd=fp_fd0,
        title=f"B.2.1 Derivative check ({kernel_name})",
        outpath=plots_dir / f"derivative_check_{kernel_name}.pdf",
        rmse=rmse0,
        corr=corr0,
    )

    plot_derivative_scatter(
        fp=fp0,
        fp_fd=fp_fd0,
        title=f"B.2.1 Scatter: finite-diff vs sampled f' ({kernel_name})",
        outpath=plots_dir / f"derivative_scatter_{kernel_name}.pdf",
    )

    # Summary across all samples
    rmses, corrs = [], []
    for s in range(n_samples):
        fp_fd = finite_diff_central(f_samps[s], Xg)
        rmse_s, corr_s = rmse_and_corr(fp_samps[s], fp_fd)
        rmses.append(rmse_s)
        corrs.append(corr_s)

    return {
        "kernel": kernel_name,
        "ell": ell,
        "sigma_f2": sf2,
        "n_grid": int(Xg.size),
        "n_samples": int(n_samples),
        "jitter": float(jitter),
        "rmse_mean": float(np.nanmean(rmses)),
        "rmse_std": float(np.nanstd(rmses)),
        "corr_mean": float(np.nanmean(corrs)),
        "corr_std": float(np.nanstd(corrs)),
        "rmse_sample0": float(rmse0),
        "corr_sample0": float(corr0),
    }


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_part_B.csv")
    ap.add_argument("--outdir", type=str, default="results_b2_1")
    ap.add_argument("--kernel", type=str, default="all", choices=["all", "RBF", "Matern52"])

    ap.add_argument("--grid_n", type=int, default=100)
    ap.add_argument("--grid_min", type=float, default=-1.0)
    ap.add_argument("--grid_max", type=float, default=1.0)

    ap.add_argument("--ell", type=float, default=0.7)
    ap.add_argument("--sf2", type=float, default=1.0)

    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=1e-8)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    tables_dir = outdir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load data (we only use x for context/rug)
    df = pd.read_csv(args.data, header=None)
    df.columns = ["x", "y", "Delta"]
    x_train = df["x"].to_numpy(dtype=float)

    # Grid
    Xg = np.linspace(args.grid_min, args.grid_max, args.grid_n)

    kernels = []
    if args.kernel in ("all", "RBF"):
        kernels.append(("RBF", rbf_blocks))
    if args.kernel in ("all", "Matern52"):
        kernels.append(("Matern52", matern52_blocks))

    summaries = []
    for kname, blocks_fn in kernels:
        summ = run_b21_for_kernel(
            kernel_name=kname,
            blocks_fn=blocks_fn,
            Xg=Xg,
            x_train=x_train,
            ell=args.ell,
            sf2=args.sf2,
            n_samples=args.n_samples,
            seed=args.seed,
            jitter=args.jitter,
            plots_dir=plots_dir,
        )
        summaries.append(summ)
        print(
            f"[{kname}] RMSE(mean±std)={summ['rmse_mean']:.3g}±{summ['rmse_std']:.3g} | "
            f"corr(mean±std)={summ['corr_mean']:.3g}±{summ['corr_std']:.3g}"
        )

    # Save summary table
    summary_df = pd.DataFrame(summaries)
    summary_path = tables_dir / "b2_1_derivative_check_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved plots to:  {plots_dir}")
    print(f"Saved table to:  {summary_path}")


if __name__ == "__main__":
    main()
