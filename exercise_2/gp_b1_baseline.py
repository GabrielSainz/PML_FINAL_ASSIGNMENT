# python exercise_2/gp_b1_baseline.py --data exercise_2/data_part_B.csv --outdir exercise_2/results_b1_gp --ngrid2 120 --ngrid3 100
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize as opt
from scipy.linalg import cholesky, solve_triangular


# =========================
# Ground-truth function (for plotting only)
# =========================
def f_true(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return -x**2 + 2.0 / (1.0 + np.exp(-10.0 * x))


# =========================
# Kernels
# =========================
def _pairwise_sqdist(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    return (x1 - x2.T) ** 2


def kernel_rbf(x1: np.ndarray, x2: np.ndarray, ell: float, sigma_f2: float) -> np.ndarray:
    d2 = _pairwise_sqdist(x1, x2)
    return sigma_f2 * np.exp(-0.5 * d2 / (ell**2))


def kernel_matern52(x1: np.ndarray, x2: np.ndarray, ell: float, sigma_f2: float) -> np.ndarray:
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    r = np.abs(x1 - x2.T) / ell
    sqrt5_r = np.sqrt(5.0) * r
    return sigma_f2 * (1.0 + sqrt5_r + (5.0 / 3.0) * r**2) * np.exp(-sqrt5_r)


# =========================
# GP utilities
# =========================
def gp_log_marginal_likelihood(
    X: np.ndarray,
    y: np.ndarray,
    kernel_fn,
    ell: float,
    sigma_f2: float,
    sigma_y2: float,
    jitter: float = 1e-8,
) -> float:
    """
    log p(y | X, ell, sigma_f2, sigma_y2) for GP regression with zero mean.
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    n = X.shape[0]

    K = kernel_fn(X, X, ell, sigma_f2)
    K = K + (sigma_y2 + jitter) * np.eye(n)

    try:
        L = cholesky(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf

    # alpha = K^{-1} y via Cholesky
    z = solve_triangular(L, y, lower=True, check_finite=False)
    alpha = solve_triangular(L.T, z, lower=False, check_finite=False)

    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    ll = -0.5 * (y @ alpha) - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)
    return float(ll)


def gp_posterior(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    kernel_fn,
    ell: float,
    sigma_f2: float,
    sigma_y2: float,
    jitter: float = 1e-8,
):
    """
    Posterior over latent f(X_test) and predictive y(X_test).

    Returns:
      mean_f: [m]
      var_f:  [m]  (diagonal)
      var_y:  [m]  (diagonal) = var_f + sigma_y2
    """
    X_train = np.asarray(X_train).reshape(-1)
    y_train = np.asarray(y_train).reshape(-1)
    X_test = np.asarray(X_test).reshape(-1)

    n = X_train.shape[0]
    K = kernel_fn(X_train, X_train, ell, sigma_f2) + (sigma_y2 + jitter) * np.eye(n)
    Ks = kernel_fn(X_test, X_train, ell, sigma_f2)        # [m,n]
    Kss = kernel_fn(X_test, X_test, ell, sigma_f2)        # [m,m]

    L = cholesky(K, lower=True, check_finite=False)

    # alpha = K^{-1} y
    z = solve_triangular(L, y_train, lower=True, check_finite=False)
    alpha = solve_triangular(L.T, z, lower=False, check_finite=False)

    mean_f = Ks @ alpha

    # var_f diag = diag(Kss - Ks K^{-1} Ks^T)
    # compute v = L^{-1} Ks^T then var = diag(Kss) - sum(v^2, axis=0)
    v = solve_triangular(L, Ks.T, lower=True, check_finite=False)  # [n,m]
    var_f = np.maximum(0.0, np.diag(Kss) - np.sum(v**2, axis=0))
    var_y = var_f + sigma_y2

    return mean_f, var_f, var_y


# =========================
# Grid search
# =========================
@dataclass
class FitResult:
    kernel_name: str
    variant: str
    ell: float
    sigma_f2: float
    sigma_y2: float
    logml: float


def brute_fit_variant(
    *,
    X: np.ndarray,
    y: np.ndarray,
    kernel_name: str,
    kernel_fn,
    variant: str,
    sigma_y2_fixed: float | None,
    outdir: Path,
    ngrid2: int,
    ngrid3: int,
    log_ell_range: tuple[float, float],
    log_sf2_range: tuple[float, float],
    log_sy2_range: tuple[float, float],
) -> FitResult:
    """
    variant:
      "a" => optimize (ell, sf2, sy2)
      "b" => optimize (ell, sf2), sigma_y2 fixed, X=noisy x
      "c" => optimize (ell, sf2), sigma_y2 fixed, X=true x
    """
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)

    outdir.mkdir(parents=True, exist_ok=True)

    def safe_exp(u: float) -> float:
        # avoid under/overflow in extreme grids
        return float(np.exp(np.clip(u, -30.0, 30.0)))

    if variant == "a":
        Ns = ngrid3
        ranges = (
            slice(log_ell_range[0], log_ell_range[1], complex(Ns)),
            slice(log_sf2_range[0], log_sf2_range[1], complex(Ns)),
            slice(log_sy2_range[0], log_sy2_range[1], complex(Ns)),
        )

        def obj(u):
            log_ell, log_sf2, log_sy2 = u
            ell = safe_exp(log_ell)
            sf2 = safe_exp(log_sf2)
            sy2 = safe_exp(log_sy2)
            ll = gp_log_marginal_likelihood(X, y, kernel_fn, ell, sf2, sy2)
            # brute minimizes
            return -ll if np.isfinite(ll) else 1e30

    else:
        assert sigma_y2_fixed is not None, "sigma_y2_fixed must be provided for variants b/c"
        Ns = ngrid2
        ranges = (
            slice(log_ell_range[0], log_ell_range[1], complex(Ns)),
            slice(log_sf2_range[0], log_sf2_range[1], complex(Ns)),
        )

        def obj(u):
            log_ell, log_sf2 = u
            ell = safe_exp(log_ell)
            sf2 = safe_exp(log_sf2)
            ll = gp_log_marginal_likelihood(X, y, kernel_fn, ell, sf2, sigma_y2_fixed)
            return -ll if np.isfinite(ll) else 1e30

    # Run brute force grid
    xopt, fval, grid, Jout = opt.brute(
        obj,
        ranges=ranges,
        full_output=True,
        finish=None,
        disp=False,
    )

    # Decode optimum
    if variant == "a":
        log_ell, log_sf2, log_sy2 = xopt
        ell = safe_exp(log_ell)
        sf2 = safe_exp(log_sf2)
        sy2 = safe_exp(log_sy2)
    else:
        log_ell, log_sf2 = xopt
        ell = safe_exp(log_ell)
        sf2 = safe_exp(log_sf2)
        sy2 = float(sigma_y2_fixed)

    logml = -float(fval)

    # Save intermediate grid results
    np.savez(
        outdir / f"grid_{kernel_name}_variant_{variant}.npz",
        xopt=np.array(xopt, dtype=float),
        fval=float(fval),
        logml=float(logml),
        grid=grid,
        Jout=Jout,
        ranges=np.array(
            [
                [log_ell_range[0], log_ell_range[1]],
                [log_sf2_range[0], log_sf2_range[1]],
                [log_sy2_range[0], log_sy2_range[1]],
            ],
            dtype=float,
        ),
        variant=variant,
        kernel=kernel_name,
    )

    return FitResult(
        kernel_name=kernel_name,
        variant=variant,
        ell=ell,
        sigma_f2=sf2,
        sigma_y2=sy2,
        logml=logml,
    )


# =========================
# Plotting
# =========================
def plot_fit(
    *,
    X: np.ndarray,
    y: np.ndarray,
    kernel_name: str,
    variant: str,
    fit: FitResult,
    kernel_fn,
    outdir: Path,
    grid_n: int = 100,
):
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)

    Xg = np.linspace(-1.0, 1.0, grid_n)

    mean_f, var_f, var_y = gp_posterior(
        X_train=X,
        y_train=y,
        X_test=Xg,
        kernel_fn=kernel_fn,
        ell=fit.ell,
        sigma_f2=fit.sigma_f2,
        sigma_y2=fit.sigma_y2,
    )

    sd_f = np.sqrt(var_f)
    lo_f = mean_f - 1.96 * sd_f
    hi_f = mean_f + 1.96 * sd_f

    # (optional) predictive interval for y
    sd_y = np.sqrt(var_y)
    lo_y = mean_f - 1.96 * sd_y
    hi_y = mean_f + 1.96 * sd_y

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    ax.scatter(X, y, s=20, alpha=0.8, label="Data (x, y)")

    # True function (for reference)
    ax.plot(Xg, f_true(Xg), linewidth=2, label="True f(x)")

    # Posterior mean and CI for latent f
    ax.plot(Xg, mean_f, linewidth=2, label="Posterior mean (f)")
    ax.fill_between(Xg, lo_f, hi_f, alpha=0.25, label="95% CI (f)")

    # If you want to explicitly show predictive band for y, uncomment:
    # ax.fill_between(Xg, lo_y, hi_y, alpha=0.15)

    ax.set_title(f"GP ({kernel_name}) - Variant {variant} | ell={fit.ell:.4g}, sf2={fit.sigma_f2:.4g}, sy2={fit.sigma_y2:.4g}")
    ax.set_xlabel("x")
    ax.set_ylabel("y / f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"posterior_{kernel_name}_variant_{variant}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_comparison_overlay(
    *,
    X: np.ndarray,
    y: np.ndarray,
    variant: str,
    fit_rbf: FitResult,
    fit_m52: FitResult,
    outdir: Path,
    grid_n: int = 100,
):
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Xg = np.linspace(-1.0, 1.0, grid_n)

    # RBF
    mean_rbf, var_rbf, _ = gp_posterior(
        X_train=X, y_train=y, X_test=Xg,
        kernel_fn=kernel_rbf,
        ell=fit_rbf.ell, sigma_f2=fit_rbf.sigma_f2, sigma_y2=fit_rbf.sigma_y2
    )
    sd_rbf = np.sqrt(var_rbf)
    lo_rbf = mean_rbf - 1.96 * sd_rbf
    hi_rbf = mean_rbf + 1.96 * sd_rbf

    # Matérn 5/2
    mean_m52, var_m52, _ = gp_posterior(
        X_train=X, y_train=y, X_test=Xg,
        kernel_fn=kernel_matern52,
        ell=fit_m52.ell, sigma_f2=fit_m52.sigma_f2, sigma_y2=fit_m52.sigma_y2
    )
    sd_m52 = np.sqrt(var_m52)
    lo_m52 = mean_m52 - 1.96 * sd_m52
    hi_m52 = mean_m52 + 1.96 * sd_m52

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    ax.scatter(X, y, s=20, alpha=0.8, label="Data (x, y)")
    ax.plot(Xg, f_true(Xg), linewidth=2, label="True f(x)")

    ax.plot(Xg, mean_rbf, linewidth=2, label="RBF mean")
    ax.fill_between(Xg, lo_rbf, hi_rbf, alpha=0.20, label="RBF 95% CI")

    ax.plot(Xg, mean_m52, linewidth=2, label="Matérn 5/2 mean")
    ax.fill_between(Xg, lo_m52, hi_m52, alpha=0.20, label="Matérn 5/2 95% CI")

    ax.set_title(f"Overlay comparison - Variant {variant} (RBF vs Matérn 5/2)")
    ax.set_xlabel("x")
    ax.set_ylabel("y / f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"overlay_variant_{variant}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_noisy_input_diagnostic(
    *,
    x_obs: np.ndarray,
    y: np.ndarray,
    delta: np.ndarray,
    sigma_x2: float,
    outdir: Path,
    grid_n: int = 400,
    mc_samples: int = 2000,
    seed: int = 0,
):
    x_obs = np.asarray(x_obs).reshape(-1)
    y = np.asarray(y).reshape(-1)
    delta = np.asarray(delta).reshape(-1)

    xg = np.linspace(-1.0, 1.0, grid_n)

    # The noise-free latent at the true input, but plotted vs observed x
    latent_at_truth = f_true(x_obs - delta)

    # Blurred mean curve: E[f(x - Δ)] via Monte Carlo (optional but very informative)
    rng = np.random.default_rng(seed)
    deltas = rng.normal(loc=0.0, scale=np.sqrt(sigma_x2), size=(mc_samples, xg.size))
    blurred_mean = f_true(xg[None, :] - deltas).mean(axis=0)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    ax.scatter(x_obs, y, s=20, alpha=0.8, label="Observed data (x, y)")
    ax.scatter(x_obs, latent_at_truth, s=25, marker="x", alpha=0.9, label="Latent: f(x - Δ_i) at points")

    ax.plot(xg, f_true(xg), linewidth=2, label="True f(x)")
    ax.plot(xg, blurred_mean, linewidth=2, linestyle="--", label="Blurred mean: E[f(x - Δ)]")

    ax.set_title("Input-noise diagnostic (why (a)/(b) mismatch f(x))")
    ax.set_xlabel("x (observed)")
    ax.set_ylabel("y / f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "diagnostic_input_noise.pdf", bbox_inches="tight")
    plt.close(fig)

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_part_B.csv")
    ap.add_argument("--outdir", type=str, default="results_b1_gp")
    ap.add_argument("--ngrid2", type=int, default=60, help="grid points per dim for 2D grids (variants b,c)")
    ap.add_argument("--ngrid3", type=int, default=25, help="grid points per dim for 3D grid (variant a)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(args.data, header=None)
    df.columns = ["x", "y", "Delta"]
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    delta = df["Delta"].to_numpy(dtype=float)
    x_truth = x - delta

    plot_noisy_input_diagnostic(
    x_obs=x,
    y=y,
    delta=delta,
    sigma_x2=0.01,
    outdir=outdir / "plots",)


    # Fixed noise variance for variants (b) and (c)
    sigma_y2_fixed = 0.0025

    # Recommended log-space grid ranges (tweak if needed)
    # x is in [-1,1], so length-scales in ~[0.05, 2] are sensible starts.
    log_ell_range = (math.log(0.01), math.log(3.0))     # allow smaller + larger length-scales
    # signal variance range (based on scale of y; you can widen if needed)
    log_sf2_range = (math.log(1e-4), math.log(30.0))    # slightly wider amplitude range
    # noise variance range for variant (a)
    log_sy2_range = (math.log(1e-6), math.log(1.0))     # crucial: allow sy2 > 0.1 for (a)

    kernels = {
        "RBF": kernel_rbf,
        "Matern52": kernel_matern52,
    }

    # Prepare variants
    variants = {
        "a": dict(X=x,        sigma_y2_fixed=None,           label="X=noisy x, sy2 unknown"),
        "b": dict(X=x,        sigma_y2_fixed=sigma_y2_fixed, label="X=noisy x, sy2 fixed"),
        "c": dict(X=x_truth,  sigma_y2_fixed=sigma_y2_fixed, label="X=true x,  sy2 fixed"),
    }

    all_results: list[FitResult] = []

    # Fit all (kernel, variant)
    for kernel_name, kernel_fn in kernels.items():
        for vname, vcfg in variants.items():
            Xv = vcfg["X"]
            sy2_fix = vcfg["sigma_y2_fixed"]

            fit = brute_fit_variant(
                X=Xv,
                y=y,
                kernel_name=kernel_name,
                kernel_fn=kernel_fn,
                variant=vname,
                sigma_y2_fixed=sy2_fix,
                outdir=outdir / "grids",
                ngrid2=args.ngrid2,
                ngrid3=args.ngrid3,
                log_ell_range=log_ell_range,
                log_sf2_range=log_sf2_range,
                log_sy2_range=log_sy2_range,
            )
            all_results.append(fit)

            print(f"[{kernel_name} | variant {vname}] ell={fit.ell:.6g}, sf2={fit.sigma_f2:.6g}, sy2={fit.sigma_y2:.6g}, logML={fit.logml:.6g}")

            # Save posterior plot
            plot_fit(
                X=Xv, y=y,
                kernel_name=kernel_name,
                variant=vname,
                fit=fit,
                kernel_fn=kernel_fn,
                outdir=outdir / "plots",
                grid_n=100,
            )

    # Build summary table
    rows = []
    for r in all_results:
        rows.append({
            "kernel": r.kernel_name,
            "variant": r.variant,
            "ell": r.ell,
            "sigma_f2": r.sigma_f2,
            "sigma_y2": r.sigma_y2,
            "log_marginal_likelihood": r.logml,
        })
    summary = pd.DataFrame(rows).sort_values(["variant", "kernel"])
    summary_path = outdir / "summary_table.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary table to: {summary_path}")

    # Overlay comparison plots (RBF vs Matern52) per variant
    for vname, vcfg in variants.items():
        Xv = vcfg["X"]
        fit_rbf = next(r for r in all_results if r.kernel_name == "RBF" and r.variant == vname)
        fit_m52 = next(r for r in all_results if r.kernel_name == "Matern52" and r.variant == vname)

        plot_comparison_overlay(
            X=Xv, y=y,
            variant=vname,
            fit_rbf=fit_rbf,
            fit_m52=fit_m52,
            outdir=outdir / "plots",
            grid_n=100,
        )

    print(f"\nAll plots saved under: {outdir / 'plots'}")
    print(f"All grid dumps saved under: {outdir / 'grids'}")


if __name__ == "__main__":
    main()
