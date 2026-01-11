# gp_b2_noisy_inputs.py
# python exercise_2/gp_p2_q2.py --data exercise_2/data_part_B.csv --outdir exercise_2/results_b2_gp --ngrid3 80
# Part B.2 (Q2): GP regression with noisy inputs via Taylor approximation
# Fits: y = f(X) - D f'(X) + eps, with known Delta from CSV
# Uses joint GP prior over (f, f') via kernel derivatives
#
# Outputs:
# - PDF plots of posterior mean + 95% CI on grid [-1,1] with true f(x) and data scatter
# - grid dumps (npz) from brute search
# - summary_table_b2.csv
# - (optional) combined_summary_with_b1.csv if you pass --b1_summary path

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

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
# Helpers
# =========================
def _pairwise_diff(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    return x1 - x2.T  # [n1, n2]


def safe_exp(u: float) -> float:
    return float(np.exp(np.clip(u, -30.0, 30.0)))


# =========================
# RBF kernel: k, dk/dx1, blocks (K,K1,K2)
# =========================
def rbf_k(x1: np.ndarray, x2: np.ndarray, ell: float, sf2: float) -> np.ndarray:
    d = _pairwise_diff(x1, x2)
    return sf2 * np.exp(-0.5 * (d**2) / (ell**2))


def rbf_dk_dx1(x1: np.ndarray, x2: np.ndarray, ell: float, sf2: float) -> np.ndarray:
    # derivative w.r.t. first argument x1
    d = _pairwise_diff(x1, x2)
    K = sf2 * np.exp(-0.5 * (d**2) / (ell**2))
    return -(d / (ell**2)) * K


def rbf_blocks(X: np.ndarray, ell: float, sf2: float):
    X = np.asarray(X).reshape(-1)
    d = _pairwise_diff(X, X)
    K = sf2 * np.exp(-0.5 * (d**2) / (ell**2))
    K1 = -(d / (ell**2)) * K
    K2 = (1.0 / (ell**2) - (d**2) / (ell**4)) * K
    return K, K1, K2


# =========================
# Matérn 5/2 kernel: k, dk/dx1, blocks (K,K1,K2)
# =========================
def matern52_k(x1: np.ndarray, x2: np.ndarray, ell: float, sf2: float) -> np.ndarray:
    d = _pairwise_diff(x1, x2)
    u = np.abs(d)
    a = np.sqrt(5.0)
    t = u / ell
    return sf2 * (1.0 + a * t + (5.0 / 3.0) * t**2) * np.exp(-a * t)


def matern52_dk_dx1(x1: np.ndarray, x2: np.ndarray, ell: float, sf2: float) -> np.ndarray:
    # derivative w.r.t. first argument x1
    d = _pairwise_diff(x1, x2)
    u = np.abs(d)
    a = np.sqrt(5.0)
    t = u / ell
    return -(5.0 * sf2 / (3.0 * ell**2)) * np.exp(-a * t) * d * (1.0 + a * t)


def matern52_blocks(X: np.ndarray, ell: float, sf2: float):
    X = np.asarray(X).reshape(-1)
    d = _pairwise_diff(X, X)
    u = np.abs(d)
    a = np.sqrt(5.0)
    t = u / ell

    K = sf2 * (1.0 + a * t + (5.0 / 3.0) * t**2) * np.exp(-a * t)
    K1 = -(5.0 * sf2 / (3.0 * ell**2)) * np.exp(-a * t) * d * (1.0 + a * t)
    K2 = (5.0 * sf2 / (3.0 * ell**2)) * np.exp(-a * t) * (1.0 + a * t - 5.0 * t**2)
    return K, K1, K2


# =========================
# Noisy-input GP (Taylor) core
# =========================
def build_C_delta(K: np.ndarray, K1: np.ndarray, K2: np.ndarray, Delta: np.ndarray, sigma_y2: float, jitter: float = 1e-8):
    """
    CΔ = K - K1^T D - D K1 + D K2 D + σy^2 I
    with D = diag(Delta).
    """
    Delta = np.asarray(Delta).reshape(-1)
    n = Delta.size

    term_K1T_D = K1.T * Delta[None, :]                 # cols scaled
    term_D_K1 = Delta[:, None] * K1                    # rows scaled
    term_DK2D = (Delta[:, None] * K2) * Delta[None, :] # both sides

    C = K - term_K1T_D - term_D_K1 + term_DK2D
    C = C + (sigma_y2 + jitter) * np.eye(n)
    return C


def logml_noisy_inputs(X: np.ndarray, y: np.ndarray, Delta: np.ndarray, blocks_fn, ell: float, sf2: float, sy2: float, jitter: float = 1e-8) -> float:
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)

    K, K1, K2 = blocks_fn(X, ell, sf2)
    C = build_C_delta(K, K1, K2, Delta, sy2, jitter=jitter)

    try:
        L = cholesky(C, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf

    z = solve_triangular(L, y, lower=True, check_finite=False)
    alpha = solve_triangular(L.T, z, lower=False, check_finite=False)

    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    n = y.size
    ll = -0.5 * (y @ alpha) - 0.5 * logdet - 0.5 * n * np.log(2.0 * np.pi)
    return float(ll)


def posterior_f_noisy_inputs(
    X_train: np.ndarray,
    y: np.ndarray,
    Delta: np.ndarray,
    X_test: np.ndarray,
    *,
    k_fn,
    dk_dx1_fn,
    blocks_fn,
    ell: float,
    sf2: float,
    sy2: float,
    jitter: float = 1e-8,
):
    """
    Posterior of latent f(X_test) given noisy-input model with known Delta at training points.
    Prediction at new points uses Delta_* = 0, i.e. we predict f(x*) (not f(x*) - Delta_* f'(x*)).

    Returns:
      mean_f [m], var_f_diag [m]
    """
    X_train = np.asarray(X_train).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Delta = np.asarray(Delta).reshape(-1)
    X_test = np.asarray(X_test).reshape(-1)

    K, K1, K2 = blocks_fn(X_train, ell, sf2)
    C = build_C_delta(K, K1, K2, Delta, sy2, jitter=jitter)
    L = cholesky(C, lower=True, check_finite=False)

    # alpha = C^{-1} y
    z = solve_triangular(L, y, lower=True, check_finite=False)
    alpha = solve_triangular(L.T, z, lower=False, check_finite=False)

    # Cov(f*, y) = K_{*X} - Cov(f*, f'_X) D
    Kxs = k_fn(X_test, X_train, ell, sf2)                  # [m,n]
    dK_train_test = dk_dx1_fn(X_train, X_test, ell, sf2)   # [n,m] = ∂/∂x_train k(x_train, x_test)
    Cov_fstar_fprime = dK_train_test.T                     # [m,n]
    Cov_fy = Kxs - (Cov_fstar_fprime * Delta[None, :])     # scale columns by Delta_i

    mean_f = Cov_fy @ alpha

    # var diag = diag(K** - Cov_fy C^{-1} Cov_yf)
    v = solve_triangular(L, Cov_fy.T, lower=True, check_finite=False)  # [n,m]
    Kss_diag = np.diag(k_fn(X_test, X_test, ell, sf2))
    var_f = np.maximum(0.0, Kss_diag - np.sum(v**2, axis=0))
    return mean_f, var_f


# =========================
# Brute search (3D): (log ell, log sf2, log sy2)
# =========================
@dataclass
class FitResult:
    kernel: str
    ell: float
    sigma_f2: float
    sigma_y2: float
    logml: float


def brute_fit_b2(
    X: np.ndarray,
    y: np.ndarray,
    Delta: np.ndarray,
    *,
    kernel_name: str,
    k_fn,
    dk_dx1_fn,
    blocks_fn,
    outdir_grids: Path,
    ngrid3: int,
    log_ell_range: tuple[float, float],
    log_sf2_range: tuple[float, float],
    log_sy2_range: tuple[float, float],
):
    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Delta = np.asarray(Delta).reshape(-1)

    outdir_grids.mkdir(parents=True, exist_ok=True)

    ranges = (
        slice(log_ell_range[0], log_ell_range[1], complex(ngrid3)),
        slice(log_sf2_range[0], log_sf2_range[1], complex(ngrid3)),
        slice(log_sy2_range[0], log_sy2_range[1], complex(ngrid3)),
    )

    def obj(u):
        log_ell, log_sf2, log_sy2 = u
        ell = safe_exp(log_ell)
        sf2 = safe_exp(log_sf2)
        sy2 = safe_exp(log_sy2)
        ll = logml_noisy_inputs(X, y, Delta, blocks_fn, ell, sf2, sy2)
        return -ll if np.isfinite(ll) else 1e30

    xopt, fval, grid, Jout = opt.brute(obj, ranges=ranges, full_output=True, finish=None, disp=False)

    log_ell, log_sf2, log_sy2 = xopt
    ell = safe_exp(log_ell)
    sf2 = safe_exp(log_sf2)
    sy2 = safe_exp(log_sy2)
    logml = -float(fval)

    # boundary warnings
    eps = 1e-12
    hit_ell = abs(log_ell - log_ell_range[0]) < 1e-6 or abs(log_ell - log_ell_range[1]) < 1e-6
    hit_sf2 = abs(log_sf2 - log_sf2_range[0]) < 1e-6 or abs(log_sf2 - log_sf2_range[1]) < 1e-6
    hit_sy2 = abs(log_sy2 - log_sy2_range[0]) < 1e-6 or abs(log_sy2 - log_sy2_range[1]) < 1e-6

    np.savez(
        outdir_grids / f"grid_b2_{kernel_name}.npz",
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
        kernel=kernel_name,
        boundary_hits=np.array([hit_ell, hit_sf2, hit_sy2], dtype=bool),
    )

    return FitResult(kernel=kernel_name, ell=ell, sigma_f2=sf2, sigma_y2=sy2, logml=logml), (hit_ell, hit_sf2, hit_sy2)


# =========================
# Plotting
# =========================
def plot_posterior_b2(
    *,
    X: np.ndarray,
    y: np.ndarray,
    Delta: np.ndarray,
    fit: FitResult,
    kernel_name: str,
    k_fn,
    dk_dx1_fn,
    blocks_fn,
    outdir_plots: Path,
    grid_n: int = 200,
):
    outdir_plots.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Delta = np.asarray(Delta).reshape(-1)

    Xg = np.linspace(-1.0, 1.0, grid_n)
    mean_f, var_f = posterior_f_noisy_inputs(
        X_train=X,
        y=y,
        Delta=Delta,
        X_test=Xg,
        k_fn=k_fn,
        dk_dx1_fn=dk_dx1_fn,
        blocks_fn=blocks_fn,
        ell=fit.ell,
        sf2=fit.sigma_f2,
        sy2=fit.sigma_y2,
    )

    sd = np.sqrt(var_f)
    lo = mean_f - 1.96 * sd
    hi = mean_f + 1.96 * sd

    fig = plt.figure(figsize=(8, 5))
    ax = plt.gca()

    ax.scatter(X, y, s=20, alpha=0.8, label="Data (x, y)")
    ax.plot(Xg, f_true(Xg), linewidth=2, label="True f(x)")
    ax.plot(Xg, mean_f, linewidth=2, label="Posterior mean (f)")
    ax.fill_between(Xg, lo, hi, alpha=0.25, label="95% CI (f)")

    ax.set_title(f"B.2 (noisy inputs) GP - {kernel_name} | ell={fit.ell:.3g}, sf2={fit.sigma_f2:.3g}, sy2={fit.sigma_y2:.3g}")
    ax.set_xlabel("x (observed)")
    ax.set_ylabel("y / f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    fig.savefig(outdir_plots / f"b2_posterior_{kernel_name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_overlay_b2(
    *,
    X: np.ndarray,
    y: np.ndarray,
    Delta: np.ndarray,
    fit_rbf: FitResult,
    fit_m52: FitResult,
    outdir_plots: Path,
    grid_n: int = 200,
):
    outdir_plots.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X).reshape(-1)
    y = np.asarray(y).reshape(-1)
    Delta = np.asarray(Delta).reshape(-1)
    Xg = np.linspace(-1.0, 1.0, grid_n)

    mean_rbf, var_rbf = posterior_f_noisy_inputs(
        X_train=X, y=y, Delta=Delta, X_test=Xg,
        k_fn=rbf_k, dk_dx1_fn=rbf_dk_dx1, blocks_fn=rbf_blocks,
        ell=fit_rbf.ell, sf2=fit_rbf.sigma_f2, sy2=fit_rbf.sigma_y2
    )
    sd_rbf = np.sqrt(var_rbf)
    lo_rbf = mean_rbf - 1.96 * sd_rbf
    hi_rbf = mean_rbf + 1.96 * sd_rbf

    mean_m52, var_m52 = posterior_f_noisy_inputs(
        X_train=X, y=y, Delta=Delta, X_test=Xg,
        k_fn=matern52_k, dk_dx1_fn=matern52_dk_dx1, blocks_fn=matern52_blocks,
        ell=fit_m52.ell, sf2=fit_m52.sigma_f2, sy2=fit_m52.sigma_y2
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

    ax.set_title("B.2 overlay comparison (RBF vs Matérn 5/2)")
    ax.set_xlabel("x (observed)")
    ax.set_ylabel("y / f(x)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)

    fig.savefig(outdir_plots / "b2_overlay_rbf_vs_matern52.pdf", bbox_inches="tight")
    plt.close(fig)


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_part_B.csv")
    ap.add_argument("--outdir", type=str, default="results_b2_gp")
    ap.add_argument("--ngrid3", type=int, default=25, help="grid points per dim (3D brute)")

    # optional: combine with B1 summary
    ap.add_argument("--b1_summary", type=str, default=None, help="path to B1 summary_table.csv (optional)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir_plots = outdir / "plots"
    outdir_grids = outdir / "grids"
    outdir.mkdir(parents=True, exist_ok=True)
    outdir_plots.mkdir(parents=True, exist_ok=True)
    outdir_grids.mkdir(parents=True, exist_ok=True)

    # Load data: columns are x, y, Delta (no header)
    df = pd.read_csv(args.data, header=None)
    df.columns = ["x", "y", "Delta"]
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    Delta = df["Delta"].to_numpy(dtype=float)

    # Grid ranges (log-space)
    # These are good starting ranges; if you hit boundaries, widen accordingly.
    log_ell_range = (math.log(0.01), math.log(3.0))
    log_sf2_range = (math.log(1e-4), math.log(30.0))
    log_sy2_range = (math.log(1e-6), math.log(0.2))  # allow a bit above 0.1 but not too huge

    kernels = {
        "RBF": dict(k_fn=rbf_k, dk_dx1_fn=rbf_dk_dx1, blocks_fn=rbf_blocks),
        "Matern52": dict(k_fn=matern52_k, dk_dx1_fn=matern52_dk_dx1, blocks_fn=matern52_blocks),
    }

    results: list[FitResult] = []
    boundary_notes = {}

    # Fit both kernels
    for kname, cfg in kernels.items():
        fit, hits = brute_fit_b2(
            X=x, y=y, Delta=Delta,
            kernel_name=kname,
            k_fn=cfg["k_fn"],
            dk_dx1_fn=cfg["dk_dx1_fn"],
            blocks_fn=cfg["blocks_fn"],
            outdir_grids=outdir_grids,
            ngrid3=args.ngrid3,
            log_ell_range=log_ell_range,
            log_sf2_range=log_sf2_range,
            log_sy2_range=log_sy2_range,
        )
        results.append(fit)
        boundary_notes[kname] = hits

        print(f"[B2 | {kname}] ell={fit.ell:.6g}, sf2={fit.sigma_f2:.6g}, sy2={fit.sigma_y2:.6g}, logML={fit.logml:.6g}")
        if any(hits):
            print(f"  WARNING: optimum near boundary (ell,sf2,sy2) hits={hits}. Consider widening ranges.")

        # Posterior plot on grid with true f and data scatter
        plot_posterior_b2(
            X=x, y=y, Delta=Delta,
            fit=fit,
            kernel_name=kname,
            k_fn=cfg["k_fn"],
            dk_dx1_fn=cfg["dk_dx1_fn"],
            blocks_fn=cfg["blocks_fn"],
            outdir_plots=outdir_plots,
            grid_n=200,
        )

    # Overlay comparison
    fit_rbf = next(r for r in results if r.kernel == "RBF")
    fit_m52 = next(r for r in results if r.kernel == "Matern52")
    plot_overlay_b2(
        X=x, y=y, Delta=Delta,
        fit_rbf=fit_rbf,
        fit_m52=fit_m52,
        outdir_plots=outdir_plots,
        grid_n=200,
    )

    # Summary table (B2)
    rows = []
    for r in results:
        rows.append({
            "kernel": r.kernel,
            "variant": "b2_noisy_inputs",
            "ell": r.ell,
            "sigma_f2": r.sigma_f2,
            "sigma_y2": r.sigma_y2,
            "log_marginal_likelihood": r.logml,
            "boundary_hit_ell": boundary_notes[r.kernel][0],
            "boundary_hit_sf2": boundary_notes[r.kernel][1],
            "boundary_hit_sy2": boundary_notes[r.kernel][2],
        })
    summary_b2 = pd.DataFrame(rows).sort_values(["kernel"])
    b2_path = outdir / "summary_table_b2.csv"
    summary_b2.to_csv(b2_path, index=False)
    print(f"\nSaved B2 summary table to: {b2_path}")

    # Optional: combine with B1 summary for comparison
    if args.b1_summary is not None:
        b1_path = Path(args.b1_summary)
        if b1_path.exists():
            summary_b1 = pd.read_csv(b1_path)
            combined = pd.concat([summary_b1, summary_b2], axis=0, ignore_index=True, sort=False)
            combined_path = outdir / "combined_summary_with_b1.csv"
            combined.to_csv(combined_path, index=False)
            print(f"Saved combined table (B1 + B2) to: {combined_path}")
        else:
            print(f"WARNING: --b1_summary path not found: {b1_path}")

    print(f"\nPlots saved under: {outdir_plots}")
    print(f"Grid dumps saved under: {outdir_grids}")


if __name__ == "__main__":
    main()
