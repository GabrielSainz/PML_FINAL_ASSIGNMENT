import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

def rbf_kernel(X1, X2, ell, sigma_f2):
    X1 = X1.reshape(-1,1)
    X2 = X2.reshape(-1,1)
    d2 = (X1 - X2.T)**2
    return sigma_f2 * np.exp(-0.5 * d2 / (ell**2))

def matern52_kernel(X1, X2, ell, sigma_f2):
    X1 = X1.reshape(-1,1)
    X2 = X2.reshape(-1,1)
    r = np.abs(X1 - X2.T)
    t = r / ell
    return sigma_f2 * (1 + np.sqrt(5)*t + 5*t**2/3) * np.exp(-np.sqrt(5)*t)

def gp_prior_samples(Xg, kernel_fn, ell, sigma_f2, n_samples=5, jitter=1e-8, seed=0):
    rng = np.random.default_rng(seed)
    K = kernel_fn(Xg, Xg, ell, sigma_f2) + jitter*np.eye(len(Xg))
    L = cholesky(K, lower=True)
    Z = rng.standard_normal(size=(len(Xg), n_samples))
    F = L @ Z  # mean 0
    return F  # shape [n_grid, n_samples]

def plot_prior_samples(Xg, F, title, outpath):
    plt.figure(figsize=(6.5, 3.8))
    for j in range(F.shape[1]):
        plt.plot(Xg, F[:, j], linewidth=1.8)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

# ---- Usage
Xg = np.linspace(-1.0, 1.0, 200)

# Choose hyperparameters (you can tweak)
ell = 0.4
sigma_f2 = 1.0
n_samples = 6

F_rbf = gp_prior_samples(Xg, rbf_kernel, ell, sigma_f2, n_samples=n_samples, seed=1)
F_m52 = gp_prior_samples(Xg, matern52_kernel, ell, sigma_f2, n_samples=n_samples, seed=1)

plot_prior_samples(Xg, F_rbf, f"Prior samples: RBF (ell={ell}, sigma_f2={sigma_f2})", "prior_samples_rbf.pdf")
plot_prior_samples(Xg, F_m52, f"Prior samples: Matérn 5/2 (ell={ell}, sigma_f2={sigma_f2})", "prior_samples_m52.pdf")

