import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to your summary (update if needed)
df = pd.read_csv("exercise_2/results_b2_3_9_10/tables/delta_posterior_summary.csv")

x = df["Delta_true"].to_numpy()
y = df["Delta_post_mean"].to_numpy()

# 95% interval error bars (asymmetric)
yerr_low  = y - df["Delta_post_q025"].to_numpy()
yerr_high = df["Delta_post_q975"].to_numpy() - y

plt.figure(figsize=(6.5, 5.5))
plt.errorbar(
    x, y,
    yerr=[yerr_low, yerr_high],
    fmt="o", capsize=3, elinewidth=1, markersize=5, alpha=0.85
)

# y = x reference line
lo = min(x.min(), y.min())
hi = max(x.max(), y.max())
pad = 0.05 * (hi - lo + 1e-12)
lims = (lo - pad, hi + pad)
plt.plot(lims, lims, "--", linewidth=1)

plt.xlim(lims)
plt.ylim(lims)
plt.xlabel(r"True input error $\Delta_i^{\mathrm{true}}$")
plt.ylabel(r"Posterior mean $\mathbb{E}[\Delta_i \mid X,y]$")
plt.title(r"Recovery of $\Delta$: posterior mean vs truth (95% CI)")

# Optional: label points by index i (can clutter; comment out if too busy)
for _, r in df.iterrows():
    plt.annotate(
        str(int(r["i"])),
        (r["Delta_true"], r["Delta_post_mean"]),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=7
    )

plt.tight_layout()
plt.savefig("exercise_2/results_b2_3_9_10/plots/delta_true_vs_postmean.pdf")
plt.close()
