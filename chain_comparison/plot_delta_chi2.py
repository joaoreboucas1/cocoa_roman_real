# Histograms of the chi2 differences in compare_like_samples.txt
# (run compare_like_samples.py first). Writes delta_chi2_hist.pdf next to this script.
# Usage (cocoa environment active): python plot_delta_chi2.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
d = np.loadtxt(OUT + "/compare_like_samples.txt")
stored, orig, cocoa = d[:, 4], d[:, 5], d[:, 6]
INK, ACCENT = "#3d3d3d", "#2a6fdb"
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for ax, x, title, xl in [
  (axes[0], cocoa - orig, "Cocoa port vs original code", r"$\chi^2_\mathrm{Cocoa} - \chi^2_\mathrm{original}$"),
  (axes[1], stored - orig, "emcee stored vs original recomputed", r"$\chi^2_\mathrm{stored} - \chi^2_\mathrm{recomputed}$")]:
  ax.hist(x, bins=20, color=ACCENT, edgecolor="white", linewidth=1)
  ax.axvline(0, color=INK, lw=1, ls="--")
  ax.set_title(title, color=INK, fontsize=11)
  ax.set_xlabel(xl, color=INK); ax.set_ylabel("samples", color=INK)
  ax.text(0.02, 0.97, "mean %+.3f\nstd %.3f\nmax |.| %.3f" % (x.mean(), x.std(), np.abs(x).max()),
          transform=ax.transAxes, va="top", fontsize=9, color=INK)
  for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
fig.suptitle(r"$\Delta\chi^2$ at %d random emcee samples (steps 150-300)" % len(d), color=INK)
fig.tight_layout()
fig.savefig(OUT + "/delta_chi2_hist.pdf")
