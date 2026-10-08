# Trace and triangle plots of the emcee chain below (ns_fix parameter order).
# Usage (cocoa environment active): python plot_chain.py
# The figures are written next to this script.
import os
import h5py
import numpy as np
import emcee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from getdist import MCSamples, plots

OUT = os.path.dirname(os.path.abspath(__file__))
FN = os.path.join(OUT, "MCMC_DESY1CL_fidchain_AB_Analytic_fid_priors_theta_prior_HOD_evol_ns_fix_rich10_v5.hdf5")
BURN = 150 # steps discarded as burn-in (of 300)

# parameter order of p0 in emcee_cosmo_emu_bin_rich_lens_3bin_HOD_evol_ns_fix_v2.py
names = ["siglogM_1", "logMmin_1", "logM20_1", "alpha_1",
         "siglogM_3", "logMmin_3", "logM20_3", "alpha_3",
         "B", "fmis", "tau", "Am1", "Am2", "Am3", "sigma8", "omegam", "omegab"]
labels = [r"\sigma_{\log M}^{(1)}", r"\log M_\mathrm{min}^{(1)}", r"\log M_{20}^{(1)}", r"\alpha^{(1)}",
          r"\sigma_{\log M}^{(3)}", r"\log M_\mathrm{min}^{(3)}", r"\log M_{20}^{(3)}", r"\alpha^{(3)}",
          r"B", r"f_\mathrm{mis}", r"\tau", r"A_{m,1}", r"A_{m,2}", r"A_{m,3}",
          r"\sigma_8", r"\Omega_\mathrm{m}", r"\Omega_\mathrm{b}"]

r = emcee.backends.HDFBackend(FN, read_only=True)
chain = r.get_chain()            # (nstep, nwalker, ndim)
lp = r.get_log_prob()
nstep, nwalk, ndim = chain.shape
tau_ac = emcee.autocorr.integrated_time(chain, quiet=True)
print("steps %d, walkers %d, ndim %d, acceptance %.3f" % (nstep, nwalk, ndim, r.accepted.mean()/nstep))
for n, t in zip(names, tau_ac):
  print("  tau_int %-10s %6.1f  (nstep/tau = %.1f)" % (n, t, nstep/t))

burn = BURN
flat = chain[burn:].reshape(-1, ndim)
S8 = flat[:, 14]*np.sqrt(flat[:, 15]/0.3)
samples = MCSamples(samples=np.column_stack([flat, S8]),
                    names=names + ["S8"], labels=labels + [r"S_8"],
                    ranges={"fmis": [0, 1], "tau": [0, 1]},
                    settings={"smooth_scale_2D": 0.3, "smooth_scale_1D": 0.3})
print(samples.getTable(limit=1).tableTex())

# ---- marginalized constraints on the main cosmological parameters
stats = samples.getMargeStats()
lines = ["Marginalized constraints (burn-in %d of %d steps, %d walkers; S8 = sigma8 sqrt(omegam/0.3))"
         % (burn, nstep, nwalk),
         "  %-8s %18s %22s %22s" % ("", "mean +- std", "68% interval", "95% interval")]
for p in ["omegam", "sigma8", "S8"]:
  s = stats.parWithName(p)
  lines.append("  %-8s %8.4f +- %6.4f    [%.4f, %.4f]    [%.4f, %.4f]" % (p, s.mean, s.err,
               s.limits[0].lower, s.limits[0].upper, s.limits[1].lower, s.limits[1].upper))
print("\n" + "\n".join(lines))
with open(os.path.join(OUT, "marginalized_constraints.txt"), "w") as f:
  f.write("\n".join(lines) + "\n")

ACCENT = "#2a6fdb"
INK = "#3d3d3d"

# ---- trace plots: logp + cosmology
fig, axes = plt.subplots(5, 1, figsize=(9, 9), sharex=True)
rng = np.random.default_rng(1)
sub = rng.choice(nwalk, 40, replace=False)
for ax, (y, lab) in zip(axes, [(lp, r"$\log P$"), (chain[:, :, 14], r"$\sigma_8$"),
                               (chain[:, :, 15], r"$\Omega_\mathrm{m}$"),
                               (chain[:, :, 16], r"$\Omega_\mathrm{b}$"),
                               (chain[:, :, 8], r"$B$")]):
  ax.plot(y[:, sub], color="0.75", lw=0.5, alpha=0.6)
  ax.plot(np.median(y, axis=1), color=ACCENT, lw=2, label="median over walkers")
  ax.axvline(burn, color=INK, lw=1, ls="--")
  ax.set_ylabel(lab, color=INK)
  for s in ["top", "right"]: ax.spines[s].set_visible(False)
axes[0].set_ylim(np.percentile(lp, 1), lp.max() + 2)
axes[0].legend(loc="lower right", frameon=False, fontsize=9)
axes[0].text(burn, axes[0].get_ylim()[1], " burn-in cut", va="top", fontsize=9, color=INK)
axes[-1].set_xlabel("step", color=INK)
fig.suptitle("Trace: 40 random walkers (gray) of 1000", color=INK)
fig.tight_layout()
fig.savefig(OUT + "/trace.png", dpi=130)

# ---- triangle: cosmology
g = plots.get_subplot_plotter(width_inch=7)
g.settings.axes_fontsize = 10
g.triangle_plot(samples, ["omegam", "sigma8", "S8", "omegab"], filled=True,
                contour_colors=[ACCENT])
g.export(OUT + "/triangle_cosmo.png", dpi=130)

# ---- triangle: all nuisance parameters
g = plots.get_subplot_plotter(width_inch=14)
g.settings.axes_fontsize = 8
g.triangle_plot(samples, names[:14], filled=True, contour_colors=[ACCENT])
g.export(OUT + "/triangle_nuisance.png", dpi=90)
