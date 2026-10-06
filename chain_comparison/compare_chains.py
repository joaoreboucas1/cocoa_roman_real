# Compares the original emcee chain (original_code/, ns_fix parameter order)
# with the Cocoa port chain (chains/EXAMPLE_MCMC_CLUSTER_LENSING1).
# Usage (cocoa environment active): python compare_chains.py
# Figures and the constraints table are written next to this script.
import os
import numpy as np
import emcee
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, loadMCSamples, plots

OUT = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(OUT) # projects/roman_real
ORIG_BURN = 150     # same as original_code/plot_chain.py
COCOA_BURN = 0.3    # fraction of each Cobaya chain dropped

# ---- original emcee chain (ns_fix parameter order)
orig_names = ["SIGLOGM_1", "LOGMMIN_1", "LOGM20_1", "ALPHA_1",
              "SIGLOGM_3", "LOGMMIN_3", "LOGM20_3", "ALPHA_3",
              "BARYON_B", "FMIS", "TAU", "AM1", "AM2", "AM3", "sigma8", "omegam", "omegab"]
ch = emcee.backends.HDFBackend(P + "/original_code/MCMC_DESY1CL_fidchain_AB_Analytic_fid_priors_theta_prior_HOD_evol_ns_fix_rich10_v5.hdf5",
                               read_only=True).get_chain()
flat = ch[ORIG_BURN:].reshape(-1, ch.shape[2])
cols = {n: flat[:, i] for i, n in enumerate(orig_names)}
cols["S8"] = cols["sigma8"]*np.sqrt(cols["omegam"]/0.3)

labels = {"omegam": r"\Omega_\mathrm{m}", "sigma8": r"\sigma_8", "S8": r"S_8", "omegab": r"\Omega_\mathrm{b}",
          "SIGLOGM_1": r"\sigma_{\log M}^{(1)}", "LOGMMIN_1": r"\log M_\mathrm{min}^{(1)}", "LOGM20_1": r"\log M_{20}^{(1)}",
          "ALPHA_1": r"\alpha^{(1)}", "SIGLOGM_3": r"\sigma_{\log M}^{(3)}", "LOGMMIN_3": r"\log M_\mathrm{min}^{(3)}",
          "LOGM20_3": r"\log M_{20}^{(3)}", "ALPHA_3": r"\alpha^{(3)}", "BARYON_B": r"B", "FMIS": r"f_\mathrm{mis}",
          "TAU": r"\tau", "AM1": r"A_{m,1}", "AM2": r"A_{m,2}", "AM3": r"A_{m,3}"}
names = list(labels)
settings = {"smooth_scale_2D": 0.3, "smooth_scale_1D": 0.3}
orig = MCSamples(samples=np.column_stack([cols[n] for n in names]), names=names,
                 labels=[labels[n] for n in names], settings=settings,
                 label="original (emcee, steps %d-%d)" % (ORIG_BURN, ch.shape[0]))

# ---- Cocoa port chain (Cobaya), renamed to the same parameter names
c = loadMCSamples(P + "/chains/EXAMPLE_MCMC_CLUSTER_LENSING1", settings={"ignore_rows": COCOA_BURN})
cocoa = MCSamples(samples=np.column_stack([c[n if n in ("omegam", "sigma8", "S8", "omegab") else "roman_CL_" + n] for n in names]),
                  weights=c.weights, loglikes=c.loglikes, names=names,
                  labels=[labels[n] for n in names], settings=settings,
                  label="Cocoa")

# ---- marginalized constraints side by side
lines = ["%-10s %26s %26s" % ("", "original: mean +- std", "Cocoa: mean +- std")]
for n in names:
  a, b = orig.getMargeStats().parWithName(n), cocoa.getMargeStats().parWithName(n)
  lines.append("%-10s %15.4f +- %7.4f %15.4f +- %7.4f   shift = %+.2f sigma" % (n, a.mean, a.err, b.mean, b.err,
               (b.mean - a.mean)/np.hypot(a.err, b.err)))
print("\n".join(lines))
with open(OUT + "/compare_chains_constraints.txt", "w") as f:
  f.write("\n".join(lines) + "\n")

ORIG_C, COCOA_C = "#7a7a7a", "#2a6fdb"
g = plots.get_subplot_plotter(width_inch=7)
g.settings.axes_fontsize = 10
g.triangle_plot([orig, cocoa], ["omegam", "sigma8", "S8", "omegab"], filled=[False, True],
                contour_colors=[ORIG_C, COCOA_C], contour_lws=[1.5, 1.5], legend_loc="upper right")
g.export(OUT + "/compare_chains_cosmo.pdf")

g = plots.get_subplot_plotter(width_inch=15)
g.settings.axes_fontsize = 8
g.triangle_plot([orig, cocoa], names[4:], filled=[False, True],
                contour_colors=[ORIG_C, COCOA_C], contour_lws=[1.2, 1.2], legend_loc="upper right")
g.export(OUT + "/compare_chains_nuisance.pdf")

g = plots.get_subplot_plotter(width_inch=18)
g.settings.axes_fontsize = 7
g.triangle_plot([orig, cocoa], names, filled=[False, True],
                contour_colors=[ORIG_C, COCOA_C], contour_lws=[1.0, 1.0], legend_loc="upper right")
g.export(OUT + "/compare_chains_full.pdf")
