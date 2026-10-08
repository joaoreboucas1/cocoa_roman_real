# Compares the original emcee chain (original_code/, ns_fix parameter order)
# with a Cocoa port chain (chains/EXAMPLE_MCMC_CLUSTER_LENSING<case>).
# Usage (cocoa environment active): python compare_chains.py [case] [emcee_burn]
#   case 1 (default): Table I priors, outputs compare_chains_*.{pdf,txt}
#   case 2: priors of the emcee script, outputs compare_chains_*_2.{pdf,txt}
#   case all: original and both Cocoa chains, outputs compare_chains_*_all.{pdf,txt}
# emcee_burn: emcee steps discarded (default 150); other values add _burn<N> to the output names.
# Figures and the constraints table are written next to this script.
import os
import sys
import numpy as np
import emcee
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, loadMCSamples, plots

OUT = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(OUT) # projects/roman_real
ORIG_BURN = int(sys.argv[2]) if len(sys.argv) > 2 else 150 # same as original_code/plot_chain.py
COCOA_BURN = 0.3    # fraction of each Cobaya chain dropped
CASE = sys.argv[1] if len(sys.argv) > 1 else "1"
SUFFIX = ("" if CASE == "1" else "_" + CASE) + ("" if ORIG_BURN == 150 else "_burn%d" % ORIG_BURN)

# ---- original emcee chain (ns_fix parameter order)
orig_names = ["SIGLOGM_1", "LOGMMIN_1", "LOGM20_1", "ALPHA_1",
              "SIGLOGM_3", "LOGMMIN_3", "LOGM20_3", "ALPHA_3",
              "BARYON_B", "FMIS", "TAU", "AM1", "AM2", "AM3", "sigma8", "omegam", "omegab"]
ch = emcee.backends.HDFBackend(P + "/original_code/MCMC_DESY1CL_fidchain_AB_Analytic_fid_priors_theta_prior_HOD_evol_ns_fix_rich10_v5.hdf5",
                               read_only=True).get_chain()
flat = ch[ORIG_BURN:].reshape(-1, ch.shape[2])
cols = {n: flat[:, i] for i, n in enumerate(orig_names)}
cols["S8"] = cols["sigma8"]*np.sqrt(cols["omegam"]/0.3)

# omega_b h^2 of the emcee chain: h is not stored, the original code derives it from CLASS
# at fixed 100 theta_s (get_hubble in ns_fix_v2.py, reproduced below). CLASS is run on a
# subsample and h(Omega_b, Omega_m) is fitted with a smooth polynomial, checked on held-out samples.
from classy import Class
def get_hubble(Ob, Ocdm, ns=0.9649):
  cosmo = Class()
  cosmo.set({'Omega_b': Ob, 'Omega_cdm': Ocdm, 'n_s': ns, 'N_ur': 2.0328, 'N_ncdm': 1.0,
             'omega_ncdm': 0.0006442, '100*theta_s': 1.041533, 'recombination': 'HyRec',
             'tau_reio': 0.0544, 'non linear': 'halofit', 'P_k_max_1/Mpc': 10.0, 'output': 'mPk'})
  cosmo.compute()
  return cosmo.h()
def poly(b, m):
  bn, mn = (b - 0.045)/0.005, (m - 0.27)/0.03
  return np.column_stack([bn**i * mn**j for i in range(4) for j in range(4) if i + j <= 3])
rng = np.random.default_rng(3)
sub = rng.choice(len(flat), 300, replace=False)
h_sub = np.array([get_hubble(cols["omegab"][i], cols["omegam"][i] - cols["omegab"][i]) for i in sub])
fit, test = sub[:250], slice(250, 300)
coef, *_ = np.linalg.lstsq(poly(cols["omegab"][fit], cols["omegam"][fit]), np.log(h_sub[:250]), rcond=None)
h_test = np.exp(poly(cols["omegab"][sub[test]], cols["omegam"][sub[test]]) @ coef)
print("h(Omega_b, Omega_m) fit for the emcee chain: max |error| on 50 held-out samples = %.1e"
      % np.abs(h_test - h_sub[test]).max())
cols["ombh2"] = cols["omegab"]*np.exp(poly(cols["omegab"], cols["omegam"]) @ coef)**2

labels = {"omegam": r"\Omega_\mathrm{m}", "sigma8": r"\sigma_8", "S8": r"S_8", "omegab": r"\Omega_\mathrm{b}",
          "ombh2": r"\Omega_\mathrm{b} h^2",
          "SIGLOGM_1": r"\sigma_{\log M}^{(1)}", "LOGMMIN_1": r"\log M_\mathrm{min}^{(1)}", "LOGM20_1": r"\log M_{20}^{(1)}",
          "ALPHA_1": r"\alpha^{(1)}", "SIGLOGM_3": r"\sigma_{\log M}^{(3)}", "LOGMMIN_3": r"\log M_\mathrm{min}^{(3)}",
          "LOGM20_3": r"\log M_{20}^{(3)}", "ALPHA_3": r"\alpha^{(3)}", "BARYON_B": r"B", "FMIS": r"f_\mathrm{mis}",
          "TAU": r"\tau", "AM1": r"A_{m,1}", "AM2": r"A_{m,2}", "AM3": r"A_{m,3}"}
names = list(labels)
cosmo_names = ["omegam", "sigma8", "S8", "omegab", "ombh2"]
nuis_names = [n for n in names if n not in cosmo_names]
cocoa_col = {"omegam": "omegam", "sigma8": "sigma8", "S8": "S8", "omegab": "omegab", "ombh2": "omegabh2"}
settings = {"smooth_scale_2D": 0.3, "smooth_scale_1D": 0.3}
orig = MCSamples(samples=np.column_stack([cols[n] for n in names]), names=names,
                 labels=[labels[n] for n in names], settings=settings,
                 label="original (emcee, steps %d-%d)" % (ORIG_BURN, ch.shape[0]))

# ---- Cocoa port chains (Cobaya), renamed to the same parameter names
def load_cocoa(case, label):
  c = loadMCSamples(P + "/chains/EXAMPLE_MCMC_CLUSTER_LENSING" + case, settings={"ignore_rows": COCOA_BURN})
  return MCSamples(samples=np.column_stack([c[cocoa_col.get(n, "roman_CL_" + n)] for n in names]),
                   weights=c.weights, loglikes=c.loglikes, names=names,
                   labels=[labels[n] for n in names], settings=settings, label=label)

if CASE == "all":
  cocoas = [load_cocoa("1", "Cocoa, Table I priors"), load_cocoa("2", "Cocoa, wide priors")]
  colors = ["#7a7a7a", "#2a6fdb", "#e07b24"]
else:
  cocoas = [load_cocoa(CASE, "Cocoa")]
  colors = ["#7a7a7a", "#2a6fdb"]
samples_list = [orig] + cocoas
filled = [False] + [True]*len(cocoas)

# ---- marginalized constraints side by side (shifts of each Cocoa chain w.r.t. the original)
heads = ["original"] + (["Cocoa Table I", "Cocoa wide"] if CASE == "all" else ["Cocoa"])
lines = ["%-10s" % "" + "".join("%30s" % (h + ": mean +- std") for h in heads) + "".join("   shift %-13s" % h for h in heads[1:])]
for n in names:
  st = [s_.getMargeStats().parWithName(n) for s_ in samples_list]
  a = st[0]
  lines.append("%-10s" % n + "".join("%19.4f +- %7.4f" % (b.mean, b.err) for b in st)
               + "".join("   %+6.2f sigma      " % ((b.mean - a.mean)/np.hypot(a.err, b.err)) for b in st[1:]))
print("\n".join(lines))
with open(OUT + "/compare_chains_constraints" + SUFFIX + ".txt", "w") as f:
  f.write("\n".join(lines) + "\n")

g = plots.get_subplot_plotter(width_inch=7)
g.settings.axes_fontsize = 10
g.triangle_plot(samples_list, cosmo_names, filled=filled,
                contour_colors=colors, contour_lws=1.5, legend_loc="upper right")
g.export(OUT + "/compare_chains_cosmo" + SUFFIX + ".pdf")

g = plots.get_subplot_plotter(width_inch=15)
g.settings.axes_fontsize = 8
g.triangle_plot(samples_list, nuis_names, filled=filled,
                contour_colors=colors, contour_lws=1.2, legend_loc="upper right")
g.export(OUT + "/compare_chains_nuisance" + SUFFIX + ".pdf")

g = plots.get_subplot_plotter(width_inch=18)
g.settings.axes_fontsize = 7
g.triangle_plot(samples_list, names, filled=filled,
                contour_colors=colors, contour_lws=1.0, legend_loc="upper right")
g.export(OUT + "/compare_chains_full" + SUFFIX + ".pdf")
