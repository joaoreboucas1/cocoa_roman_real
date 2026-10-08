# Reweights the Cocoa chain with the emcee nuisance priors (chains/EXAMPLE_MCMC_CLUSTER_LENSING2)
# to the cosmological priors of the original emcee script (flat in sigma8, Omega_m, Omega_b,
# Gaussian priors on h and omega_b with h from CLASS at fixed theta_s) and compares it
# with the original emcee chain (ns_fix_rich10_v5, steps 150-300).
# Usage (cocoa environment active, from the cocoa main folder Cocoa/):
#   python ./projects/roman_real/chain_comparison/reweight_cocoa.py
# Outputs (reweight_cosmo.pdf, reweight_constraints.txt) are written next to this script.
import os, sys, io, contextlib, importlib.util
import numpy as np
import emcee
import matplotlib
matplotlib.use("Agg")
from getdist import MCSamples, loadMCSamples, plots
from scipy.stats import norm

OUT = os.path.dirname(os.path.abspath(__file__))
P = os.path.dirname(OUT) # projects/roman_real
D = P + "/original_code"
quiet = lambda: contextlib.redirect_stdout(io.StringIO())

# ---------------- original get_hubble (CLASS at fixed 100 theta_s), loaded from the ns_fix script
ORIG = "emcee_cosmo_emu_bin_rich_lens_3bin_HOD_evol_ns_fix_v2.py"
emu = "Emu_delsig_bin_rich_quad_noscatter_HMFCorr_%sAB_cosmo_no_h_%s_v2.hdf5"
zb, zc = ["z0p20toz0p35", "z0p35toz0p50", "z0p50toz0p65"], ["z0p20-0p35", "z0p35-0p50", "z0p50-0p65"]
args = [os.path.join(D, emu % (c, z)) for z in ["z0p3", "z0p4", "z0p5"] for c in ["cen", "mis"]]
args += [os.path.join(D, f"DESY1CL_DelSig_final_analytic_boost_corr_cov_{z}.txt") for z in zc]
args += [os.path.join(D, f"DESY1CL_corr_DelSig_{z}_v3.txt") for z in zb]
args += ["0.30", "0.40", "0.50", "0.70", "x.hdf5", "1.021", "0.025", "1.014", "0.024", "1.016", "0.025"]
args += [os.path.join(D, f"DESY1_src_dist_lens_{z}.txt") for z in zb]
args += ["0.279", "0.429", "0.571", "0.3", "70.", "0", "66"]
sys.path.insert(0, D); sys.argv = [ORIG] + args
spec = importlib.util.spec_from_file_location("nsfix", os.path.join(D, ORIG))
m = importlib.util.module_from_spec(spec)
with quiet():
  spec.loader.exec_module(m)

# ---------------- Cocoa chain with the emcee nuisance priors (case 2)
c = loadMCSamples(P + "/chains/EXAMPLE_MCMC_CLUSTER_LENSING2", settings={"ignore_rows": 0.3})
w0 = c.weights.copy()
ob, oc, H0 = c["omegabh2"], c["omegach2"], c["H0"]
s8, Om, Ob = c["sigma8"], c["omegam"], c["omegab"]
h = H0/100
onu = (Om*h**2 - ob - oc).mean() # massive neutrinos in CAMB's omegam
print("omega_nu h^2 from chain: %.6f (spread %.1e)" % (onu, (Om*h**2 - ob - oc).std()))

# h_CAMB(omega_b, omega_c) at fixed theta_MC: smooth fit in log h -> analytic Jacobian
def design(b, cc, deriv=None):
  terms = [(i, j) for i in range(4) for j in range(4) if i + j <= 3]
  if deriv is None: return np.column_stack([b**i * cc**j for i, j in terms]), terms
  if deriv == "b": return np.column_stack([i*b**max(i-1, 0)*cc**j for i, j in terms])
  return np.column_stack([j*b**i*cc**max(j-1, 0) for i, j in terms])
bn, cn = (ob - 0.022)/0.001, (oc - 0.11)/0.01 # normalised coordinates
X, terms = design(bn, cn)
coef, *_ = np.linalg.lstsq(X, np.log(h), rcond=None)
print("fit of log h_CAMB(omega_b, omega_c): max |residual| %.1e" % np.abs(X @ coef - np.log(h)).max())
dlnh_db = design(bn, cn, "b") @ coef / 0.001
dlnh_dc = design(bn, cn, "c") @ coef / 0.01
# Omega_b = ob/h^2, Omega_m = (ob + oc + onu)/h^2
dOb_db = 1/h**2 - 2*Ob*dlnh_db
dOb_dc = -2*Ob*dlnh_dc
dOm_db = 1/h**2 - 2*Om*dlnh_db
dOm_dc = 1/h**2 - 2*Om*dlnh_dc
J = np.abs(dOb_db*dOm_dc - dOb_dc*dOm_db) # |d(Ob,Om)/d(ob,oc)|

# h_CLASS (original convention: Omega_cdm = Om - Ob) on a subsample, then a smooth fit of h_CAMB - h_CLASS
rng = np.random.default_rng(2)
sub = rng.choice(len(h), 300, replace=False)
with quiet():
  hcl_sub = np.array([m.get_hubble([Ob[i], Om[i] - Ob[i], m.fixed_n_s]) for i in sub])
dh_sub = h[sub] - hcl_sub
Xs, _ = design(bn[sub], cn[sub])
cdh, *_ = np.linalg.lstsq(Xs, dh_sub, rcond=None)
print("h_CAMB - h_CLASS on 300 samples: mean %.5f, std %.5f; fit residual max %.1e"
      % (dh_sub.mean(), dh_sub.std(), np.abs(Xs @ cdh - dh_sub).max()))
hcl = h - X @ cdh

# ---------------- importance weights: emcee prior / Cocoa prior, in (sigma8, Omega_m, Omega_b)
box = (s8 >= 0.65) & (s8 <= 1.05) & (Om >= 0.16) & (Om <= 0.42) & (Ob >= 0.03) & (Ob <= 0.07)
lp_emcee = norm.logpdf(hcl, 0.70, 0.1) + norm.logpdf(Ob*hcl**2, 0.02208, 0.00052)
lp_cocoa = norm.logpdf(ob, 0.02208, 0.00052) - np.log(s8) - np.log(J)
lw = np.where(box, lp_emcee - lp_cocoa, -np.inf)
w = w0*np.exp(lw - lw[np.isfinite(lw)].max())
ess = lambda ww: ww.sum()**2/(ww**2).sum()
print("samples outside the emcee prior box: %.4f of the weight" % (w0[~box].sum()/w0.sum()))
print("effective sample size: before %.0f, after %.0f" % (ess(w0), ess(w)))
print("weight ratio spread (std of log w over samples): %.3f" % np.std(lw[box]))

# ---------------- comparison
names = ["omegam", "sigma8", "S8", "omegab"]
labels = [r"\Omega_\mathrm{m}", r"\sigma_8", r"S_8", r"\Omega_\mathrm{b}"]
st = {"smooth_scale_2D": 0.3, "smooth_scale_1D": 0.3}
cols = np.column_stack([Om, s8, s8*np.sqrt(Om/0.3), Ob])
cw = MCSamples(samples=cols, weights=w0, names=names, labels=labels, settings=st, label="Cocoa, wide priors")
crw = MCSamples(samples=cols, weights=w, names=names, labels=labels, settings=st, label="Cocoa, wide priors, reweighted to emcee priors")
ch = emcee.backends.HDFBackend(D + "/MCMC_DESY1CL_fidchain_AB_Analytic_fid_priors_theta_prior_HOD_evol_ns_fix_rich10_v5.hdf5",
                               read_only=True).get_chain()
f = ch[150:].reshape(-1, 17)
orig = MCSamples(samples=np.column_stack([f[:, 15], f[:, 14], f[:, 14]*np.sqrt(f[:, 15]/0.3), f[:, 16]]),
                 names=names, labels=labels, settings=st, label="original (emcee, steps 150-300)")

lines = ["%-8s %24s %24s %24s %16s %16s" % ("", "original", "Cocoa wide", "Cocoa wide reweighted",
                                             "shift before", "shift after")]
for n in names:
  a, b, r = [s_.getMargeStats().parWithName(n) for s_ in (orig, cw, crw)]
  lines.append("%-8s %13.4f +- %6.4f %13.4f +- %6.4f %13.4f +- %6.4f %+12.2f sigma %+12.2f sigma" %
               (n, a.mean, a.err, b.mean, b.err, r.mean, r.err,
                (b.mean - a.mean)/np.hypot(a.err, b.err), (r.mean - a.mean)/np.hypot(a.err, r.err)))
print("\n".join(lines))
with open(OUT + "/reweight_constraints.txt", "w") as fh:
  fh.write("\n".join(lines) + "\n")

g = plots.get_subplot_plotter(width_inch=7)
g.settings.axes_fontsize = 10
g.triangle_plot([orig, cw, crw], names, filled=[False, True, True],
                contour_colors=["#7a7a7a", "#e07b24", "#2a6fdb"], contour_lws=1.5, legend_loc="upper right")
g.export(OUT + "/reweight_cosmo.pdf")
