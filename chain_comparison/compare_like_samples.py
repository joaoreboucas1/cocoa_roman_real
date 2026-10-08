# Evaluates the cluster lensing chi2 at random samples of the original emcee chain
# (ns_fix_rich10_v5, steps 150-300) with (1) the original code (ns_fix lnprob minus its
# Gaussian priors) and (2) the Cocoa likelihood, with CAMB and with astropy distances.
# Usage (cocoa environment active, from the cocoa main folder Cocoa/):
#   python ./projects/roman_real/chain_comparison/compare_like_samples.py [nsamples]
# The per-sample table is written next to this script (compare_like_samples.txt).
import os, sys, io, contextlib, importlib.util
import numpy as np
import emcee
import astropy.cosmology
from cobaya.model import get_model

NSAMP = int(sys.argv[1]) if len(sys.argv) > 1 else 20
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # projects/roman_real
D = P + "/original_code"
ORIG = "emcee_cosmo_emu_bin_rich_lens_3bin_HOD_evol_ns_fix_v2.py"
CHAIN = D + "/MCMC_DESY1CL_fidchain_AB_Analytic_fid_priors_theta_prior_HOD_evol_ns_fix_rich10_v5.hdf5"

# ---------------- original code (ns_fix), arguments of run_final_DESY1_MCMC_all.pbs
emu = "Emu_delsig_bin_rich_quad_noscatter_HMFCorr_%sAB_cosmo_no_h_%s_v2.hdf5"
zb = ["z0p20toz0p35", "z0p35toz0p50", "z0p50toz0p65"]
zc = ["z0p20-0p35", "z0p35-0p50", "z0p50-0p65"]
args = [os.path.join(D, emu % (c, z)) for z in ["z0p3", "z0p4", "z0p5"] for c in ["cen", "mis"]]
args += [os.path.join(D, f"DESY1CL_DelSig_final_analytic_boost_corr_cov_{z}.txt") for z in zc]
args += [os.path.join(D, f"DESY1CL_corr_DelSig_{z}_v3.txt") for z in zb]
args += ["0.30", "0.40", "0.50", "0.70", "unused_backend.hdf5", "1.021", "0.025", "1.014", "0.024", "1.016", "0.025"]
args += [os.path.join(D, f"DESY1_src_dist_lens_{z}.txt") for z in zb]
args += ["0.279", "0.429", "0.571", "0.3", "70.", "0", "66"]
sys.path.insert(0, D)
sys.argv = [ORIG] + args
spec = importlib.util.spec_from_file_location("nsfix", os.path.join(D, ORIG))
m = importlib.util.module_from_spec(spec)
quiet = lambda: contextlib.redirect_stdout(io.StringIO())
with quiet():
  spec.loader.exec_module(m)

def orig_priors(x, h):
  g = m.compute_gaussian_prior
  return (g(m.Am_prior_mean1, m.Am_prior_sig1, x[11]) + g(m.Am_prior_mean2, m.Am_prior_sig2, x[12])
          + g(m.Am_prior_mean3, m.Am_prior_sig3, x[13]) + g(m.hubble_prior_mean, m.hubble_prior_sig, h)
          + g(m.omegab_prior_mean, m.omegab_prior_sig, x[16]*h**2) + g(m.fmis_prior_mean, m.fmis_prior_sig, x[9])
          + g(m.tau_prior_mean, m.tau_prior_sig, x[10]))

# ---------------- Cocoa model (H0, omegam, omegab basis, LCDM, ns fixed as in ns_fix)
info = {
  "likelihood": {"roman_real.cluster_lensing": {"path": "./external_modules/data/roman_real",
                                                "data_file": "cluster_lensing_desy1.dataset"}},
  "theory": {"camb": {"path": "./external_modules/code/CAMB", "use_renames": True,
             "extra_args": {"halofit_version": "takahashi", "dark_energy_model": "ppf",
                            "lens_potential_accuracy": 1.0, "kmax": 7.5}}},
  "params": {
    "As_1e9": {"prior": {"min": 0.1, "max": 10}, "drop": True},
    "As": {"value": "lambda As_1e9: 1e-9 * As_1e9"},
    "ns": {"value": 0.9649},
    "H0": {"prior": {"min": 40, "max": 100}},
    "omegab": {"prior": {"min": 0.02, "max": 0.08}, "drop": True},
    "omegam": {"prior": {"min": 0.1, "max": 0.6}, "drop": True},
    "mnu": {"value": 0.06}, "w": {"value": -1.0}, "wa": {"value": 0.0}, "tau": {"value": 0.0544},
    "omegabh2": {"value": "lambda omegab, H0: omegab*(H0/100)**2"},
    "omegach2": {"value": "lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708"},
    "sigma8": None,
  },
}
with quiet():
  model = get_model(info)
like = model.likelihood["roman_real.cluster_lensing"]
nuis = ["SIGLOGM_1", "LOGMMIN_1", "LOGM20_1", "ALPHA_1", "SIGLOGM_3", "LOGMMIN_3", "LOGM20_3", "ALPHA_3",
        "BARYON_B", "FMIS", "TAU", "AM1", "AM2", "AM3"]

def cocoa_eval(x, h):
  point = {"roman_CL_" + n: x[i] for i, n in enumerate(nuis)}
  point.update({"H0": 100*h, "omegab": x[16], "omegam": x[15], "As_1e9": 2.1})
  for _ in range(3): # match sigma8 of the emcee sample by rescaling A_s (sigma8 ~ sqrt(A_s))
    ll, der = model.loglikes(point, as_dict=True, return_derived=True)
    if abs(der["sigma8"] - x[14]) < 1e-6: break
    point["As_1e9"] *= (x[14]/der["sigma8"])**2
  chi2_camb = -2*ll["roman_real.cluster_lensing"]
  # same point with the original's astropy distances (FlatLambdaCDM, as in ns_fix)
  chi_ap = astropy.cosmology.FlatLambdaCDM(H0=100*h, Om0=x[15]).comoving_distance(like.cl_zgrid).value
  get_chi = like.provider.get_comoving_radial_distance
  like.provider.get_comoving_radial_distance = lambda z: chi_ap
  dv = like.get_cluster_lensing_datavector(**point)
  like.provider.get_comoving_radial_distance = get_chi
  chi2_ap, i0 = 0.0, 0
  for obs, icov in zip(like.cl_obs, like.cl_icov):
    d = dv[i0:i0+len(obs)] - obs; chi2_ap += d @ icov @ d; i0 += len(obs)
  return chi2_camb, chi2_ap, der["sigma8"]

# ---------------- samples
r = emcee.backends.HDFBackend(CHAIN, read_only=True)
ch, lp = r.get_chain(), r.get_log_prob()
flat, lpf = ch[150:].reshape(-1, 17), lp[150:].reshape(-1)
rng = np.random.default_rng(1)
idx = rng.choice(len(flat), NSAMP, replace=False)

print("%5s %7s %7s %7s %7s | %10s %10s | %10s %10s | %10s %10s" % ("#", "Om", "sigma8", "logMm1", "Ocdm",
      "chi2 stor", "chi2 orig", "chi2 cocoa", "d(c-o)", "cocoa+ap", "d(ap-o)"))
rows = []
for k, i in enumerate(idx):
  x = flat[i]
  with quiet():
    h = m.get_hubble([x[16], x[15] - x[16], m.fixed_n_s])
    lnp = m.lnprob(x)
  pri = orig_priors(x, h)
  chi2_stored, chi2_orig = -2*(lpf[i] - pri), -2*(lnp - pri)
  chi2_c, chi2_ap, s8 = cocoa_eval(x, h)
  rows.append([x[15], x[14], x[1], x[15] - x[16], chi2_stored, chi2_orig, chi2_c, chi2_ap])
  print("%5d %7.4f %7.4f %7.3f %7.4f | %10.4f %10.4f | %10.4f %+10.4f | %10.4f %+10.2e" % (k, x[15], x[14], x[1], x[15] - x[16],
        chi2_stored, chi2_orig, chi2_c, chi2_c - chi2_orig, chi2_ap, chi2_ap - chi2_orig))
rows = np.array(rows)
d = rows[:, 6] - rows[:, 5]
print("\nCocoa (CAMB distances) - original chi2: mean %+.3f, std %.3f, min %+.3f, max %+.3f" % (d.mean(), d.std(), d.min(), d.max()))
print("Cocoa (astropy distances) - original chi2: max |diff| %.2e" % np.abs(rows[:, 7] - rows[:, 5]).max())
print("stored - recomputed original chi2: mean %+.3f, max |diff| %.3f" % ((rows[:, 4] - rows[:, 5]).mean(), np.abs(rows[:, 4] - rows[:, 5]).max()))
for j, n in [(0, "Omega_m"), (1, "sigma8"), (3, "Omega_CDM")]:
  print("corr(Cocoa - original, %s) = %+.2f" % (n, np.corrcoef(d, rows[:, j])[0, 1]))
np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_like_samples.txt"), rows,
           header="Om sigma8 logMmin_1 Ocdm chi2_stored chi2_orig chi2_cocoa_camb chi2_cocoa_astropy", fmt="%.6f")
