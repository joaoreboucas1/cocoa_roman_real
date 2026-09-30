#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Compares the Cocoa port of the cluster lensing likelihood
# (roman_real.cluster_lensing) against the original code in Cluster_cosmo/
# (emcee_cosmo_emu_bin_rich_lens_3bin_HOD_evol_ns_free_w0waCDM_v2.py).
#
# The original script is imported as a module, with the command-line
# arguments of run_final_DESY1_MCMC_all.pbs, so the comparison uses its own
# lnprob, lensing_corr, emulators, masked covariances and data.
#
# For each test point:
#   1. h is derived with the original get_hubble (CLASS at fixed theta_s),
#   2. Cocoa is evaluated at H0 = 100 h, and sigma8 is read from CAMB,
#   3. the original lnprob is evaluated with that sigma8; its Gaussian priors
#      are subtracted so that only -chi2/2 remains.
# The Cocoa port is compared twice: with distances from the theory code (the
# port as it runs) and with the original's astropy distances (which isolates
# the port itself; the difference should be at machine precision).
#
# Run from the cocoa main folder (Cocoa/), after
#   conda activate cocoa; source start_cocoa.sh
#   python ./projects/roman_real/scripts/compare_cluster_lensing.py
# ------------------------------------------------------------------------------

# USER INPUT: path to the Cluster_cosmo/ folder (e.g. "/Users/me/cosmo/Cluster_cosmo")
CLUSTER_COSMO_DIR = ""
CLUSTER_COSMO_DIR = "/Users/joao/cosmo/Cluster_cosmo"

# ------------------------------------------------------------------------------
import os
import sys
import importlib.util
import numpy as np
import astropy.cosmology
from cobaya.model import get_model

if not CLUSTER_COSMO_DIR:
  sys.exit("Set CLUSTER_COSMO_DIR at the top of this script to the Cluster_cosmo/ folder")
D = os.path.abspath(os.path.expanduser(CLUSTER_COSMO_DIR))
ORIG = "emcee_cosmo_emu_bin_rich_lens_3bin_HOD_evol_ns_free_w0waCDM_v2.py"
if not os.path.isfile(os.path.join(D, ORIG)):
  sys.exit(f"{ORIG} not found in {D}")

# ------------------------------------------------------------------------------
# Import the original code, with the arguments of run_final_DESY1_MCMC_all.pbs
# ------------------------------------------------------------------------------
emu = "Emu_delsig_bin_rich_quad_noscatter_HMFCorr_%sAB_cosmo_no_h_%s_v2.hdf5"
zb  = ["z0p20toz0p35", "z0p35toz0p50", "z0p50toz0p65"]
zc  = ["z0p20-0p35", "z0p35-0p50", "z0p50-0p65"]
args = [os.path.join(D, emu % (c, z)) for z in ["z0p3", "z0p4", "z0p5"] for c in ["cen", "mis"]]
args += [os.path.join(D, f"DESY1CL_DelSig_final_analytic_boost_corr_cov_{z}.txt") for z in zc]
args += [os.path.join(D, f"DESY1CL_corr_DelSig_{z}_v3.txt") for z in zb]
args += ["0.30", "0.40", "0.50", "0.70", "unused_backend.hdf5",
         "1.021", "0.025", "1.014", "0.024", "1.016", "0.025"]
args += [os.path.join(D, f"DESY1_src_dist_lens_{z}.txt") for z in zb]
args += ["0.279", "0.429", "0.571", "0.3", "70.", "0", "66"]

sys.path.insert(0, D) # for the original predict_emulator and config
sys.argv = [ORIG] + args
spec = importlib.util.spec_from_file_location("cluster_cosmo_orig", os.path.join(D, ORIG))
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig) # __name__ != "__main__": emcee does not run

# ------------------------------------------------------------------------------
# Cocoa model: cluster lensing likelihood + CAMB
# ------------------------------------------------------------------------------
info = {
  "likelihood": {
    "roman_real.cluster_lensing": {
      "path": "./external_modules/data/roman_real",
      "data_file": "cluster_lensing_desy1.dataset",
    }
  },
  "theory": {
    "camb": {
      "path": "./external_modules/code/CAMB",
      "use_renames": True,
      "extra_args": {"halofit_version": "takahashi", "dark_energy_model": "ppf",
                     "lens_potential_accuracy": 1.0, "kmax": 7.5},
    }
  },
  "params": {
    "As_1e9": {"prior": {"min": 0.5, "max": 5}, "drop": True},
    "As": {"value": "lambda As_1e9: 1e-9 * As_1e9"},
    "ns": {"prior": {"min": 0.87, "max": 1.07}},
    "H0": {"prior": {"min": 40, "max": 100}},
    "omegab": {"prior": {"min": 0.03, "max": 0.07}, "drop": True},
    "omegam": {"prior": {"min": 0.1, "max": 0.9}, "drop": True},
    "mnu": {"value": 0.06},
    "w": {"prior": {"min": -3, "max": -0.01}},
    "wa": {"prior": {"min": -3, "max": 3}},
    "tau": {"value": 0.0697186},
    "omegabh2": {"value": "lambda omegab, H0: omegab*(H0/100)**2"},
    "omegach2": {"value": "lambda omegam, omegab, mnu, H0: (omegam-omegab)*(H0/100)**2-(mnu*(3.046/3)**0.75)/94.0708"},
    "sigma8": None,
  },
}
model = get_model(info)
like = model.likelihood["roman_real.cluster_lensing"]

# ------------------------------------------------------------------------------
# Test points: HOD/nuisance values from the original script (variable x)
# ------------------------------------------------------------------------------
nuisance = {"roman_CL_SIGLOGM_1": 0.3, "roman_CL_LOGMMIN_1": 12.5, "roman_CL_LOGM20_1": 14.5, "roman_CL_ALPHA_1": 1.26,
            "roman_CL_SIGLOGM_3": 0.3, "roman_CL_LOGMMIN_3": 12.5, "roman_CL_LOGM20_3": 14.5, "roman_CL_ALPHA_3": 1.26,
            "roman_CL_PCA": -0.5, "roman_CL_FMIS": 0.16, "roman_CL_TAU": 0.16,
            "roman_CL_AM1": 1.03, "roman_CL_AM2": 1.03, "roman_CL_AM3": 0.97}
points = [
  {"name": "LCDM", "As_1e9": 2.1, "ns": 0.9649, "omegab": 0.05, "omegam": 0.27, "w": -1.0, "wa": 0.0},
  {"name": "w0wa", "As_1e9": 2.1, "ns": 0.9649, "omegab": 0.05, "omegam": 0.27, "w": -0.9, "wa": 0.3},
]

def orig_datavector(x):
  # lnprob lines 257-303, using the original module's functions and data
  t = (orig.redshift2_dat - orig.redshift1_dat)/(orig.redshift3_dat - orig.redshift1_dat)
  hodA, hodC = np.array(x[0:4]), np.array(x[4:8])
  hods = [hodA, hodA + t*(hodC - hodA), hodC]
  cosmo = orig.astropy.cosmology.Flatw0waCDM(H0=h*100.0, Om0=x[15], w0=x[18], wa=x[19])
  dv = []
  for i in range(3):
    n = i + 1
    src = [getattr(orig, "%s%d" % (s, n)) for s in ["src_dist", "zbins_upper", "zbins_lower"]]
    corr = orig.lensing_corr(getattr(orig, "redshift%d_dat" % n),
                             (src[1] + src[2])/2.0, orig.cosmo_fid, cosmo, *src)
    cos = [x[17], x[14], x[18], x[19], x[15]-x[16], x[16], orig.fixed_alpha_s, orig.fixed_Neff]
    pc = np.array(list(hods[i]) + [x[8]] + cos)
    pm = np.array(list(hods[i]) + [x[8], x[10]] + cos)
    mask, a = getattr(orig, "mask%d" % n), getattr(orig, "a%d" % n)
    cen = getattr(orig, "emulate_fun%dcen" % n)(pc)[mask][orig.rich_cut_min:orig.rich_cut_max]
    mis = getattr(orig, "emulate_fun%dmis" % n)(pm)[mask][orig.rich_cut_min:orig.rich_cut_max]
    dv.append(x[11+i]*corr*((1.0 - x[9])*cen + x[9]*mis)/a**2)
  return dv

def chi2_per_bin(dv, obs, icov):
  return np.array([(d - o) @ ic @ (d - o) for d, o, ic in zip(dv, obs, icov)])

def split(dv):
  return np.split(dv, np.cumsum([len(o) for o in like.cl_obs])[:-1])

print("\n" + "="*78)
for p in points:
  # (1) h from the original code (CLASS at fixed theta_s)
  h = orig.get_hubble([p["omegab"], p["omegam"] - p["omegab"], p["ns"], p["w"], p["wa"]])
  # (2) Cocoa at H0 = 100 h
  point = {k: v for k, v in p.items() if k != "name"}
  point.update(nuisance)
  point["H0"] = 100.0*h
  loglikes, derived = model.loglikes(point, as_dict=True, return_derived=True)
  sigma8 = derived["sigma8"]
  dv_cocoa = like.get_cluster_lensing_datavector(**nuisance)
  # same, with the original's astropy distances
  chi_astropy = orig.astropy.cosmology.Flatw0waCDM(H0=100.0*h, Om0=p["omegam"],
    w0=p["w"], wa=p["wa"]).comoving_distance(like.cl_zgrid).value
  get_chi = like.provider.get_comoving_radial_distance
  like.provider.get_comoving_radial_distance = lambda z: chi_astropy
  dv_cocoa_astropy = like.get_cluster_lensing_datavector(**nuisance)
  like.provider.get_comoving_radial_distance = get_chi

  # (3) original code, same point
  x = [nuisance[k] for k in ["roman_CL_SIGLOGM_1", "roman_CL_LOGMMIN_1", "roman_CL_LOGM20_1", "roman_CL_ALPHA_1",
                             "roman_CL_SIGLOGM_3", "roman_CL_LOGMMIN_3", "roman_CL_LOGM20_3", "roman_CL_ALPHA_3",
                             "roman_CL_PCA", "roman_CL_FMIS", "roman_CL_TAU",
                             "roman_CL_AM1", "roman_CL_AM2", "roman_CL_AM3"]]
  x += [sigma8, p["omegam"], p["omegab"], p["ns"], p["w"], p["wa"]]
  lnprob = orig.lnprob(x)
  priors = sum(orig.compute_gaussian_prior(np.float64(m), np.float64(s), v) for m, s, v in [
    (orig.Am_prior_mean1, orig.Am_prior_sig1, x[11]), (orig.Am_prior_mean2, orig.Am_prior_sig2, x[12]),
    (orig.Am_prior_mean3, orig.Am_prior_sig3, x[13]), (orig.hubble_prior_mean, orig.hubble_prior_sig, h),
    (orig.omegab_prior_mean, orig.omegab_prior_sig, x[16]*h**2), (orig.fmis_prior_mean, orig.fmis_prior_sig, x[9]),
    (orig.tau_prior_mean, orig.tau_prior_sig, x[10])])
  chi2_orig_lnprob = -2.0*(lnprob - priors)
  dv_orig = orig_datavector(x)
  obs  = [orig.obs1, orig.obs2, orig.obs3]
  icov = [orig.icov1, orig.icov2, orig.icov3]
  c_orig = chi2_per_bin(dv_orig, obs, icov)
  c_cocoa = chi2_per_bin(split(dv_cocoa), like.cl_obs, like.cl_icov)
  c_cocoa_astropy = chi2_per_bin(split(dv_cocoa_astropy), like.cl_obs, like.cl_icov)
  dv_orig = np.concatenate(dv_orig)

  print(f"Point {p['name']}: w0 = {p['w']}, wa = {p['wa']}, Omega_m = {p['omegam']}, "
        f"Omega_b = {p['omegab']}, ns = {p['ns']}")
  print(f"  h (CLASS, fixed theta_s) = {h:.6f}, sigma8 (CAMB) = {sigma8:.6f}")
  print(f"  original: chi2 from lnprob (priors removed) = {chi2_orig_lnprob:.6f}, "
        f"from its data vector = {c_orig.sum():.6f}")
  print(f"  Cocoa logp = {loglikes['roman_real.cluster_lensing']:.6f} (chi2 = {-2*loglikes['roman_real.cluster_lensing']:.6f})")
  print("  %-32s %12s %12s %12s %12s %14s" % ("", "chi2 z-bin1", "chi2 z-bin2", "chi2 z-bin3", "chi2 total", "max|ddv/dv|"))
  print("  %-32s %12.6f %12.6f %12.6f %12.6f %14s" % ("original", *c_orig, c_orig.sum(), "-"))
  for label, c, dv in [("Cocoa (theory-code distances)", c_cocoa, dv_cocoa),
                       ("Cocoa (astropy distances)", c_cocoa_astropy, dv_cocoa_astropy)]:
    print("  %-32s %12.6f %12.6f %12.6f %12.6f %14.3e" % (label, *c, c.sum(), np.max(np.abs(dv/dv_orig - 1))))
    print("  %-32s %12.6f %12.6f %12.6f %12.6f" % ("  difference (Cocoa - original)", *(c - c_orig), c.sum() - c_orig.sum()))
  print("="*78)
