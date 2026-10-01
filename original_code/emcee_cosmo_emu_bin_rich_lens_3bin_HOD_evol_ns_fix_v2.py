import predict_emulator
import argparse
import numpy as np
import time
import emcee
import h5py as h5
import scipy 
import os
import camb
from classy import Class

from astropy.io import fits
from astropy import units as u
import astropy.cosmology

import multiprocessing
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser()
parser.add_argument('input_emu_filename1cen')
parser.add_argument('input_emu_filename1mis')
parser.add_argument('input_emu_filename2cen')
parser.add_argument('input_emu_filename2mis')
parser.add_argument('input_emu_filename3cen')
parser.add_argument('input_emu_filename3mis')
parser.add_argument('input_cov1')
parser.add_argument('input_cov2')
parser.add_argument('input_cov3')
parser.add_argument('input_data1')
parser.add_argument('input_data2')
parser.add_argument('input_data3')
parser.add_argument('redshift1')
parser.add_argument('redshift2')
parser.add_argument('redshift3')
parser.add_argument('hubble') #this is the hubble constant assumed when interpreting data units (i.e. for the rp cut)
parser.add_argument('backend')
parser.add_argument('Am_prior_mean1') #mean of gaussian prior on lensing calibration uncertainty parameter see McClintock et al. 2019
parser.add_argument('Am_prior_sigma1') #width of gaussian prior on lensing calibration uncertainty parameter see McClintock et al. 2019
parser.add_argument('Am_prior_mean2') #mean of gaussian prior on lensing calibration uncertainty parameter see McClintock et al. 2019
parser.add_argument('Am_prior_sigma2') #width of gaussian prior on lensing calibration uncertainty parameter see McClintock et al. 2019
parser.add_argument('Am_prior_mean3') #mean of gaussian prior on lensing calibration uncertainty parameter see McClintock et al. 2019
parser.add_argument('Am_prior_sigma3') #width of gaussian prior on lensing calibration uncertainty parameter see McClintock et al. 2019
parser.add_argument('src_dist_file1')
parser.add_argument('src_dist_file2')
parser.add_argument('src_dist_file3')
parser.add_argument('redshift_corr1')
parser.add_argument('redshift_corr2')
parser.add_argument('redshift_corr3')
parser.add_argument('fid_Om')
parser.add_argument('fid_H0')
parser.add_argument('richness_cut_min') #each richness bin is 11 elements
parser.add_argument('richness_cut_max')
parser.add_argument('--previous_chain')
parser.add_argument('--diag_cov')
args = parser.parse_args()

Am_prior_mean1 = np.float64(args.Am_prior_mean1)
Am_prior_sig1  = np.float64(args.Am_prior_sigma1)

Am_prior_mean2 = np.float64(args.Am_prior_mean2)
Am_prior_sig2  = np.float64(args.Am_prior_sigma2)

Am_prior_mean3 = np.float64(args.Am_prior_mean3)
Am_prior_sig3  = np.float64(args.Am_prior_sigma3)

redshift1 = np.float64(args.redshift1)
redshift2 = np.float64(args.redshift2)
redshift3 = np.float64(args.redshift3)

redshift1_dat = np.float64(args.redshift_corr1)
redshift2_dat = np.float64(args.redshift_corr2)
redshift3_dat = np.float64(args.redshift_corr3)

hubble = float(args.hubble)

src_dist_file1 = np.genfromtxt(args.src_dist_file1)
zbins_upper1 = src_dist_file1[:,0]
zbins_lower1 = src_dist_file1[:,1]
src_dist1    = src_dist_file1[:,2]

src_dist_file2 = np.genfromtxt(args.src_dist_file2)
zbins_upper2 = src_dist_file2[:,0]
zbins_lower2 = src_dist_file2[:,1]
src_dist2    = src_dist_file2[:,2]

src_dist_file3 = np.genfromtxt(args.src_dist_file3)
zbins_upper3 = src_dist_file3[:,0]
zbins_lower3 = src_dist_file3[:,1]
src_dist3    = src_dist_file3[:,2]

rich_cut_max = int(args.richness_cut_max)
rich_cut_min = int(args.richness_cut_min)

cosmo_fid = astropy.cosmology.FlatLambdaCDM(H0 = float(args.fid_H0), Om0 = float(args.fid_Om))

a1 = 1. / (1. + redshift1)
emulate_fun1cen, rp_binmin1, rp_binmax1 = predict_emulator.get_emulate_fun(args.input_emu_filename1cen)
emulate_fun1mis, rp_binmin1, rp_binmax1 = predict_emulator.get_emulate_fun(args.input_emu_filename1mis)
rp1 = (a1 / hubble)*(rp_binmin1 + rp_binmax1)/2.0
mask1 = np.where(rp1>0.2)
cov1 = np.genfromtxt(args.input_cov1)

a2 = 1. / (1. + redshift2)
emulate_fun2cen, rp_binmin2, rp_binmax2 = predict_emulator.get_emulate_fun(args.input_emu_filename2cen)
emulate_fun2mis, rp_binmin2, rp_binmax2 = predict_emulator.get_emulate_fun(args.input_emu_filename2mis)
rp2 = (a2 / hubble)*(rp_binmin2 + rp_binmax2)/2.0
mask2 = np.where(rp2>0.2)
cov2 = np.genfromtxt(args.input_cov2)

a3 = 1. / (1. + redshift3)
emulate_fun3cen, rp_binmin3, rp_binmax3 = predict_emulator.get_emulate_fun(args.input_emu_filename3cen)
emulate_fun3mis, rp_binmin3, rp_binmax3 = predict_emulator.get_emulate_fun(args.input_emu_filename3mis)
rp3 = (a3 / hubble)*(rp_binmin3 + rp_binmax3)/2.0
mask3 = np.where(rp3>0.2)
cov3 = np.genfromtxt(args.input_cov3)

#fixed_d1   = float(args.fix_d1)
#fixed_d2   = float(args.fix_d2)
#fixed_d3   = float(args.fix_d3)
fixed_n_s  = 0.9649
fixed_w0   = -1.0
fixed_wa   = 0.0
fixed_alpha_s = 0.0
fixed_Neff    = 3.0238
hubble_prior_mean  = 0.70
hubble_prior_sig = 0.1

omegab_prior_mean  = 0.02208
omegab_prior_sig = 0.00052

fmis_prior_mean = 0.165
fmis_prior_sig  = 0.09

tau_prior_mean  = 0.166
tau_prior_sig   = 0.07

theta_star_prior_mean = 1.04110
theta_star_prior_sig  = 0.00031

if len(cov1[0])==90:
  cov1 = cov1[mask1]
  cov1 = cov1.T[mask1]
  cov1 = cov1.T

if len(cov2[0])==90:
  cov2 = cov2[mask2]
  cov2 = cov2.T[mask2]
  cov2 = cov2.T

if len(cov3[0])==90:
  cov3 = cov3[mask3]
  cov3 = cov3.T[mask3]
  cov3 = cov3.T

cov1 = cov1[rich_cut_min:rich_cut_max]
cov1 = cov1.T[rich_cut_min:rich_cut_max]
cov1 = cov1.T

cov2 = cov2[rich_cut_min:rich_cut_max]
cov2 = cov2.T[rich_cut_min:rich_cut_max]
cov2 = cov2.T

cov3 = cov3[rich_cut_min:rich_cut_max]
cov3 = cov3.T[rich_cut_min:rich_cut_max]
cov3 = cov3.T

if args.diag_cov:
  cov1 = np.diag(np.diag(cov1))
  cov2 = np.diag(np.diag(cov2))
  cov3 = np.diag(np.diag(cov3))
  
icov1 = np.linalg.pinv(cov1)
icov2 = np.linalg.pinv(cov2)
icov3 = np.linalg.pinv(cov3)

#obs1 = np.genfromtxt(args.input_data1)[:,1]/a1**2.0 #observable is already masked
#obs2 = np.genfromtxt(args.input_data2)[:,1]/a2**2.0 #observable is already masked
#obs3 = np.genfromtxt(args.input_data3)[:,1]/a3**2.0 #observable is already masked

obs1 = np.genfromtxt(args.input_data1)[:,1][rich_cut_min:rich_cut_max] #observable is already masked
obs2 = np.genfromtxt(args.input_data2)[:,1][rich_cut_min:rich_cut_max] #observable is already masked
obs3 = np.genfromtxt(args.input_data3)[:,1][rich_cut_min:rich_cut_max] #observable is already masked

def compute_sigcrit_inv(z_lens, z_source, Zgrid, Rgrid):
  csq_over_G = 2.494e12 # 3c^2/(8*pi*G) Msun pc^-1
  
  Dlens = np.interp(z_lens, Zgrid, Rgrid) / (1. + z_lens) #physical-pc
  Dsrc = np.interp(z_source, Zgrid, Rgrid) / (1. + z_source) #physical-pc
  
  dummy =  ( ( 4. * np.pi * (csq_over_G)**-1.0 ) * (Dlens/Dsrc) * (Dsrc - Dlens) )

  dummy[dummy<0]=0.0
  return dummy

def compute_theta_star(H0, ombh2, omch2, w0, wa, ns, sig8, num_nu_massless, num_nu_massive, omnuh2):
  As_ref = 2.0830e-9

  cp=camb.set_params(H0=H0, ombh2=ombh2, omch2=omch2, omnuh2=omnuh2, w=w0, wa=wa, ns=ns, As=As_ref, num_nu_massless=num_nu_massless, num_nu_massive=num_nu_massive, WantTransfer=False, WantDerivedParameters=True, dark_energy_model="ppf")
  results = camb.get_results(cp)
  derived_params = results.get_derived_params()

  return derived_params['thetastar']

def lensing_corr(z_lens, z_source, cosmo_fid, cosmo, src_dist, zbins_upper, zbins_lower):
  Zgrid = np.linspace(0.0, 4.0, 1000)
  Rgrid_fid = np.array([cosmo_fid.comoving_distance(x).value*1e6 for x in Zgrid]) #comoving pc
  Rgrid     = np.array([cosmo.comoving_distance(x).value*1e6 for x in Zgrid])

  sigcrit_inv_fid = compute_sigcrit_inv(z_lens, z_source, Zgrid, Rgrid_fid)
  sigcrit_inv     = compute_sigcrit_inv(z_lens, z_source, Zgrid, Rgrid)
  
  return np.sum( src_dist * (zbins_upper - zbins_lower) * sigcrit_inv_fid*sigcrit_inv) / np.sum( src_dist * (zbins_upper - zbins_lower) * sigcrit_inv_fid**2.0)
  
def compute_gaussian_prior(mu, sigma, val):
  gauss_likelihood = - ( (val - mu)**2.0 )  / (2.0 * sigma**2.0)
  return gauss_likelihood

def get_hubble(p):
    #set parameters to configure CLASS computations
    params = {
        'Omega_b': p[0],
        'Omega_cdm': p[1],
        'n_s': p[2], 
        'N_ur': 2.0328,
        'N_ncdm' : 1.0,
        'omega_ncdm' : 0.0006442,
        '100*theta_s' : 1.041533, #from Summit paper
        'recombination' : 'HyRec',
        'tau_reio' : 0.0544,
        'non linear' : 'halofit',#option for computing non-linear power spectum
        'P_k_max_1/Mpc':10.0,
        'output': 'mPk' #which quantities we want CLASS to compute
    }
    #initialize Class instance and set parameters
    cosmo = Class()
    cosmo.set(params)
    cosmo.compute()
    h = cosmo.h()
    return h

def lnprob(x):
  siglogMA = x[0]
  logMminA = x[1]
  logM20A  = x[2]
  alphaA   = x[3]
  
  siglogMC = x[4]
  logMminC = x[5]
  logM20C  = x[6]
  alphaC   = x[7]

  siglogMB = siglogMA + ( (redshift2_dat - redshift1_dat)/(redshift3_dat - redshift1_dat) ) * (siglogMC - siglogMA)
  logMminB = logMminA + ( (redshift2_dat - redshift1_dat)/(redshift3_dat - redshift1_dat) ) * (logMminC - logMminA)
  logM20B  = logM20A  + ( (redshift2_dat - redshift1_dat)/(redshift3_dat - redshift1_dat) ) * (logM20C  - logM20A)
  alphaB   = alphaA   + ( (redshift2_dat - redshift1_dat)/(redshift3_dat - redshift1_dat) ) * (alphaC   - alphaA)

  if ( (siglogMA<0.01) or (siglogMA>0.60) ) or ( (logMminA>13.4) or (logMminA<11.2) ) or ( (logM20A>15.2) or (logM20A<14.0) ) or ( (alphaA<0.5) or (alphaA>2.50) ) or ( (siglogMB<0.01) or (siglogMB>0.60) ) or ( (logMminB>13.4) or (logMminB<11.2) ) or ( (logM20B>15.2) or (logM20B<14.0) ) or ( (alphaB<0.5) or (alphaB>2.50) )  or ( (siglogMC<0.01) or (siglogMC>0.60) ) or ( (logMminC>13.4) or (logMminC<11.2) ) or ( (logM20C>15.2) or (logM20C<14.0) ) or ( (alphaC<0.5) or (alphaC>2.50) )  or ( (x[8]>0.0) or (x[8]<-2.0) ) or ( (x[9]<0.0) or (x[9]>1.0) ) or ( (x[10]<0.0) or (x[10]>1.0) ) or ( (x[14]<0.65) or (x[14]>1.05) )  or ( (x[15]<0.16) or (x[15]>0.42) )  or ( (x[16]<0.03) or (x[16]>0.07) ) :
    print("Out of Prior Bounds!")
    return - np.inf
  else:
    try:
      h_link      = get_hubble([x[16], x[15]-x[16], fixed_n_s])
      Omb_link    = np.float64(x[16])
      Am1_link    = np.float64(x[11])
      Am2_link    = np.float64(x[12])
      Am3_link    = np.float64(x[13])
      PCA_link    = np.float64(x[8])
      omegab_link = np.float64(Omb_link*h_link**2.0)
      fmis_link   = np.float64(x[9])
      tau_link    = np.float64(x[10])

      cosmo = astropy.cosmology.FlatLambdaCDM(H0 = h_link*100.0, Om0 = x[15])

      #print("Checkpoint 1")

      corr_factor1 = lensing_corr(redshift1_dat, (zbins_upper1+zbins_lower1)/2.0, cosmo_fid, cosmo, src_dist1, zbins_upper1, zbins_lower1)
      corr_factor2 = lensing_corr(redshift2_dat, (zbins_upper2+zbins_lower2)/2.0, cosmo_fid, cosmo, src_dist2, zbins_upper2, zbins_lower2)
      corr_factor3 = lensing_corr(redshift3_dat, (zbins_upper3+zbins_lower3)/2.0, cosmo_fid, cosmo, src_dist3, zbins_upper3, zbins_lower3)

      #print("Checkpoint 2")
    
      param_vec1cen = np.array([siglogMA, logMminA, logM20A, alphaA, x[8], fixed_n_s, x[14], fixed_w0, fixed_wa, x[15]-x[16], x[16], fixed_alpha_s, fixed_Neff])
      param_vec2cen = np.array([siglogMB, logMminB, logM20B, alphaB, x[8], fixed_n_s, x[14], fixed_w0, fixed_wa, x[15]-x[16], x[16], fixed_alpha_s, fixed_Neff])
      param_vec3cen = np.array([siglogMC, logMminC, logM20C, alphaC, x[8], fixed_n_s, x[14], fixed_w0, fixed_wa, x[15]-x[16], x[16], fixed_alpha_s, fixed_Neff])

      param_vec1mis = np.array([siglogMA, logMminA, logM20A, alphaA, x[8], tau_link, fixed_n_s, x[14], fixed_w0, fixed_wa, x[15]-x[16], x[16], fixed_alpha_s, fixed_Neff])
      param_vec2mis = np.array([siglogMB, logMminB, logM20B, alphaB, x[8], tau_link, fixed_n_s, x[14], fixed_w0, fixed_wa, x[15]-x[16], x[16], fixed_alpha_s, fixed_Neff])
      param_vec3mis = np.array([siglogMC, logMminC, logM20C, alphaC, x[8], tau_link, fixed_n_s, x[14], fixed_w0, fixed_wa, x[15]-x[16], x[16], fixed_alpha_s, fixed_Neff])

      #print("Checkpoint 3")
      
      predicted_lensing1 = Am1_link * corr_factor1 * ( (1. - fmis_link)*emulate_fun1cen(param_vec1cen)[mask1][rich_cut_min:rich_cut_max] + fmis_link*emulate_fun1mis(param_vec1mis)[mask1][rich_cut_min:rich_cut_max] ) / a1**2.0
      predicted_lensing2 = Am2_link * corr_factor2 * ( (1. - fmis_link)*emulate_fun2cen(param_vec2cen)[mask2][rich_cut_min:rich_cut_max] + fmis_link*emulate_fun2mis(param_vec2mis)[mask2][rich_cut_min:rich_cut_max] ) / a2**2.0
      predicted_lensing3 = Am3_link * corr_factor3 * ( (1. - fmis_link)*emulate_fun3cen(param_vec3cen)[mask3][rich_cut_min:rich_cut_max] + fmis_link*emulate_fun3mis(param_vec3mis)[mask3][rich_cut_min:rich_cut_max] ) / a3**2.0

      #print("Checkpoint 4")
      
      diff1 = predicted_lensing1 - obs1      
      chisq1 = np.dot(diff1, np.dot(icov1, diff1))
      
      diff2 = predicted_lensing2 - obs2      
      chisq2 = np.dot(diff2, np.dot(icov2, diff2))

      diff3 = predicted_lensing3 - obs3      
      chisq3 = np.dot(diff3, np.dot(icov3, diff3))

      #print("Checkpoint 5")

      Am_prior1  = compute_gaussian_prior(Am_prior_mean1, Am_prior_sig1, Am1_link)
      Am_prior2  = compute_gaussian_prior(Am_prior_mean2, Am_prior_sig2, Am2_link)
      Am_prior3  = compute_gaussian_prior(Am_prior_mean3, Am_prior_sig3, Am3_link)
      h_prior   = compute_gaussian_prior(np.float64(hubble_prior_mean), np.float64(hubble_prior_sig), h_link)
      omb_prior = compute_gaussian_prior(np.float64(omegab_prior_mean), np.float64(omegab_prior_sig), omegab_link)
      fmis_prior = compute_gaussian_prior(np.float64(fmis_prior_mean), np.float64(fmis_prior_sig), fmis_link)
      tau_prior  = compute_gaussian_prior(np.float64(tau_prior_mean), np.float64(tau_prior_sig), tau_link)

      print(  (- chisq1 / 2.0) + (- chisq2 / 2.0) + (- chisq3 / 2.0) + Am_prior1 + Am_prior2 + Am_prior3 + omb_prior + h_prior + fmis_prior + tau_prior )
      return  (- chisq1 / 2.0) + (- chisq2 / 2.0) + (- chisq3 / 2.0) + Am_prior1 + Am_prior2 + Am_prior3 + omb_prior + h_prior + fmis_prior + tau_prior
    except:
      return -np.inf
          
#################################################################################################################################
#################################################### Initializing Walkers #######################################################
#################################################################################################################################

#X params, 1000 walkers, paper says you want lots of walkers. The variable p0 is an initial point in HOD param space.
ndim, nwalkers = 17, 1000 #+3 to always include Am (lensing calibration uncertainty) 

siglogMA_init = np.linspace(0.25, 0.35, nwalkers)
logMminA_init = np.linspace(12.0, 12.5, nwalkers)
logM20A_init  = np.linspace(14.1, 14.5, nwalkers)
alphaA_init   = np.linspace(1.10, 1.50, nwalkers)
Am1_init     = np.linspace(1.00, 1.04, nwalkers)
Am2_init     = np.linspace(0.99, 1.03, nwalkers)
Am3_init     = np.linspace(0.99, 1.03, nwalkers)
OmM_init     = np.linspace(0.24, 0.30, nwalkers)
Omb_init     = np.linspace(0.04, 0.05, nwalkers)
sig8_init    = np.linspace(0.78, 0.84, nwalkers)
PCA_init     = np.linspace(-0.6, -0.4, nwalkers)
fmis_init    = np.linspace(0.10, 0.20, nwalkers)
tau_init     = np.linspace(0.05, 0.20, nwalkers)
siglogMC_init = np.linspace(0.25, 0.35, nwalkers)
logMminC_init = np.linspace(12.0, 12.5, nwalkers)
logM20C_init  = np.linspace(14.1, 14.5, nwalkers)
alphaC_init   = np.linspace(1.10, 1.50, nwalkers)

np.random.shuffle(siglogMA_init)
np.random.shuffle(logMminA_init)
np.random.shuffle(logM20A_init)
np.random.shuffle(alphaA_init)
np.random.shuffle(Am1_init)
np.random.shuffle(Am2_init)
np.random.shuffle(Am3_init)
np.random.shuffle(OmM_init)
np.random.shuffle(sig8_init)
np.random.shuffle(Omb_init)
np.random.shuffle(PCA_init)
np.random.shuffle(fmis_init)
np.random.shuffle(tau_init)
np.random.shuffle(siglogMC_init)
np.random.shuffle(logMminC_init)
np.random.shuffle(logM20C_init)
np.random.shuffle(alphaC_init)

p0 = np.transpose(np.array([siglogMA_init, logMminA_init, logM20A_init, alphaA_init, siglogMC_init, logMminC_init, logM20C_init, alphaC_init, PCA_init, fmis_init, tau_init, Am1_init, Am2_init, Am3_init, sig8_init, OmM_init, Omb_init]))
x = [0.3, 12.5, 14.5, 1.26, 0.3, 12.5, 14.5, 1.26, -0.5, 0.16, 0.16, 1.03, 1.03, 0.97, 0.81, 0.27, 0.05]

#start = time.time()
#print(lnprob(x))
#end = time.time()
#print(end-start)

#################################################################################################################################
######################################################## Running emcee ##########################################################
#################################################################################################################################


if __name__=='__main__':
  if args.previous_chain:
    infile = h5.File(args.previous_chain, 'r')
    dummy = infile['mcmc/chain']
    dummy_chain = np.ndarray.flatten(np.array(dummy))
    #dummy_chain = dummy_chain[0::ndim]
    dummy_chain = dummy_chain[np.where(dummy_chain!=0)]
    previous_length = int(len(dummy_chain)/nwalkers)
    print("Previous Length: "+str(previous_length))
    infile.close()

  start = time.time()

  #with Pool() as pool:
  with multiprocessing.get_context("spawn").Pool() as pool:
    if args.previous_chain:
      new_backend = emcee.backends.HDFBackend(args.previous_chain)
      new_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=new_backend)
      new_sampler.run_mcmc(None, 150 - previous_length)
    else:
      filename = str(args.backend)
      backend = emcee.backends.HDFBackend(filename, name = 'mcmc')
      backend.reset(nwalkers, ndim)
      sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
      pos, prob, state = sampler.run_mcmc(p0, 20)
      sampler.reset()
      sampler.run_mcmc(pos, nsteps=150, rstate0 = state, log_prob0 = prob)
      
  end = time.time()
  print(end-start)

