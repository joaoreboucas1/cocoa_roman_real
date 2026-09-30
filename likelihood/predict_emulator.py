#!/usr/bin/env python

import argparse
import configparser
import numpy as np
import scipy.special
import h5py
from math import exp, sqrt
from numba import jit

def get_emulate_fun(input_emu_filename):

	"""return a closure that emulates as a function of input parameters."""

	## read emulator data
	
	f = h5py.File(input_emu_filename, mode='r')

	emu_filename = f['this_filename'][()]
	print(f"emu_filename: {emu_filename}")
	#assert input_emu_filename == emu_filename
	
	sims_dir = f['simulations_dir'][()]
	redshift_dir = f['redshift_dir'][()]
	obs_filename_ext = f['filename_extension_inputs'][()]
	
	param_names = [b.decode('utf-8') for b in f['param_names_inputs']]
	
	gp_kernel_name = f['gp_kernel_name']
	gp_kernel_hyperparameters = f['gp_kernel_hyperparameters']
	gp_kernel_sourcecode = f['gp_kernel_sourcecode']
	
	wp_binmin = f['rbins_min'][:]				# r_p bin edges
	wp_binmax = f['rbins_max'][:]
	
	y_mean = f['mean_training_outputs'][:]		# mean of training data
	y_sigma = f['stdev_training_outputs'][:]	# standard deviation of training data
	
	x = f['raw_training_inputs'][:]		 		# parameters for training data
	X = f['normalized_training_inputs'][:] 		# normalized parameters for training data
	
	coefs = f['best_fit_coefs']					# GP coefficients [ shape: (y.shape[0], X.shape[0]) ]

	ndata = x.shape[0]
	nparams = x.shape[1]
	nbins = y_mean.shape[0]
	
	print(f"nparams = {nparams}")
	print(f"nbins = {nbins}")
	print(f"ndata = {ndata}")
	print(f"param names: {param_names}")


	## define mappings from input data to emulator inputs, from emulator output to 'real' outputs

	xmin_params = np.empty(x.shape[1])
	xmax_params = np.empty(x.shape[1])
	
	for j in range(x.shape[1]):
		xmin_params[j] = x[:,j].min()
		xmax_params[j] = x[:,j].max()
		
	range_params = xmax_params - xmin_params
		
	def normalize_parameters(parameters):
	
		"""Convert HOD+cosmo parameters to emulator inputs in the range [0, 1].
			Should be identical to 'compute_labels' in fit_data.py.
			Applying this function to x should return X."""
		
		return (parameters - xmin_params) / range_params
		
	def denormalize_outputs(emu_output):
	
		"""Convert emulator output into ratio w.r.t analytic prediction."""
		
		return y_sigma*emu_output + y_mean
		
	assert np.allclose( normalize_parameters(x), X )	# test normalize_parameters()	


	## hyperparameters
	
	hyperparams = gp_kernel_hyperparameters[:]
	lambdas = hyperparams[:, 0]
	
	if hyperparams.shape[1] > 1:
		scales = hyperparams[:, 1:]
	else:
		scales = np.zeros(hyperparams.shape[0])
	
	
	## dynamically initialize kernel function (**don't run code from untrusted files!!**)
	
	function_source = gp_kernel_sourcecode[()]
	print(f"{function_source}")

	code_obj = compile(function_source, emu_filename, 'exec')
	code_locals = dict()
	exec(code_obj, globals(), code_locals) # def function into code_locals
	function_name = list(code_locals)[0]

	print(f"Using kernel function: {function_name}")
	kernel_fun = code_locals[function_name]


	## compute model prediction (for each bin j)

	#@jit
	def emulate_fun(input_parameters):

		X_star = normalize_parameters(input_parameters)
		emu_output = np.zeros(nbins)
	
		for j in range(nbins):
			for i in range(coefs.shape[1]):
				emu_output[j] += kernel_fun(X[i,:], X_star, scales[j, :]) * coefs[j,i]

		return denormalize_outputs(emu_output)

	return emulate_fun, wp_binmin, wp_binmax

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('input_emu_filename')
	parser.add_argument('logMmin')
	parser.add_argument('logM20')
	parser.add_argument('alpha')
	parser.add_argument('depth')
	parser.add_argument('redshift')
	parser.add_argument('hubble')
	parser.add_argument('output_lens_file')
	parser.add_argument('--logM1')	
	args = parser.parse_args()
	
	## get emulator function
	
	redshift = float(args.redshift)
	hubble = float(args.hubble)
	a = 1. / (1. + redshift)

	logMmin = float(args.logMmin)
	alpha = float(args.alpha)
	depth = float(args.depth)
	if args.logM1:
		M1 = 10.0**float(args.logM1)
		M0 = 10.0**11.7
		logM20 = np.log10( (20.**(1./alpha)) * M1 + M0 )
		print("Inferred logM20: "+str(logM20))
	else:
		logM20 = float(args.logM20)
	
	emulate_fun, rp_binmin, rp_binmax = get_emulate_fun(args.input_emu_filename)
	input_parameters = np.array([logMmin, logM20, alpha, depth])
	
	rp = (a / hubble)*(rp_binmin + rp_binmax)/2.0
	mask = np.where(rp>0.2)
	
	predicted_rich_bin_lensing = emulate_fun(input_parameters)[mask] / a**2.0

	print(f"Predicted Lensing = {predicted_rich_bin_lensing}")
		
	## output prediction to file
	
	
	np.savetxt(args.output_lens_file, np.c_[rp[mask], predicted_rich_bin_lensing])
