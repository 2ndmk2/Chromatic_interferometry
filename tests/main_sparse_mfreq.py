import logging, logging.config
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger(__name__)

import os
import sys
from pathlib import Path
import pickle

rootdir = Path().resolve()
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../source')))
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../config')))

from setting_freq import *

import data_make 
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import solver_mfreq as s_freq
import plot_make 


def main():


	lambda_arr = np.array([0.8, 1.0])
	lambda0 = 1.0
	nu_arr = 1/lambda_arr
	nu0 = 1/lambda0

	#Making images
	beta_func = lambda xx, yy: (0.01)* data_make.gaussian_function_2d(xx, yy, 0.03, 0.03, 0,0)
	input_model, xx, yy = data_make.ring_make_multi_frequency(XNUM, YNUM, DX, DY, \
		RAD_RING, WIDTH_RING, nu_arr, nu0, beta_func, function = data_make.gaussian_function_1d)
	print(np.shape(input_model))

	np.savez('input_model', model = input_model, others = beta_func(xx, yy) )

	#"""
	## Vertual Obervatory
	obs_name = "test_observatory_mfreq"
	obs_file = obs_name + ".pk"
	if not os.path.exists(obs_file) or REPLACE_OBS:
		obs_ex = data_make.observatory_mfreq(input_model, NDATA , PERIOD, SN , OBS_DUR  , N_ANTE ,BASELINE_UVMAX, [0., 0], lambda_arr,lambda0, save_folder = FIG_FOLDER)
		obs_ex.set_antn()
		obs_ex.plotter_uv_sampling()

		with open(obs_file, "wb") as f:
		    pickle.dump(obs_ex, f)

	else:
		with open(obs_file, "rb") as f:
			obs_ex = pickle.load(f) 

		## Making obs data
	vis_obs, num_mat, fft_now = obs_ex.obs_make(DX, DY, SN)
	vis_test = vis_obs

	## Images from obs
	"""
	beam_func = np.zeros(np.shape(num_mat))
	beam_func[num_mat>0] = 1
	beam_func = np.fft.fftshift(beam_func)
	dirty_image = np.fft.ifft2(vis_obs)
	beam_image = np.fft.fftshift(np.fft.ifft2(beam_func))
	plot_make.image_2dplot(np.abs(beam_image),  lim = 127)
	"""


	## Setting priors for model
	N_tot = XNUM * YNUM
	model_init = np.zeros(2 * N_tot)
	model_init[0:N_tot] = 1
	model_init[N_tot:2*N_tot] = 2
	##
	bounds = []
	for i in range(N_tot):
		bounds.append([-1e-3,30])

	for i in range(N_tot):
		bounds.append([-10,10])

	bounds = np.array(bounds)
	f_cost= s_freq.multi_freq_cost_l1_tsv
	df_cost = None#s_freq.multi_freq_grad
	lambda_l1 = 1e0
	lambda_ltsv = 1e0
	result = s_freq.solver_mfreq(f_cost,df_cost, model_init, bounds,  vis_obs, nu_arr, nu0, lambda_l1, lambda_ltsv) 
	image, beta = s_freq.x_to_I_beta(result[0])
	np.savez('test', image=image, beta = beta)

if __name__ == "__main__":
	logger.info("Main part starts")
	main()
