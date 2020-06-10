#import logging
## for file logging
"""
logging.basicConfig("test", filename='test.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(threadName)-10s %(message)s',)
"""
import os
import sys
from pathlib import Path
import pickle

rootdir = Path().resolve()
#rootdir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../source')))

import data_make 

import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import solver 
import plot_make 
import log
import logging
#from log import *
logger = logging.getLogger(__name__)



def main():

	## 

	##
	save_folder ="../fig/"
	data_make.make_dir(save_folder)


	#Making images
	x_len, y_len = 256, 256
	dx, dy = 0.01, 0.01
	r_main = 0.03 *1
	width  = 0.02 * 1
	input_model, xx, yy = data_make.ring_make(x_len, y_len, dx, dy, r_main, \
		width, function = data_make.gaussian_function_1d)


	##### Setting observations
	imsize = 256 ##pixel num
	period = 24 ## hrs
	data_num = 30 ## data num
	obs_duration = 2 ##hrs
	n_antena = 8 ## num antena
	s_n = 3 ## signal-to-noise ratio
	wavelength = 1 #mm
	arcsec = 1/206265 
	radius_km = 1.5 ## sphere radius for obs position
	radius_mm = radius_km  * 1000 * 1000 # mm
	baseline_mag = radius_mm * arcsec /wavelength ## baseline u-v dist
	target_pos = [0., 0] ## alpha, beta

	## Vertual Obervatory
	obs_name = "test_observatory"
	obs_file = obs_name + ".pk"
	if not os.path.exists(obs_file):
		obs_ex = data_make.observatory(input_model, data_num,\
		 period, s_n, obs_duration , n_antena, baseline_mag, target_pos , save_folder = save_folder)
		obs_ex.set_antn()
		obs_ex.plotter_uv_sampling()

		with open(obs_file, "wb") as f:
		    pickle.dump(obs_ex, f)

	else:
		with open(obs_file, "rb") as f:
			obs_ex = pickle.load(f) 


	## Making obs data

	vis_obs, num_mat, fft_now = obs_ex.obs_make(dx, dy, s_n)
	vis_test = vis_obs


	## Images from obs
	vis_obs_shift = np.fft.ifftshift(vis_obs)
	beam_func = np.zeros(np.shape(num_mat))
	beam_func[num_mat>0] = 1
	beam_func = np.fft.fftshift(beam_func)
	dirty_image = np.fft.ifft2(vis_obs)
	beam_image = np.fft.fftshift(np.fft.ifft2(beam_func))
	plot_make.image_2dplot(vis_obs_shift,  lim = 127)
	plot_make.image_2dplot(np.abs(beam_image),  lim = 127)



	## Setting priors for model
	model_prior = np.abs(dirty_image)
	images = model_prior



	## L2 regluarzation + positive condition
	stop_ratio = 1e-7
	l2_lambda = 1e0 
	L_init = 1e-4
	eta_init = 1.1
	maxiter = 1000
	miniter = 300
	positive_solve = True


	model_map0, solved_flag0 = solver.only_fx_mfista(images, solver.loss_function_arr_l2, \
		solver.grad_loss_l2, solver.zero_func, eta_init, L_init, maxiter, miniter, positive_solve ,\
		 stop_ratio, vis_obs, model_prior, l2_lambda)
	
	## TSV regluarzation + positive condition
	ltsv_lambda = 1e0
	eta_init =1.1


	model_map1, solved_flag2 = solver.only_fx_mfista(images, solver.loss_function_arr_TSV, \
		solver.grad_loss_tsv, solver.zero_func, eta_init, L_init, maxiter, miniter,positive_solve,\
		 stop_ratio, vis_obs, model_prior, ltsv_lambda)

	## L1 & TSV regluarzation + positive condition
	l2_lambda = 10**2
	l1_lambda = 10**3
	eta_init = 1.05
	positive_solve = False


	model_map3, solved_flag3 = solver.fx_L1_mfista(images, solver.loss_function_arr_TSV, \
		solver.grad_loss_tsv, solver.L1_norm, eta_init, L_init, maxiter, miniter,positive_solve,\
		stop_ratio,  vis_obs, l1_lambda, l2_lambda)


	plot_make.plots_comp(np.abs(beam_image), model_prior, model_map0, model_map1, model_map3, input_model,width_im = 30, fig_size = (10,10), save_folder = save_folder)
	plot_make.plots_vis(model_map0, model_map1,  model_map3, input_model, vis_obs =  vis_obs,save_folder = save_folder)
	#plots_model(model_map0, model_map1, model_map2, model_map3,data, width_im = 40, save_folder = save_folder)


if __name__ == "__main__":
	logger.info("Main part starts")
	main()
