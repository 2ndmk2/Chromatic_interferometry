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

from setting import *

import data_make 
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import solver 
import multiple_solver
import plot_make 


def main():

	data_make.make_dir(FIG_FOLDER)

	#Making images
	input_model, xx, yy = data_make.ring_make(XNUM, YNUM, DX, DY, RAD_RING, WIDTH_RING, function = data_make.gaussian_function_1d)

	## Vertual Obervatory
	obs_name = "test_observatory"
	obs_file = obs_name + ".pk"
	if not os.path.exists(obs_file) or REPLACE_OBS:
		obs_ex = data_make.observatory(input_model, NDATA , PERIOD, SN , OBS_DUR  , N_ANTE ,BASELINE_UVMAX, [0., 0], save_folder = FIG_FOLDER)
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
	vis_obs_shift = np.fft.ifftshift(vis_obs)
	beam_func = np.zeros(np.shape(num_mat))
	beam_func[num_mat>0] = 1
	beam_func = np.fft.fftshift(beam_func)
	dirty_image = np.fft.ifft2(vis_obs)
	beam_image = np.fft.fftshift(np.fft.ifft2(beam_func))
	plot_make.image_2dplot(vis_obs_shift,  lim = 127)
	plot_make.image_2dplot(np.abs(beam_image),  lim = 127)



	## Setting priors for model
	model_prior = np.zeros(np.shape(dirty_image))
	images = model_prior



	## L2 regluarzation + positive condition
	l2_lambda = 1e1
	positive_solve = False
	restart_flag = RESTART 


	model_map0, solved_flag0 = solver.only_fx_mfista(images, solver.loss_function_arr_l2, \
		solver.grad_loss_l2, solver.zero_func, ETA_INIT, L_INIT, MAXITE, MINITE, positive_solve ,\
		STOP_RATIO, restart_flag, vis_obs, model_prior, l2_lambda)

	l2_lambda_arr = 10**np.linspace(-1,5,10)
	"""
	multiple_solver.l2_rep_solver(images, solver.loss_function_arr_l2, \
		solver.grad_loss_l2, solver.zero_func, ETA_INIT, L_INIT, MAXITE, MINITE, positive_solve ,\
		STOP_RATIO, input_model,dirty_image, beam_func, RESTART ,"../result/l2_rep","log.dat", vis_obs, model_prior, l2_lambda_arr)
	"""
	
	## TSV regluarzation + positive condition
	ltsv_lambda = 1e3
	eta_init =1.3 
	positive_solve = True



	model_map1, solved_flag2 = solver.only_fx_mfista(images, solver.loss_function_arr_TSV, \
		solver.grad_loss_tsv, solver.zero_func, ETA_INIT, L_INIT, MAXITE, MINITE, positive_solve ,\
		 STOP_RATIO,RESTART,  vis_obs, model_prior, ltsv_lambda)


	## L1 & TSV regluarzation + positive condition
	l2_lambda = 10**2
	l1_lambda = 10**3
	eta_init = 1.3
	positive_solve =True


	model_map3, solved_flag3 = solver.fx_L1_mfista(images, solver.loss_function_arr_TSV, \
		solver.grad_loss_tsv, solver.L1_norm, ETA_INIT, L_INIT, MAXITE, MINITE, positive_solve ,\
		 STOP_RATIO,RESTART, vis_obs, l1_lambda, l2_lambda)


	plot_make.plots_comp(np.abs(beam_image), dirty_image, model_map0, model_map1, model_map3, input_model,\
		width_im = 50, fig_size = (10,10), save_folder = FIG_FOLDER )
	plot_make.plots_vis(model_map0, model_map1,  model_map3, input_model, vis_obs =  vis_obs,save_folder = FIG_FOLDER )


if __name__ == "__main__":
	logger.info("Main part starts")
	main()
