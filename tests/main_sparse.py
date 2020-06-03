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
	data, xx, yy = data_make.ring_make(x_len, y_len, dx, dy, r_main, width, function = data_make.gaussian_function_1d)

	u_max = 0.5/dx
	v_max = 0.5/dy
	du = 1/(dx * x_len)
	dv = 1/(dx * y_len)
	u= np.arange(0,x_len * du, du)
	v = np.arange(0,y_len * dv, dv)
	u_shift = u - np.mean(u)
	v_shift = v - np.mean(v)

	##### Setting observations
	imsize = 256
	period = 24
	data_num = 30
	obs_duration = 2
	n_antena = 12
	s_n = 10
	wavelength = 1#mm
	arcsec = 1/206265
	radius_km = 1.5 ##sphere radius
	radius_mm = radius_km  * 1000 * 1000 # mm
	baseline_mag = radius_mm * arcsec /wavelength ##
	obs_ex = data_make.observatory(data, data_num , period, s_n, obs_duration , n_antena, baseline_mag, [0., 0], save_folder = save_folder)
	obs_ex.set_antn()
	obs_ex.plotter_uv_sampling()


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
	l2_lambda = 1e2 
	images = model_prior
	stop_ratio = 1e-7

	model_map0 = solver.only_fx_mfista(images, solver.loss_function_arr_l2, solver.grad_loss_l2, solver.zero_func, 1.1, 1e-4, 1000, 300, True, stop_ratio, vis_obs, model_prior, l2_lambda)
	#model_map1 = only_fx_mfista(images, loss_function_arr_l2, grad_loss_l2, zero_func, 1.01, 1e-4, 500, True, vis_obs, model_prior, l2_lambda)
	model_map2 = solver.only_fx_mfista(images, solver.loss_function_arr_TSV, solver.grad_loss_tsv, solver.zero_func, 1.1, 1e-4, 1000, 300, True, stop_ratio, vis_obs, model_prior, l2_lambda)
	#model_map0 = model_map2
	model_map1 = model_map2
	##

	l2_lambda = 10**4
	l1_lambda = 10**5
	model_map3 = solver.fx_L1_mfista(images, solver.loss_function_arr_TSV, solver.grad_loss_tsv, solver.L1_norm, 1.05, 1e-4, 1000,300, False,stop_ratio,  vis_obs, l1_lambda, l2_lambda)
	print(len(model_map2[model_map2==0]), len(model_map3[model_map3==0]))


	plot_make.plots_comp(np.abs(beam_image), model_prior, model_map0, model_map1, model_map2, model_map3, data,width_im = 30, fig_size = (10,10), save_folder = save_folder)
	plot_make.plots_vis(model_map0, model_map1, model_map2, model_map3, data, vis_obs =  vis_obs,save_folder = save_folder)
	#plots_model(model_map0, model_map1, model_map2, model_map3,data, width_im = 40, save_folder = save_folder)


if __name__ == "__main__":
	logger.info("Main part starts")
	main()
