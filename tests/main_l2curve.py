
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
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../setting')))
from setting import *

import data_make 

import matplotlib as mpl
if not LOCAL_FLAG:
    mpl.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import multiple_solver
import solver
import plot_make 
#import log
import logging
#from log import *
logger = logging.getLogger(__name__)


def main():



	data_make.make_dir(FIG_FOLDER)


	#Making images

	input_model, xx, yy = data_make.ring_make(XNUM, YNUM, DX, DY, RAD_RING, WIDTH_RING, function = data_make.gaussian_function_1d)


	##### Setting observations
	obs_ex = data_make.observatory(input_model, NDATA , PERIOD, SN , OBS_DUR  , N_ANTE ,BASELINE_UVMAX, [0., 0], save_folder = FIG_FOLDER)
	obs_ex.set_antn()
	obs_ex.plotter_uv_sampling()


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
	model_prior = np.abs(dirty_image)
	l2_lambda = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5] 
	images_prior = model_prior

	models_arr = multiple_solver.chi2_reg_curve(images_prior, solver.loss_function_arr_l2, solver.grad_loss_l2, solver.zero_func, ETA_INIT, L_INIT , MAXITE , MINITE , True, stop_ratio, vis_obs, model_prior, l2_lambda)
	models_for_plot = np.append(models_arr, [np.abs(model_prior), np.abs(beam_image),input_model ], axis=0)
	plot_make.plots_comp(models_for_plot, width_im = 50, fig_size = (10,10), save_folder =FIG_IMAGES_FOLDER)
	models_for_plot = np.append(models_arr, [input_model], axis=0)
	plot_make.plots_vis(models_for_plot, vis_obs =  vis_obs,save_folder =FIG_IMAGES_FOLDER)

if __name__ == "__main__":
	logger.info("Main part starts")
	main()
