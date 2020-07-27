import logging, logging.config
logging.config.fileConfig('../config/logging.conf')
logger = logging.getLogger(__name__)

import os
import sys
from pathlib import Path
import pickle
import importlib

rootdir = Path().resolve()
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../source')))
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../config')))

import setting_freq_ring
from setting_freq_ring import *
import data_make 
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import solver_mfreq as s_freq
import plot_make 


#lambda_arr = np.array([0.8, 1.2])
lambda_arr = np.array([1.0])
lambda0 = 1.0
nu_arr = 1/lambda_arr
nu0 = 1/lambda0

xx, yy = data_make.coordinate_make(XNUM, YNUM, DX, DY)
rr = (xx**2 + yy**2)**0.5

#Making images
image_I0s = data_make.rings_gaps_I0_components(XNUM, YNUM, DX, DY, GAPS_POS,\
                                                GAPS_WIDTH, GAPS_FRAC , MAJ_RINGS)
beta_model, r_arr_beta, beta_arr = data_make.rings_gaps_beta(XNUM, YNUM, DX, DY, GAPS_POS , \
                                                       GAPS_WIDTH, BETA_GAPS_HEIGHT ,  MAJ_RINGS)

print(np.shape(image_I0s))
n_comp, nx, ny = np.shape(image_I0s)
input_models = []
for n_comps in range(n_comp):
    input_models.append(data_make.multi_spectral_data_make(image_I0s[n_comps], beta_model,nu_arr,nu0))
print(np.shape(input_models))
input_model = input_model[:,0,:,:]

save_fig = FIG_FOLDER
fft_models = []
for n_comps in range(n_comp):
    fft_models.append(data_make.fourier_image(input_model[n_comps], DX, DY, lambda_arr, lambda0))

if PLOT_INPUT:
    beta_calc = np.log(image_nu0/image_nu1)/np.log(nu_arr[0]/nu_arr[1])
    flag = (np.isfinite(beta_calc) !=True)
    beta_calc[flag] = 0
    images = [image_nu0, image_nu1, image_origin, beta_model]
    titles = ["image nu0","image nu1", "input at nu0", "beta"]
    plot_make.plots_parallel(images,titles, width_im = WIDTH_PLOT,\
     save_folder = save_fig, file_name = "input_image")

    nx, ny = np.shape(image_nu0)
    ffts = [fft_model[0].real, fft_model[1].real]
    titles = ["vis model", "vis obs"]
    plot_make.plots_parallel(ffts, titles, width_im = int( (nx-1)/2), \
    	save_folder = save_fig, file_name = "vis_input")

    plot_make.plots_vis_radial_model(fft_model, rr,save_fig)

    