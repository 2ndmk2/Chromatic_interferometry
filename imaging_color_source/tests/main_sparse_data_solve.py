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

import setting_data_ana
from setting_data_ana import *
from setting_freq_common import *
import data_make 
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import solver_mfreq as s_freq
import plot_make 
import data_load

save_fig = FIG_FOLDER


vis_folder = "../../simulator_casa/vis_out"
vis_files = [ os.path.join(vis_folder, "psim_freq350.alma.out20.noisy.csv"), 
os.path.join(vis_folder,"psim_freq250.alma.out20.noisy.csv") ]
freqs_file = os.path.join(vis_folder,"vis_file_freqs.npy")
u_obs, v_obs, vis_obs, nu_arr = data_load.loader_of_visibility_from_csv(vis_files, freqs_file)
nu_arr = np.array([350.00, 250.0])
vis_freq, num_mat_freq, noise_freq = data_load.grided_vis_from_obs(vis_obs, \
    u_obs, v_obs, DX, DY, len(nu_arr), XNUM, YNUM)
nu0 = 300 #/GHz

vis_freq = vis_freq*1000
noise_freq = noise_freq*1000


plot_make.plotter_uv_sampling(np.array([u_obs, v_obs]), FIG_FOLDER, "uv_plot.png")


vis_for_plots = [vis_freq[0].real, vis_freq[1].real]
titles = ["Real Vis 350 GHz", "Real Vis 250 GHz"]
plot_make.plots_parallel(vis_for_plots, titles, width_im = 80, \
	save_folder = FIG_FOLDER, file_name = "vis_obs")


#Making images
xx, yy, uu, vv = data_make.coordinate_make(XNUM, YNUM, DX, DY)

#obs_ex.plotter_uv_sampling()

## Setting priors for model

N_tot = XNUM * YNUM
model_init = np.zeros(2 * N_tot)
model_init[0:N_tot] = 3* np.random.rand(N_tot)  
model_init[N_tot:2*N_tot] =3


## Solver each frequency
## Plot solutions

lambda_l1 = 10**(0.5) 
lambda_ltsv = 10**(0.5) 


image_I0, beta,  model_freqs = s_freq.solver_mfreq_independent(s_freq.loss_function_arr_TSV, s_freq.grad_loss_tsv, s_freq.zero_func, \
                                    vis_freq, noise_freq, nu_arr, nu0, lambda_l1,lambda_ltsv, DX, DY, XNUM, YNUM, beta_def =2.0, \
                                    positive_solve =True)
    
#plot indepenet images 

images = [model_freqs[0], model_freqs[1], image_I0, beta]
titles = ["Estimagenu0 ind", "Estimagenu1 ind", "EstI0 ind", "Estbeta ind"]
plot_make.plots_parallel(images,titles, \
	width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_ind_image")


 ## Solver for chromatic case
reg_para = np.array([-1.0, 1.0, 0.5]) 
lambda_l1 =  10**(reg_para[0])
lambda_ltsv =  10**(reg_para[1])
lambda_beta_ltsv =  10**(reg_para[2])

beta_reg = "TSV"
beta_prior = 0 + np.zeros(np.shape(beta))
model_init = s_freq.edge_zero(model_init, flag_2d =False)


## set bounds for l_bfgs_b minimization
bounds = s_freq.set_bounds(N_tot, beta_max=np.inf , set_beta_zero_at_edge =False)

## setting for l_bfgs_b minimization
#beta = np.ones(np.shape(beta)) * 3
f_cost= s_freq.multi_freq_cost_l1_tsv
df_cost = s_freq.multi_freq_grad

clean_file= "../../simulator_casa/test/I0_alpha_clean.npz"
clean_result = np.load(clean_file)

#model_init = np.append(image_I0, beta)
model_init = np.append(clean_result["I0"], clean_result["alpha"])


## l_bfgs_b minimization
result = s_freq.solver_mfreq(f_cost,df_cost, model_init, bounds,  vis_freq, \
                             noise_freq,nu_arr, nu0, lambda_l1, \
                             lambda_ltsv, lambda_beta_ltsv,beta_reg, beta_prior, DX, DY, maxiter = 200) 
image, beta = s_freq.x_to_I_beta(result[0])
np.savez('test', image = image, beta = beta)

beta[image==0] = 0
images_result = data_make.image_beta_to_images(image, beta, nu_arr, nu0)
images = [images_result[0], images_result[1], image, beta]
titles = ["Est nu0", "Est nu1", "EstI0", "Estbeta"]
plot_make.plots_parallel(images, titles , width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_image")



