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


nu_arr = np.array([350, 250]) ## Ghz
c_const = 299792458.0 * 1e3/1e9#mm Ghz
nu0 = 300 #/GHz
lambda_arr = c_const/nu_arr
lambda0 = c_const/nu0



#Making images
image_origin, r_arr, flux_arr = data_make.rings_gaps_I0(XNUM, YNUM, DX, DY, GAPS_POS,\
                                                GAPS_WIDTH, GAPS_FRAC , MAJ_RINGS)
beta_model, r_arr_beta, beta_arr = data_make.rings_gaps_beta(XNUM, YNUM, DX, DY, GAPS_POS , \
                                                       GAPS_WIDTH, BETA_GAPS_HEIGHT ,  MAJ_RINGS)
xx, yy = data_make.coordinate_make(XNUM, YNUM, DX, DY)
input_model = data_make.multi_spectral_data_make(image_origin, beta_model,nu_arr , nu0)
rr = (xx**2 + yy**2)**0.5

outfile = "input_model"
np.savez(outfile, model = input_model, others = beta_model )
#"""
## Vertual Obervatory
obs_name = "test_observatory_mfreq"
obs_file = obs_name + ".pk"

vis_file = "vis_mfreq" + ".pk"

if not os.path.exists(obs_file) or REPLACE_OBS:
    obs_ex = data_make.observatory_mfreq(input_model, NDATA , PERIOD, \
                                         SN , OBS_DUR  , N_ANTE, BASELINE_UVMAX, [0., 0], \
                                         lambda_arr,lambda0, save_folder = FIG_FOLDER)
    obs_ex.set_antn()
    vis_obs, num_mat, fft_now, noise = obs_ex.obs_make(DX, DY, SN)

    with open(obs_file, "wb") as f:
        pickle.dump(obs_ex, f)
        
        
    with open(vis_file, "wb") as f:
        pickle.dump((vis_obs, num_mat, fft_now, noise ), f )

else:
    
    with open(obs_file, "rb") as f:
        obs_ex = pickle.load(f) 
        
    with open(vis_file, "rb") as f:
        vis_obs, num_mat, fft_now, noise = pickle.load(f) 
     
print(np.shape(vis_obs), np.shape(noise), np.shape(image_origin))

obs_ex.plotter_uv_sampling()

## Setting priors for model
N_tot = XNUM * YNUM
model_init = np.zeros(2 * N_tot)
model_init[0:N_tot] = 3* np.random.rand(N_tot)  


model_init[N_tot:2*N_tot] =3


grad_flag = False


data = np.load("./%s.npz" % outfile)
image_nu0 = data["model"][0]
image_nu1 = data["model"][1]
save_fig = FIG_FOLDER

if PLOT_INPUT:
    beta_calc = np.log(image_nu0/image_nu1)/np.log(nu_arr[0]/nu_arr[1])
    flag = (np.isfinite(beta_calc) !=True)
    beta_calc[flag] = 0
    images = [image_nu0, image_nu1, image_origin, beta_model]
    titles = ["image nu0","image nu1", "input at nu0", "beta"]
    plot_make.plots_parallel(images,titles, width_im = WIDTH_PLOT,\
     save_folder = save_fig, file_name = "input_image")

    nx, ny = np.shape(image_nu0)
    obs_zero_fft = np.copy(fft_now[0].real)
    obs_zero_fft[num_mat[0]==0] = 0 
    images = [fft_now[0].real, obs_zero_fft]
    titles = ["vis model", "vis obs"]
    plot_make.plots_parallel(images, titles, width_im = 80, \
    	save_folder = save_fig, file_name = "vis_input")


    plot_make.plots_vis_radial(vis_obs, fft_now, rr,save_fig)

    
if GRAD_CONF:
    grad_num = s_freq.grad_mfreq_numerical(model_init, vis_obs, nu_arr, nu0, lambda_l1, lambda_ltsv)
    grad_ana = s_freq.multi_freq_grad(model_init, vis_obs, nu_arr, nu0, lambda_l1, lambda_ltsv)
    plt.scatter(grad_ana, grad_num- grad_ana)
    plt.show()
    
ave_beta = np.log(np.sum(image_nu0)/np.sum(image_nu1))/np.log(nu_arr[0]/nu_arr[1])

## Solver each frequency
## Plot solutions

lambda_l1 = 10**(-0.5)
lambda_ltsv = 10**(-0.5)
 

image_I0, beta,  model_freqs = s_freq.solver_mfreq_independent(s_freq.loss_function_arr_TSV, s_freq.grad_loss_tsv, s_freq.zero_func, \
                                    vis_obs, noise, nu_arr, nu0, lambda_l1,lambda_ltsv, beta_def =1.5)
    
#plot indepenet images 

images = [model_freqs[0], model_freqs[1], image_I0, beta]
titles = ["Estimagenu0 ind", "Estimagenu1 ind", "EstI0 ind", "Estbeta ind"]
plot_make.plots_parallel(images,titles, \
	width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_ind_image")


## Grad Check for monochromatic case


if GRAD_CONF:
    input_model = 0.5 * model_freqs[0] + np.zeros(np.shape(model_freqs[0]))
    grad_num = s_freq.grad_loss_numerical_TSV(input_model, vis_obs[0], noise[0], lambda_l1, lambda_ltsv)
    grad_ana_chi, grad_ana_tsv = s_freq.grad_loss_arr_TSV(input_model , vis_obs[0], noise[0], lambda_l1, lambda_ltsv)
    grad_ana = grad_ana_chi + grad_ana_tsv
    plt.scatter(grad_num, grad_ana)

## Grad Check for chromatic case

if GRAD_CONF:
    result = s_freq.grad_mfreq_numerical( model_init,  vis_obs, noise ,nu_arr, nu0, lambda_l1, lambda_ltsv) 
    grad_2 = s_freq.multi_freq_grad( model_init,  vis_obs, noise ,nu_arr, nu0, lambda_l1, lambda_ltsv) 
    plt.scatter(result, grad_2)

 ## Solver for chromatic case
reg_para = np.array([-1.0, -0.5, -0.0])
lambda_l1 = 10**(reg_para[0])
lambda_ltsv = 10**(reg_para[1])
lambda_beta_ltsv =  10**(reg_para[2])

beta_reg = "TSV"
beta_prior = 0 + np.zeros(np.shape(beta))
#beta = s_freq.edge_zero(beta)
model_init = s_freq.edge_zero(model_init, flag_2d =False)


## set bounds for l_bfgs_b minimization

bounds = s_freq.set_bounds(N_tot, beta_max=6, set_beta_zero_at_edge =False)
print(np.shape(bounds), N_tot)

## setting for l_bfgs_b minimization

f_cost= s_freq.multi_freq_cost_l1_tsv
df_cost = s_freq.multi_freq_grad
models_init = np.append(image_I0, np.ravel(beta))
# models_init = s_freq.x_to_I_beta([image_origin, beta_calc], reverse = True)

## l_bfgs_b minimization
result = s_freq.solver_mfreq(f_cost,df_cost, model_init, bounds,  vis_obs, \
                             noise ,nu_arr, nu0, lambda_l1, \
                             lambda_ltsv, lambda_beta_ltsv,beta_reg, beta_prior, maxiter = 200) 
image, beta = s_freq.x_to_I_beta(result[0])
np.savez('test', image = image, beta = beta)


beta[image==0] = 0
images_result = data_make.image_beta_to_images(image, beta, nu_arr, nu0)
images = [images_result[0], images_result[1], image, beta]
titles = ["Est nu0", "Est nu1", "EstI0", "Estbeta"]
plot_make.plots_parallel(images, titles , width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_image")



