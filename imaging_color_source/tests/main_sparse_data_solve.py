import os
import sys
import numpy as np


### Preparations
## Logging
import logging, logging.config
logging.config.fileConfig('../../config/logging.conf')
logger = logging.getLogger(__name__)

## Loading config files
sys.path.insert(0,'../../config')
from setting_freq_image import *
from setting_freq_common import *
import utility 

## Setting paths to libraries
sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'source')))
import data_make 
import solver_mfreq as s_freq
import plot_make 
import data_load

## Others (files names)
save_fig = FOLDER_sparse
vis_folder = os.path.join(FOLDER_clean, "vis_sim")
freqs_file = os.path.join(vis_folder,"vis_file_freqs.npz")
nu_arr, nu0 = data_load.freqs_nu_file(freqs_file)




## Main Parts

## data loading & roughly determine regularization parameters 
u_obs, v_obs, vis_obs = data_load.loader_of_visibility_from_csv(vis_folder, nu_arr)
vis_freq, num_mat_freq, noise_freq = data_load.grided_vis_from_obs(vis_obs, \
    u_obs, v_obs, DX, DY, len(nu_arr), XNUM, YNUM)
clean_file= os.path.join(FOLDER_clean, "clean_result/I0_alpha_clean.npz")
clean_result = np.load(clean_file)
lambda_l1_pre, lambda_tsv_pre, lambda_alpha_pre, factor = \
s_freq.determine_regularization_scaling_from_clean(num_mat_freq, \
	clean_result["I0"], clean_result["alpha"])

print(lambda_l1_pre, lambda_tsv_pre, lambda_alpha_pre, factor)

vis_freq = vis_freq*factor
noise_freq = noise_freq*factor

## Makes plots for uv data
plot_make.plotter_uv_sampling(np.array([u_obs, v_obs]), save_fig, "uv_plot.png")
vis_for_plots = [vis_freq[0].real, vis_freq[0].imag]
titles = ["Real Vis", "Imag Vis"]
plot_make.plots_parallel(vis_for_plots, titles, width_im = 80, \
	save_folder = save_fig, file_name = "vis_obs")

#Making images
xx, yy, uu, vv = data_make.coordinate_make(XNUM, YNUM, DX, DY)


## Setting priors for model
N_tot = XNUM * YNUM
model_init = np.zeros(2 * N_tot)
model_init[0:N_tot] = 3* np.random.rand(N_tot)  
model_init[N_tot:2*N_tot] =3


## Solver each frequency
## Plot solutions

lambda_l1 = 10**(-2.0) 
lambda_ltsv = 10**(-0.5) 
print(nu_arr)

image_I0, alpha,  model_freqs = s_freq.solver_mfreq_independent(s_freq.loss_function_arr_TSV, s_freq.grad_loss_tsv, s_freq.zero_func, \
                                    vis_freq, noise_freq, nu_arr, nu0, lambda_l1,lambda_ltsv, DX, DY, XNUM, YNUM, alpha_def =3.0, \
                                    positive_solve =True)
image_I0_def = image_I0/factor
model_freqs_def = model_freqs/factor

#plot indepenet images 
images = np.append(model_freqs_def,[image_I0_def,alpha],  axis = 0)
titles = utility.title_makes(nu_arr, nu0)
plot_make.plots_parallel(images,titles, \
	width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_ind_image")
np.savez(os.path.join(FOLDER_sparse,'mfreq_ind_solution'), image = image_I0_def, alpha = alpha, image_nu = model_freqs_def)


 ## Solver for chromatic case
reg_para = np.array([-2.0, 0.5, -1.0])  
lambda_l1 =  10**(reg_para[0])
lambda_ltsv =  10**(reg_para[1])
lambda_alpha_ltsv =  10**(reg_para[2])

alpha_reg = "TSV"
alpha_prior = 0 + np.zeros(np.shape(alpha))
model_init = s_freq.edge_zero(model_init, flag_2d =False)


## set bounds for l_bfgs_b minimization
bounds = s_freq.set_bounds(N_tot, alpha_max=6 , set_alpha_zero_at_edge =False)

## setting for l_bfgs_b minimization
#alpha = np.ones(np.shape(alpha)) * 3
f_cost= s_freq.multi_freq_cost_l1_tsv
df_cost = s_freq.multi_freq_grad
clean_file= os.path.join(FOLDER_clean, "clean_result/I0_alpha_clean.npz")
clean_result = np.load(clean_file)

#model_init = np.append(image_I0, alpha)
init_I0, init_alpha = s_freq.make_init_models(clean_result["I0"], clean_result["alpha"], 0.00002)
model_init = np.append(init_I0 * factor, init_alpha )


## l_bfgs_b minimization
image, alpha = s_freq.solver_mfreq(model_init, vis_freq, \
                             noise_freq,nu_arr, nu0, lambda_l1, \
                             lambda_ltsv, lambda_alpha_ltsv, DX, DY, maxiter = 500) 
image_def = image/factor

alpha[image_def==0] = 0
images_result = data_make.image_alpha_to_images(image_def, alpha, nu_arr, nu0)
np.savez(os.path.join(FOLDER_sparse, 'mfreq_solution'), image = image_def, alpha = alpha, image_nu = images_result)

images = np.append(images_result,[image_def, alpha] ,  axis = 0)
titles = utility.title_makes(nu_arr, nu0)
plot_make.plots_parallel(images, titles , width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_image")



