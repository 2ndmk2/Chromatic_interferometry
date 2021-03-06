import os
import sys
from pathlib import Path
import pickle
import importlib
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


import logging, logging.config
logging.config.fileConfig('../../config/logging.conf')
logger = logging.getLogger(__name__)
sys.path.insert(0,'../../config')


from setting_freq_image import *
from setting_freq_common import *
from setting_freq_HD import *
import utility

sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'source')))
import data_make 
import solver_mfreq as s_freq
import plot_make 
import data_load

save_fig = FOLDER_pre


args = parser()
SOLVE_RUN = args.solve

nu_arr = NU_OBS ## Ghz
c_const = 299792458.0 * 1e3/1e9#mm Ghz
nu0 = NU0 #/GHz
lambda_arr = LAMBDA_ARR
lambda0 = LAMBDA0



#Making images
image_origin, r_arr, flux_arr = data_make.HD14_like_I0(XNUM, YNUM, DX, DY, R_POS,\
    R_WIDTH, THETA_POS, THETA_WIDTH, flux_total = FLUX_TOT_I0)
alpha_model, r_arr_alpha, alpha_arr = data_make.HD14_like_alpha(XNUM, YNUM, DX, DY, R_POS_alpha , \
                                                       R_WIDTH_alpha, THETA_POS_alpha , THETA_WIDTH_alpha)
xx, yy, uu, vv = data_make.coordinate_make(XNUM, YNUM, DX, DY)
input_model = data_make.multi_spectral_data_make(image_origin, alpha_model,nu_arr , nu0)
outfile = os.path.join(FOLDER_pre, "input_model") 
np.savez(outfile, model = input_model, model_nu0 = image_origin, alpha = alpha_model, nu_arr = nu_arr, nu0 = nu0 )

uv_rr = (uu*uu + vv*vv)**0.5


outfile = os.path.join(FOLDER_pre, "input_model") 
np.savez(outfile, model = input_model, model_nu0 = image_origin, alpha = alpha_model, nu_arr = nu_arr, nu0 = nu0 )
#"""
## Vertual Obervatory
obs_name = os.path.join(FOLDER_pre, "observatory_mfreq")
obs_file = obs_name + ".pk"
vis_file = os.path.join(FOLDER_pre, "vis_mfreq.pk")


if not os.path.exists(obs_file) or REPLACE_OBS:
    obs_ex = data_make.observatory_mfreq(input_model, NDATA , PERIOD, \
                                         SN , OBS_DUR  , N_ANTE, BASELINE_UVMAX, [0., 0], \
                                         lambda_arr,lambda0, save_folder = save_fig)
    obs_ex.set_antn()
    vis_obs, num_mat, fft_now, noise = obs_ex.obs_make(DX, DY, SN)

    with open(obs_file, "wb") as f:
        pickle.dump(obs_ex, f)
        
        
    with open(vis_file, "wb") as f:
        pickle.dump((vis_obs, num_mat, fft_now, noise, uv_rr), f )

else:
    
    with open(obs_file, "rb") as f:
        obs_ex = pickle.load(f) 
        
    with open(vis_file, "rb") as f:
        vis_obs, num_mat, fft_now, noise, uv_rr = pickle.load(f) 
     
data_make.print_chi_L1_TSV_for_inputmodel(vis_obs, fft_now, noise, image_origin, alpha_model)
obs_ex.plotter_uv_sampling()

## Setting priors for model
N_tot = XNUM * YNUM
model_init = np.zeros(2 * N_tot)
model_init[0:N_tot] = 3* np.random.rand(N_tot)  


model_init[N_tot:2*N_tot] =3


grad_flag = False

data = np.load(outfile + ".npz")
image_nus = data["model"]
image_I0 = data["model_nu0"]
image_alpha = data["alpha"]


if PLOT_INPUT:

    images = np.append(image_nus,[image_I0, image_alpha],  axis = 0)
    titles = utility.title_makes(nu_arr, nu0)
    plot_make.plots_parallel(images,titles, width_im = WIDTH_PLOT,\
     save_folder = save_fig, file_name = "input_image")
    obs_zero_fft = np.copy(fft_now[0].real)
    obs_zero_fft[num_mat[0]==0] = 0 
    images = [fft_now[0].real, fft_now[0].imag]
    titles = ["real", "imag"]
    plot_make.plots_parallel(images, titles, width_im = 80, \
    	save_folder = save_fig, file_name = "vis_input")

    plot_make.plots_vis_radial(vis_obs, fft_now, uv_rr, save_fig)


    

if SOLVE_RUN:
    ## Solver each frequency
    ## Plot solutions
    ave_alpha = np.log(np.sum(image_nu0)/np.sum(image_nu1))/np.log(nu_arr[0]/nu_arr[1])
    lambda_l1 = 10**(1.5)*0.01
    lambda_ltsv = 10**(1.5)*0.01
     

    image_I0, alpha,  model_freqs = s_freq.solver_mfreq_independent(s_freq.loss_function_arr_TSV, s_freq.grad_loss_tsv, s_freq.zero_func, \
                                        vis_obs, noise, nu_arr, nu0, lambda_l1,lambda_ltsv, DX, DY, XNUM, YNUM, alpha_def =2.5,\
                                        eta_init =ETA_INIT, l_init = L_INIT, maxite = MAXITE, minite = MINITE, stop_ratio = STOP_RATIO, \
                                        restart = RESTART, plot_solve_curve=PLOT_SOLVE_CURVE, max_l = MAX_L, eta_min = ETA_MIN)

    #plot indepenet images 

    images = [model_freqs[0], model_freqs[1], image_I0, alpha]
    titles = ["Estimagenu0 ind", "Estimagenu1 ind", "EstI0 ind", "Estalpha ind"]
    plot_make.plots_parallel(images,titles, \
    	width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_ind_image")



     ## Solver for chromatic case
    reg_para = np.array([1.5, 1.5, 1.5]) - 2
    lambda_l1 = 10**(reg_para[0])
    lambda_ltsv = 10**(reg_para[1])
    lambda_alpha_ltsv =  10**(reg_para[2])

    alpha_reg = "TSV"
    alpha_prior = 0 + np.zeros(np.shape(alpha))
    model_init = s_freq.edge_zero(model_init, flag_2d =False)

 


    ## set bounds for l_bfgs_b minimization
    bounds = s_freq.set_bounds(N_tot, alpha_max=9, set_alpha_zero_at_edge =False)

    ## setting for l_bfgs_b minimization


    f_cost= s_freq.multi_freq_cost_l1_tsv
    df_cost = s_freq.multi_freq_grad
    model_init = np.append(image_I0, np.ravel(alpha))

    if GRAD_CONF:
        result = s_freq.grad_mfreq_numerical( model_init,  vis_obs, noise ,nu_arr, nu0, lambda_l1, lambda_ltsv,lambda_alpha_ltsv, DX, DY) 
        grad_2 = s_freq.multi_freq_grad( model_init,  vis_obs, noise ,nu_arr, nu0, lambda_l1, lambda_ltsv, lambda_alpha_ltsv, alpha_reg, alpha_prior, DX, DY) 
        plt.scatter(result[64*64:], grad_2[64*64:])
        plt.show()
        plt.plot(result[64*64:] -grad_2[64*64:])
        plt.show()
        plt.scatter(result[:64*64], grad_2[:64*64])
        plt.show()
        plt.plot(result[:64*64] -grad_2[:64*64])
        plt.show()    

    ## l_bfgs_b minimization
    result = s_freq.solver_mfreq(model_init, vis_obs, \
                                 noise ,nu_arr, nu0, lambda_l1, \
                                 lambda_ltsv, lambda_alpha_ltsv, alpha_reg, alpha_prior, DX, DY, maxiter = 200) 
    image_result, alpha_result = s_freq.x_to_I_alpha(result[0])
    np.savez('test', image = image_result, alpha = alpha_result)


    alpha_result[image_result==0] = 0
    images_result = data_make.image_alpha_to_images(image_result, alpha_result, nu_arr, nu0)
    images = [images_result[0], images_result[1], image_result, alpha_result]
    titles = ["Est nu0", "Est nu1", "EstI0", "Estalpha"]
    plot_make.plots_parallel(images, titles , width_im = WIDTH_PLOT, save_folder = save_fig, file_name = "mfreq_image")

    plt.imshow(alpha_result - alpha)
    plt.colorbar()
    plt.show()

