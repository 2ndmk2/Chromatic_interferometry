import numpy as np
import matplotlib.pyplot as plt
from data_make import *
from solver import *
from plot_make import *


## Data making
x_len, y_len = 201, 201
#images, xx, yy = gauss_make(x_len, y_len, (-1,1),(-1,1), gaussian_function, (0.01,0.01, 0,0))
images, xx, yy = ring_make(x_len, y_len, (-1,1),(-1,1),.04, .01,  gaussian_function_1d)

## Obs making
obs_num = 40
vis_obs = obs_make(images, obs_num, 20, sn = .1)


## Images from obs
#vis_obs_shift = np.fft.ifftshift(vis_obs)
ifft_obs = np.fft.ifft2(vis_obs)

## Comaparison of images
width_im = 20
sum_flux_obs = np.sum(ifft_obs.real[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im])
sum_flux_ans = np.sum(images.real[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im])
normalization = np.abs(sum_flux_ans/sum_flux_obs)



## Setting priors for model
model_prior = ifft_obs.real
l2_lambda = 1e0
c1 = 1e-4
rho = 0.80
eps_stop = 1e-10
iter_max =50
loss_name = "tsv"
model_prior2 = ifft_obs.real



##
model_map0 = only_fx_mfista(images, loss_function_l2, grad_loss_l2, zero_func, 1.1, 1, 500, False, vis_obs, model_prior, l2_lambda)
model_map1 = only_fx_mfista(images, loss_function_l2, grad_loss_l2, zero_func, 1.1, 1, 500, True, vis_obs, model_prior, l2_lambda)
model_map2 = only_fx_mfista(images, loss_function_TSV, grad_loss_tsv, zero_func, 1.1, 1, 500, True, vis_obs, model_prior, l2_lambda)

##

l2_lambda = 10**0
l1_lambda = 10**1.5
model_map3 = fx_L1_mfista(images, loss_function_TSV, grad_loss_tsv, L1_norm, 1.05, 1, 500,True,  vis_obs, l1_lambda, l2_lambda)
print(len(model_map2[model_map2==0]), len(model_map3[model_map3==0]))


save_folder ="./fig/"
make_dir(save_folder)
plots_comp(model_prior2, model_map0, model_map1, model_map2, model_map3, images,width_im = 40, fig_size = (10,3), save_folder = save_folder)
plots_vis(model_map0, model_map1, model_map2, model_map3, images, vis_obs =  vis_obs,save_folder = save_folder)
plots_model(model_map0, model_map1, model_map2, model_map3,images, width_im = 40, save_folder = save_folder)
