import numpy as np
import matplotlib.pyplot as plt
from data_make import *
from solver import *
from plot_make import *





## Data making
x_len, y_len = 201, 201
images, xx, yy = image_make(x_len, y_len, (-1,1),(-1,1), gaussian_function, (0.01,0.01, 0,0))

## Obs making
obs_num = 200
vis_obs = obs_make(images, obs_num, 30, sn = 5)


## Images from obs
#vis_obs_shift = np.fft.ifftshift(vis_obs)
ifft_obs = np.fft.ifft2(vis_obs)

## Comaparison of images
width_im = 20
sum_flux_obs = np.sum(ifft_obs.real[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im])
sum_flux_ans = np.sum(images.real[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im])
normalization = np.abs(sum_flux_ans/sum_flux_obs)

## Setting priors for model
model_prior = ifft_obs.real#*normalization
l2_lambda = 1e5 
c1 = 1e-6
rho = 0.8
eps_stop = 1e-10
iter_max =300
loss_name = "tsv"

model_prior2 = ifft_obs.real#*normalization

##
gradient = grad_loss_l2(model_prior, vis_obs, model_prior ,l2_lambda)
alpha = 1000 *(np.max(model_prior)/np.min(np.abs(gradient)))

##

model_map = steepest_method(images, alpha, rho, c1, \
	eps_stop, iter_max, "tsv", vis_obs, np.zeros(np.shape(model_prior)),l2_lambda)

print(np.max(model_map),np.sum(model_map))
print(np.max(images),np.sum(images))
print(np.max(model_prior),np.sum(model_prior))

save_folder ="./fig/"
make_dir(save_folder)
plots_comp(model_map, model_prior, images,width_im = 10, fig_size = (10,3), save_folder = save_folder)
plots_vis(model_map, model_prior, images, vis_obs =  vis_obs,save_folder = save_folder)
plots_model(model_map, model_prior, images, width_im = 20, save_folder = save_folder)
