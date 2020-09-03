import numpy as np
import os
import sys
import shutil
from pathlib import Path
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0,'../config')

from setting_freq_common import *
def radial_flag(rr):
	x_len, y_len = np.shape(rr)
	x = np.arange(x_len)
	y = np.arange(y_len)
	xx, yy= np.meshgrid(x, y, indexing='ij')
	dist = ((xx-x_len/2)**2 + (yy-y_len/2)**2 )**0.5
	return dist



def plots_comparison(nu0_images, nu1_images, image0s, alphas, width_im = 10, fig_size = (20,20), save_folder = None, file_name = "image"):

    len_fig_tate = 4
    len_fig_side, x_len, y_len = np.shape(image0s)
    image_for_plots = np.array([nu0_images, nu1_images, image0s, alphas])

    fig, axs = plt.subplots(len_fig_tate, len_fig_side, figsize=fig_size )

    for i_args_div in range(len_fig_tate):
        vmin_input = np.min(image_for_plots[i_args_div][0])
        vmax_input = np.max(image_for_plots[i_args_div][0])

        for i_args_mod in range(len_fig_side):

            show_region = image_for_plots[i_args_div][i_args_mod]
            show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
            pcm = axs[i_args_div, i_args_mod].imshow(image_for_plots[i_args_div][i_args_mod].T, origin = "lower", vmax = vmax_input, vmin = vmin_input)
            axs[i_args_div, i_args_mod].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
            axs[i_args_div, i_args_mod].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)
            #axs[i_args_div, i_args_mod].set_title(ref_yoko[iargs_mod])
            fig.colorbar(pcm, ax=axs[i_args_div, i_args_mod])

    if save_folder !=None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, "%s.png" % file_name), dpi = 100, bbox_inches='tight')
        plt.close()

def plots_radial(nu0_images, nu1_images, image0s, alphas, label_names,  fig_size = (10,10), save_folder = None, file_name = "image"):

	len_fig_tate = 4
	len_fig_side, x_len, y_len = np.shape(image0s)
	image_for_plots = np.array([nu0_images, nu1_images, image0s, alphas])

	x = np.arange(x_len)
	y = np.arange(y_len)
	xx, yy= np.meshgrid(x, y, indexing='ij')
	dist = ((xx-x_len/2)**2 + (yy-y_len/2)**2 )**0.5
	dist = np.ravel(dist)
	arg_dist = np.argsort(dist)
	dist = dist[arg_dist]

	name_png = ["nu0", "nu1", "image0", "alpha"]
	colors = ["r", "g", "b"]


	for i_args_div in range(len_fig_tate):
		for_plots_ims = image_for_plots[i_args_div]

		fig = plt.figure(figsize=fig_size)

		plt.rcParams['font.size'] = 18
		plt.tick_params(labelsize=15)
		plt.xlabel("r (pix)")
		plt.ylabel("Jy/pix^2")
		for j in range(len_fig_side):
			#if j == 1 or j == 3:
			#	continue

			for_plots_now = np.ravel(for_plots_ims[j])
			for_plots_now = for_plots_now[arg_dist]

			total_bins = 100
			bins = np.linspace(dist.min(),dist.max(), total_bins)
			delta = bins[1]-bins[0]
			idx  = np.digitize(dist,bins)
			running_median = [np.median(for_plots_now[idx==k]) for k in range(total_bins)]
			running_std = [np.std(for_plots_now[idx==k]) for k in range(total_bins)]

			if j ==0:
				plt.plot(dist, for_plots_now, label=label_names[j],color = "k", lw = 0.1)
			else:
				#plt.plot(dist, for_plots_now, label=label_names[j],color = colors[j-1], alpha = 0.4)
				plt.errorbar(bins-delta/2+0.1*j, running_median, running_std,color = colors[j-1],fmt='o',label=label_names[j])

		if save_folder !=None:
			if not os.path.exists(save_folder):
				os.makedirs(save_folder)
		plt.legend(fontsize = 18)
		plt.title(name_png[i_args_div])
		plt.savefig(os.path.join(save_folder, "%s_%s.png" % (file_name, name_png[i_args_div])),dpi = 100, bbox_inches='tight')
		plt.close()

def plots_intensity_alpha_comps_color(alphas,  input_alpha, input_model, label_names, save_folder = None, fig_size = (10,10), h_width = 30, file_name = "image"):
	len_fig_side, x_len, y_len = np.shape(alphas)
	dist = radial_flag(input_alpha)
	fig = plt.figure(figsize=fig_size)
	flag_r_width = dist < h_width
	input_alpha_r = np.ravel(input_alpha[flag_r_width])
	input_model_r = np.ravel(input_model[flag_r_width])
	for i in range(len_fig_side):
		alpha_now = np.ravel(alphas[i][flag_r_width])

		plt.scatter(input_model_r, alpha_now - input_alpha_r, s = 5, alpha =0.5 , label="%s" % label_names[i])
	plt.legend()
	plt.xlim(-0.0003, 0.0015)
	plt.savefig(os.path.join(save_folder, "%s.png" % (file_name)),dpi = 100, bbox_inches='tight')
	plt.close()

def plots_intensity_alpha_color(alphas, input_model, label_names, save_folder = None, fig_size = (10,10), h_width = 30, file_name = "image"):
	len_fig_side, x_len, y_len = np.shape(alphas)
	dist = radial_flag(input_alpha)
	fig = plt.figure(figsize=fig_size)
	flag_r_width = dist < h_width
	input_model_r = np.ravel(input_model[flag_r_width])
	for i in range(len_fig_side):
		alpha_now = np.ravel(alphas[i][flag_r_width])
		plt.scatter(input_model_r, alpha_now, s = 5, alpha =0.5 , label="%s" % label_names[i])
	plt.legend()
	plt.xlim(-0.0003, 0.0015)
	plt.savefig(os.path.join(save_folder, "%s.png" % (file_name)),dpi = 100, bbox_inches='tight')
	plt.close()

def plot_alpha_comps(alphas, input_alpha, label_names, save_folder = None, fig_size = (10,10), h_width = 30, file_name = "image"):
	len_fig_side, x_len, y_len = np.shape(alphas)
	dist = radial_flag(input_alpha)
	flag_r_width = dist < h_width	
	fig = plt.figure(figsize=fig_size)
	input_alpha_r = np.ravel(input_alpha[flag_r_width])
	for i in range(len_fig_side):
		alpha_now = np.ravel(alphas[i][flag_r_width])
		plt.scatter(input_alpha_r, alpha_now - input_alpha_r, s = 5, alpha =0.5 , label="%s" % label_names[i])
	plt.legend()
	alpha_min = np.min(input_alpha)-1
	alpha_max = np.max(input_alpha)+1
	#plt.xlim(alpha_min, alpha_max)
	#plt.ylim(alpha_min, alpha_max)
	#plt.plot([alpha_min, alpha_max], [alpha_min, alpha_max])

	plt.savefig(os.path.join(save_folder, "%s.png" % (file_name)),dpi = 100, bbox_inches='tight')
	plt.close()

## INPUT
input_models = np.load(os.path.join(FOLDER_pre,"input_model.npz"))
input_model_nu0 = input_models["model"][0]
input_model_nu1 = input_models["model"][1]
input_image0 = input_models["model_nu0"]
input_alpha= input_models["alpha"]
nu_arr = np.array(input_models["nu_arr"])
nu_0 = float(input_models["nu0"])

## CLEAN
fileimage_taylor0 = os.path.join(FOLDER_clean, "clean_result/try.image.tt0.fits")
hdul = fits.open(fileimage_taylor0)
image_taylor0 = np.array(hdul[0].data[0][0])
header = hdul[0].header
Bmaj = header["BMAJ"]
Bmin = header["BMIN"]
d_pixel = np.abs(header["CDELT1"])
beam_div_pixel2 = np.pi * Bmaj * Bmin/(4 * np.log(2) * d_pixel **2)
image_taylor0 = image_taylor0/beam_div_pixel2
hdul.close()

fileimage_taylor1 = os.path.join(FOLDER_clean, "clean_result/try.image.tt1.fits")
hdul = fits.open(fileimage_taylor1)
image_taylor1 = np.array(hdul[0].data[0][0])
header2 = hdul[0].header
image_taylor1 = image_taylor1/beam_div_pixel2
hdul.close()

fileimage_alpha= os.path.join(FOLDER_clean, "clean_result/try.alpha.fits")
hdul = fits.open(fileimage_alpha)
clean_alpha = np.array(hdul[0].data[0][0])
hdul.close()

clean_model_nu0 = image_taylor0 + image_taylor1 * (nu_arr[0] - nu_0)/nu_0
clean_model_nu1 = image_taylor0 + image_taylor1 * (nu_arr[1] - nu_0)/nu_0
clean_image0 = image_taylor0

## mfreq
image_result_ind = np.load(os.path.join(FOLDER_sparse, "mfreq_ind_solution.npz"))
image_result_mfreq = np.load(os.path.join(FOLDER_sparse, "mfreq_solution.npz"))

## Integration of images
image_nu0s = [input_model_nu0, clean_model_nu0, image_result_ind["image_nu"][0], image_result_mfreq["image_nu"][0]]
image_nu1s = [input_model_nu1, clean_model_nu1, image_result_ind["image_nu"][1], image_result_mfreq["image_nu"][1]]
image0_arrs = [input_image0, clean_image0,image_result_ind["image"],image_result_mfreq["image"] ]
alpha_arrs = [input_alpha, clean_alpha, image_result_ind["alpha"],image_result_mfreq["alpha"] ]

## 
names = ["input", "clean", "sp_ind", "sp_full"]

plots_comparison(image_nu0s, image_nu1s, image0_arrs, alpha_arrs, width_im = 128, save_folder = FOLDER_sparse, file_name = "comp_alls")
plots_radial(image_nu0s, image_nu1s, image0_arrs, alpha_arrs, names, save_folder = FOLDER_sparse, file_name = "comp_alls")
plots_intensity_alpha_color(alpha_arrs, image_nu0s[0], names, h_width = 30, save_folder = FOLDER_sparse, file_name = "int_color")

plots_intensity_alpha_comps_color(alpha_arrs[1:],alpha_arrs[0], image_nu0s[0], names[1:], h_width = 30, save_folder = FOLDER_sparse, file_name = "int_color_comp")
plot_alpha_comps(alpha_arrs[1:], alpha_arrs[0], names[1:], h_width = 30, save_folder = FOLDER_sparse, file_name = "alpha_comp")


