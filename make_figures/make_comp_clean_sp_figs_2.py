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

def make_titles(nu_arr, nu0):
	titles = []
	for nu_dmy in nu_arr:
		titles.append(str(nu_dmy) +"GHz")
	titles.append(str(nu0)+"GHz")
	titles.append("alpha")
	return titles

def plots_comparison_sum(image_for_plots, titles = None, models_titles = None, width_im = 10, fig_size = (20,20), save_folder = None, file_name = "image"):

    len_fig_side,len_fig_tate, x_len, y_len = np.shape(image_for_plots)

    fig, axs = plt.subplots(len_fig_side, len_fig_tate, figsize=fig_size)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    for i_args_mod in range(len_fig_tate):
        vmin_input = np.min(image_for_plots[0][i_args_mod])
        vmax_input = np.max(image_for_plots[0][i_args_mod])
        if i_args_mod == 4:
        	vmin_input -= 1
        	vmax_input += 1

        for i_args_div in range(len_fig_side):

            show_region = image_for_plots[i_args_div][i_args_mod]
            show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
            pcm = axs[i_args_div][i_args_mod].imshow(image_for_plots[i_args_div][i_args_mod], origin = "lower", vmax = vmax_input, vmin = vmin_input)
            axs[i_args_div, i_args_mod].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
            axs[i_args_div, i_args_mod].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)
            axs[i_args_div, i_args_mod].set_title(titles[i_args_div] + ":" + models_titles[i_args_mod])
            fig.colorbar(pcm, ax=axs[i_args_div][i_args_mod])

    if save_folder !=None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, "%s.png" % file_name), dpi = 100, bbox_inches='tight')
        plt.close()


def plots_comparison(nu0_images, nu1_images, image0s, alphas, width_im = 10, fig_size = (20,20), save_folder = None, file_name = "image"):

    len_fig_tate = 4
    len_fig_side, x_len, y_len = np.shape(image0s)
    image_for_plots = np.array([nu0_images, nu1_images, image0s, alphas])

    fig, axs = plt.subplots(len_fig_tate, len_fig_side, figsize=fig_size )

    for i_args_div in range(len_fig_tate):
        vmin_input = np.min(image_for_plots[i_args_div][0])-1
        vmax_input = np.max(image_for_plots[i_args_div][0])+1

        for i_args_mod in range(len_fig_side):

            show_region = image_for_plots[i_args_div][i_args_mod]
            show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
            pcm = axs[i_args_div, i_args_mod].imshow(image_for_plots[i_args_div][i_args_mod], origin = "lower", vmax = vmax_input, vmin = vmin_input)
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
input_model_nu = input_models["model"]
input_image0 = input_models["model_nu0"]
input_alpha= input_models["alpha"]
nu_arr = np.array(input_models["nu_arr"])
nu_0 = float(input_models["nu0"])
input_model_sum = np.append(input_model_nu,[input_image0, input_alpha], axis = 0)


## clean
clean_file= os.path.join(FOLDER_clean, "clean_result/I0_alpha_clean.npz")
clean_result = np.load(clean_file)
clean_model_nu = clean_result["image_nu"]
clean_image0 = clean_result["I0"]
clean_alpha = clean_result["alpha"]
clean_sum = np.append(clean_model_nu,[clean_image0, clean_alpha], axis = 0)



## mfreq
image_result_ind = np.load(os.path.join(FOLDER_sparse, "mfreq_ind_solution.npz"))
ind_sum = np.append(image_result_ind["image_nu"], [image_result_ind["image"], image_result_ind["alpha"]], axis = 0)
image_result_mfreq = np.load(os.path.join(FOLDER_sparse, "mfreq_solution.npz"))
mfreq_sum = np.append(image_result_mfreq["image_nu"], [image_result_mfreq["image"], image_result_mfreq["alpha"]], axis = 0)
analyze_sum = np.array([input_model_sum, clean_sum, ind_sum, mfreq_sum])


## 
names = ["input", "clean", "sp_ind", "sp_full"]
names_data =make_titles(nu_arr, nu_0)

plots_comparison_sum(analyze_sum,titles = names, models_titles = names_data, width_im = 50, save_folder = FOLDER_sparse, file_name = "comp_alls")

