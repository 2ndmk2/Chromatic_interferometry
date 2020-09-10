import numpy as np
import os
import shutil
from pathlib import Path
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0,'../config')
from setting_freq_image import *
from setting_freq_common import *
import utility

sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'source')))
sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'config')))
import plot_make

input_models, model_nu0,alpha_model,nu_arr ,nu_0 = utility.load_image_npfile(FOLDER_pre)
images_forplots = np.append(input_models, model_nu0, axis=0)
images_forplots = np.append(images_forplots, alpha_model, axis=0)
outfolder = "plot_images"
titles = utility.title_makes(nu_arr,nu0)
plot_make.plots_parallel(images,titles, width_im = WIDTH_PLOT,\
 save_folder = outfolder, file_name = "input_image")

clean_result = np.load("./clean_result/I0_alpha_clean.npz")
images_freqs_clean = clean_result["image_nu"]
image_for_plots_clean = np.append(images_freqs_clean, clean_result["I0"], axis = 0)
image_for_plots_clean = np.append(image_for_plots_clean, clean_result["alpha"], axis = 0)
plot_make.plots_parallel(image_for_plots_clean,titles, width_im = WIDTH_PLOT,\
 save_folder = outfolder, file_name = "output_image")

"""
##Vis plot
df_none = pd.read_csv('./vis_sim/psim_freq250.alma.out20.noisy.csv', header=None)
real = df_none[6]
imag = df_none[7]
uv_dist =( df_none[0]**2 + df_none[1]**2  )**0.5

df_none2 = pd.read_csv('./vis_sim/psim_freq250.alma.out20.csv', header=None)
real2 = df_none2[6]
imag2 = df_none2[7]
uv_dist2 =( df_none[0]**2 + df_none[1]**2  )**0.5

plt.scatter(uv_dist, real)
plt.scatter(uv_dist, real2)
plt.xlabel("uv dist", fontsize = 20)
plt.ylabel("real Visibility", fontsize = 20)
plt.savefig("./plot_images/vis_uvdist.jpg",  dpi=100)
plt.close()

plt.scatter(uv_dist, real)
plt.scatter(uv_dist, real2)
plt.xlabel("uv dist", fontsize = 20)
plt.ylabel("real Visibility", fontsize = 20)
plt.xlim(0,500000)
plt.savefig("./plot_images/vis_uvdist_zoom.jpg",  dpi=100)
plt.close()

##Vis plot
df_none = pd.read_csv('./vis_sim/psim_freq350.alma.out20.noisy.csv', header=None)
real = df_none[6]
imag = df_none[7]
uv_dist =( df_none[0]**2 + df_none[1]**2  )**0.5

df_none2 = pd.read_csv('./vis_sim/psim_freq350.alma.out20.csv', header=None)
real2 = df_none2[6]
imag2 = df_none2[7]
uv_dist2 =( df_none[0]**2 + df_none[1]**2  )**0.5

plt.scatter(uv_dist, real)
plt.scatter(uv_dist, real2)
plt.xlabel("uv dist", fontsize = 20)
plt.ylabel("real Visibility", fontsize = 20)
plt.savefig("./plot_images/vis_uvdist_350.jpg",  dpi=100)
plt.close()

plt.scatter(uv_dist, real)
plt.scatter(uv_dist, real2)
plt.xlabel("uv dist", fontsize = 20)
plt.ylabel("real Visibility", fontsize = 20)
plt.xlim(0,500000)
plt.savefig("./plot_images/vis_uvdist_zoom_350.jpg",  dpi=100)
plt.close()
"""
