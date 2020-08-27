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

sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'source')))
sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'config')))
import plot_make

input_models = np.load(os.path.join(FOLDER_pre, "input_model.npz"))
input_model = input_models["model"][0]
input_model2 = input_models["model"][1]
model_nu0 = input_models["model_nu0"]
beta_model = input_models["beta"]
nu_arr = np.array(input_models["nu_arr"])
nu_0 = float(input_models["nu0"])

images = [input_model, input_model2, model_nu0, beta_model]
outfolder = "plot_images"
titles = ["image nu[0]","image nu[1]", "input at nu0", "alpha"]
plot_make.plots_parallel(images,titles, width_im = WIDTH_PLOT,\
 save_folder = outfolder, file_name = "input_image")


fileimage_taylor0 = "./clean_result/try.image.tt0.fits"
hdul = fits.open(fileimage_taylor0)
image_taylor0 = np.array(hdul[0].data[0][0])
header = hdul[0].header
Bmaj = header["BMAJ"]
Bmin = header["BMIN"]
d_pixel = np.abs(header["CDELT1"])
beam_div_pixel2 = np.pi * Bmaj * Bmin/(4 * np.log(2) * d_pixel **2)
image_taylor0 = image_taylor0/beam_div_pixel2
hdul.close()

fileimage_taylor1 = "./clean_result/try.image.tt1.fits"
hdul = fits.open(fileimage_taylor1)
image_taylor1 = np.array(hdul[0].data[0][0])
header2 = hdul[0].header
image_taylor1 = image_taylor1/beam_div_pixel2
hdul.close()

fileimage_alpha = "./clean_result/try.alpha.fits"
hdul = fits.open(fileimage_alpha)
image_alpha = np.array(hdul[0].data[0][0])
hdul.close()

image_freq0 = image_taylor0 + image_taylor1 * (nu_arr[0] - nu_0)/nu_0
image_freq1 = image_taylor0 + image_taylor1 * (nu_arr[1] - nu_0)/nu_0



alpha_min = 0
alpha_max = 4
flag_nonfinite = np.isfinite(image_alpha)==False
image_alpha[flag_nonfinite] = alpha_min
image_alpha[image_alpha<alpha_min] = alpha_min
image_alpha[image_alpha>alpha_max] = alpha_max

np.savez("./clean_result/I0_alpha_clean", I0 = image_taylor0, alpha = image_alpha)

images = [image_freq0, image_freq1, image_taylor0, image_alpha]
outfolder = "plot_images"
titles = ["image nu[0]","image nu[1]", "input at nu0", "alpha"]
plot_make.plots_parallel(images,titles, width_im = WIDTH_PLOT,\
 save_folder = outfolder, file_name = "output_image")


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

