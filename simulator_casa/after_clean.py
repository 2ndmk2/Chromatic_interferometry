import numpy as np
import os
import shutil
from pathlib import Path
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import pandas as pd
sys.path.insert(0,'../config')
from setting_freq_image import *
from setting_freq_common import *
import utility

clean_folder ="./clean_result"
clean_name = "try"
nu_arr = NU_OBS
nu0 = NU0


utility.clean_to_images_and_save(clean_folder, clean_name, len(nu_arr), nu_arr, nu0)


sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'source')))
sys.path.insert(0, os.path.abspath(os.path.join(SOURCE_PATH , 'config')))
import plot_make

input_models, model_nu0,alpha_model,nu_arr ,nu_0 = utility.load_image_npfile(FOLDER_pre)
images_forplots = np.append(input_models, [model_nu0,alpha_model], axis=0)
outfolder = "plot_images"
titles = utility.title_makes(nu_arr,nu0)
plot_make.plots_parallel(images_forplots, titles, width_im = WIDTH_PLOT,\
 save_folder = outfolder, file_name = "input_image")

clean_result = np.load("%s/I0_alpha_clean.npz" % clean_folder)
images_freqs_clean = clean_result["image_nu"]
image_for_plots_clean = np.append(images_freqs_clean, [clean_result["I0"],clean_result["alpha"]] , axis = 0)
plot_make.plots_parallel(image_for_plots_clean,titles, width_im = WIDTH_PLOT,\
 save_folder = outfolder, file_name = "output_image")

out_folder ="./vis_sim"
clean_folder ="./clean_result"
folders = utility.make_folder_names(NU_OBS,clean_folder, out_folder)
folder_names = []
for name in folders:
	folder_names.append(name)
folder_names.append("plot_images")
folder_names.append("fits_modified")
utility.mv_folders(folder_names, FOLDER_clean)
logs = glob.glob("*.log")
lasts = glob.glob("*.last")
utility.rm_files(logs)
utility.rm_files(lasts)
