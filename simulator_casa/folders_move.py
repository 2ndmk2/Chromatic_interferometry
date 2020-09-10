import shutil
import os
import sys
sys.path.insert(0,'../config')
from setting_freq_common import *
import glob
from setting_freq_image import *
import utility

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

