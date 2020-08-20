import shutil
import os
import sys
sys.path.insert(0,'../config')
from setting_freq_common import *
import glob


def mv_folders(folder_names, move_to_place):
	for name in folder_names:
		path_new = os.path.join(move_to_place, name)
		if os.path.exists(path_new):
			shutil.rmtree(path_new)

		shutil.move(name, path_new)

def remove_files(names):
	for name in names:
		os.remove(name)

out_folder ="./vis_sim"
clean_folder ="./clean_result"
folder_names =[clean_folder, "psim_freq350", "psim_freq250", out_folder, "plot_images", "fits_modified"]
mv_folders(folder_names, ROOT_FOLDER)



logs = glob.glob("*.log")
lasts = glob.glob("*.last")

remove_files(logs)
remove_files(lasts)

