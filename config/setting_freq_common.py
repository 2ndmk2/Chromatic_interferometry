import os
import datetime
import numpy as np

SOURCE_PATH = "/Users/masatakaaizawa/research/current_research/\
chromatic_imaging/chiim_source/imaging_color_source/"
ROOT_FOLDER = "/Users/masatakaaizawa/research/current_research/chromatic_imaging/"
#PROJECT_NAME = "project_512_HD_3freqs"
#PROJECT_NAME = "project_256_HD_3freqs"
PROJECT_NAME = "project_test"

RESULT_FOLDER = os.path.join(ROOT_FOLDER,PROJECT_NAME)
FOLDER_pre =  RESULT_FOLDER+ "/for_input/"
FOLDER_sparse = RESULT_FOLDER + "/sparse_result/"
FOLDER_clean = RESULT_FOLDER + "/clean_result_and_data/"

if not os.path.exists(FOLDER_pre):
	os.makedirs(FOLDER_pre)
if not os.path.exists(FOLDER_sparse):
	os.makedirs(FOLDER_sparse)
if not os.path.exists(FOLDER_clean):
	os.makedirs(FOLDER_clean)

## flag
LOCAL_FLAG = True
RESTART = True
REPLACE_OBS = True
GRAD_CONF = False
PLOT_INPUT = True
PLOT_SOLVE_CURVE = False

##PLOT
WIDTH_PLOT= 128

## For MFISTA
STOP_RATIO = 1e-7
MINITE = 300
MAXITE = 500
ETA_INIT = 1.1
L_INIT = 1e-3
MAX_L = 1e15
ETA_MIN= 1.05

##
SOLVED_FLAG_ER = "SOLVE ERROR"
SOLVED_FLAG_DONE = "SOlVED"


