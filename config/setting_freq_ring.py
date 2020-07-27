import os
import datetime
import numpy as np

##
LOCAL_FLAG = True
RESTART = True
REPLACE_OBS = True
GRAD_CONF = False 
PLOT_INPUT = True
PLOT_SOLVE_CURVE = False


## Imaging making
XNUM, YNUM = 256, 256
DX, DY = 0.01, 0.01
GAPS_POS = np.array([0.09, 0.17]) * 1.5
GAPS_WIDTH = np.array([0.015, 0.015]) * 2.0
GAPS_FRAC = [0.7,0.7]
MAJ_RINGS= np.array(0.2) *1.5
BETA_GAPS_HEIGHT  = [1, 1]

##PLOT
WIDTH_PLOT= 50


## 
PERIOD = 8 ## hrs
NDATA= 20 
OBS_DUR = 0.5 ##hrs
N_ANTE = 40 ##num of antennas
SN = 100
LAMBDA_mm= 1#mm
ARCSEC_TO_RAD= 1/206265
RADIUS_OBS = 2.0 ##sphere radius /km
RADIUS_OBS_MM = RADIUS_OBS   * 1000 * 1000 # mm
BASELINE_UVMAX = RADIUS_OBS_MM  * ARCSEC_TO_RAD# /LAMBDA_mm ##


##
STOP_RATIO = 1e-7
MINITE = 50
MAXITE = 100
ETA_INIT = 1.1
L_INIT = 1e-4
MAX_L = 1e15
ETA_MIN= 1.09




##
SOLVED_FLAG_ER = "SOLVE ERROR"
SOLVED_FLAG_DONE = "SOlVED"
FIG_FOLDER = "../fig/"
FIG_IMAGES_FOLDER = "../fig/fig_branch"

if not os.path.exists(FIG_FOLDER):
	os.makedirs(FIG_FOLDER)

if not os.path.exists(FIG_IMAGES_FOLDER):
	os.makedirs(FIG_IMAGES_FOLDER)


 