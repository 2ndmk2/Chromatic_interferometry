import os
import datetime

##
LOCAL_FLAG = True
RESTART = True
REPLACE_OBS = True

##
XNUM, YNUM = 64, 64
DX, DY = 0.01, 0.01 ##arcsec


##
RAD_RING = 0.03  ##arcsec
WIDTH_RING  = 0.02   ## arcsec


## 
PERIOD = 24 ## hrs
NDATA= 30 
OBS_DUR = 2 ##hrs
N_ANTE = 20 ##num of antennas
SN = 5
LAMBDA_mm= 1#mm
ARCSEC_TO_RAD= 1/206265
RADIUS_OBS = 2.5 ##sphere radius /km
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


 