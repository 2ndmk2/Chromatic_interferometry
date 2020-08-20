import os
import datetime

##
LOCAL_FLAG = True
RESTART = True
REPLACE_OBS = True
REP_L2_FLAG = False

##
XNUM, YNUM = 256, 256
DX, DY = 0.01, 0.01 ##arcsec


##
RAD_RING = 0.03  ##arcsec
WIDTH_RING  = 0.02   ## arcsec


## 
PERIOD = 24 ## hrs
NDATA= 30 
OBS_DUR = 2 ##hrs
N_ANTE = 10 ##num of antennas
SN = 5
LAMBDA_mm= 1#mm
ARCSEC_TO_RAD= 1/206265
RADIUS_OBS = 2.5 ##sphere radius /km
RADIUS_OBS_MM = RADIUS_OBS   * 1000 * 1000 # mm
BASELINE_UVMAX = RADIUS_OBS_MM  * ARCSEC_TO_RAD /LAMBDA_mm ##


##
STOP_RATIO = 1e-7
MINITE = 300
MAXITE = 500
ETA_INIT = 1.1
L_INIT = 1e2
MAX_L = 1e15
ETA_MIN= 1.09




##
SOLVED_FLAG_ER = "SOLVE ERROR"
SOLVED_FLAG_DONE = "SOlVED"
FIG_FOLDER = "../fig/"


 