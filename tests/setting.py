import os
from logging import basicConfig, getLogger, StreamHandler, FileHandler, Formatter, INFO
import datetime

##
LOCAL_FLAG = True

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
RADIUS_OBS = 1.5 ##sphere radius /km
RADIUS_OBS_MM = RADIUS_OBS   * 1000 * 1000 # mm
BASELINE_UVMAX = RADIUS_OBS_MM  * ARCSEC_TO_RAD /LAMBDA_mm ##


##
stop_ratio = 1e-7
MINITE = 300
MAXITE = 400
ETA_INIT = 1.1
L_INIT = 1e-4
MAX_L = 1e15
ETA_MIN= 1.02



##
SOLVED_FLAG_ER = "SOLVE ERROR"
SOLVED_FLAG_DONE = "SOlVED"
FIG_FOLDER = "../fig/"
FIG_IMAGES_FOLDER = "../fig/fig_branch"

if not os.path.exists(FIG_FOLDER):
	os.makedirs(FIG_FOLDER)

if not os.path.exists(FIG_IMAGES_FOLDER):
	os.makedirs(FIG_IMAGES_FOLDER)



now = datetime.datetime.now()
logger_name = "test"
log_file = "../log/" + now.strftime('%Y%m%d_%H%M%S')  + ".log"

f_fmt='%(asctime)s - %(levelname)s - %(funcName)s- %(message)s'

basicConfig(
    filename=log_file,
    filemode='w', # Default is 'a'
    format=f_fmt, 
    level="DEBUG")


# define a new Handler to log to console as well
console = StreamHandler()
# optional, set the logging level
console.setLevel("DEBUG")
# set a format which is the same for console use
formatter = Formatter('%(asctime)s - %(levelname)s - %(funcName)s- %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
getLogger('').addHandler(console)
 