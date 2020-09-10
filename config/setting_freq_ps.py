import os
import datetime
import numpy as np
from argparse import ArgumentParser

def parser():
    parser = ArgumentParser()
    parser.add_argument('--solve', action='store_true')
    return parser.parse_args()


## 
SOLVE_RUN = False


## Rings & Gaps parameters
ARCSEC_TO_RAD= 1/206265.0
FACTOR = 1
FLUX_MAX_I0 = 5e-4

R_POS = 0.2 * ARCSEC_TO_RAD  * FACTOR
R_WIDTH = 0.05 * ARCSEC_TO_RAD  * FACTOR
THETA_POS = 0.5 * np.pi
THETA_WIDTH = 0.3 * np.pi
R_POS_alpha = 0.2 * ARCSEC_TO_RAD  * FACTOR
R_WIDTH_alpha = 0.07 * ARCSEC_TO_RAD  * FACTOR
THETA_POS_alpha = 1.5 * np.pi
THETA_WIDTH_alpha = 0.6 * np.pi



## Observatory parameters
PERIOD = 8 ## hrs
NDATA= 20 
OBS_DUR = 0.5 ##hrs
N_ANTE = 20 ##num of antennas
SN = 100
LAMBDA_mm= 1#mm
RADIUS_OBS = 1 ##sphere radius /km
RADIUS_OBS_MM = RADIUS_OBS * 1000 * 1000 # mm
BASELINE_UVMAX = RADIUS_OBS_MM  # /LAMBDA_mm ##
 