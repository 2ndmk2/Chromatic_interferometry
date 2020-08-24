import os
import datetime
import numpy as np


## 
SOLVE_RUN = False

## Imaging Params
ARCSEC_TO_RAD= 1/206265
#XNUM, YNUM = 1024,  1024, 
XNUM, YNUM = 256, 256
DX_pix = 0.01
DX, DY = DX_pix*ARCSEC_TO_RAD, DX_pix*ARCSEC_TO_RAD 

## Rings & Gaps parameters
FACTOR = 1
GAPS_POS = np.array([0.09, 0.17]) * 1.5*ARCSEC_TO_RAD  * FACTOR
GAPS_WIDTH = np.array([0.015, 0.015]) * 2.0*ARCSEC_TO_RAD* FACTOR
GAPS_FRAC = [0.7,0.7]
MAJ_RINGS= np.array(0.2) * 1.5 *ARCSEC_TO_RAD* FACTOR
BETA_GAPS_HEIGHT  = [1, 1]


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
 