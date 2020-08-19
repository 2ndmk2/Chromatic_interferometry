import os
import datetime
import numpy as np
from argparse import ArgumentParser

def parser():
    parser = ArgumentParser()
    parser.add_argument('--solve', action='store_true')
    return parser.parse_args()



##
LOCAL_FLAG = True
RESTART = True
REPLACE_OBS = True
GRAD_CONF = False
PLOT_INPUT = True
PLOT_SOLVE_CURVE = False


##PLOT
WIDTH_PLOT= 80

##
STOP_RATIO = 1e-7
MINITE = 50
MAXITE = 100
ETA_INIT = 1.1
L_INIT = 1e-3
MAX_L = 1e15
ETA_MIN= 1.05

##
SOLVED_FLAG_ER = "SOLVE ERROR"
SOLVED_FLAG_DONE = "SOlVED"
FIG_FOLDER = "../fig/"

if not os.path.exists(FIG_FOLDER):
	os.makedirs(FIG_FOLDER) 