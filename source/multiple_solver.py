import numpy as np

import matplotlib.pyplot as plt
import logging 
import sys
import os
logger = logging.getLogger(__name__)

from pathlib import Path
rootdir = Path().resolve()
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../source')))
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../tests')))
from setting import *
import solver

import matplotlib as mpl
if not LOCAL_FLAG:
    mpl.use('Agg')




def chi2_reg_curve(init_model, loss_f_arr, grad_f, loss_g = solver.zero_func, eta=1.1, L_init = 1, iter_max = 1000, iter_min = 300, mask_positive = True, stop_ratio = 1e-10, *args):

    (vis_obs, model_prior, l2_lambda__arr) = args
    chi2_arr =[]
    reg_arr = []
    models_arr = []
    flag_arr = []
    for l2_lambda in l2_lambda__arr:
        model_map, solve_flag = solver.only_fx_mfista(init_model, loss_f_arr, grad_f, loss_g, eta, \
            L_init, iter_max , iter_min , mask_positive,  stop_ratio ,  vis_obs, model_prior, l2_lambda)
        args_now = (vis_obs, model_prior, l2_lambda)
        chi2_now,  reg_now = loss_f_arr(model_map, *args_now)
        chi2_arr.append(chi2_now)
        reg_arr.append(reg_now/l2_lambda)
        models_arr.append(model_map)
        flag_arr.append(solve_flag==SOLVED_FLAG_DONE)

    flag_arr,chi2_arr, reg_arr = np.array(flag_arr), np.array(chi2_arr), np.array(reg_arr)
    plt.plot(np.log10(chi2_arr[flag_arr]), np.log10(reg_arr[flag_arr]))
    plt.savefig(FIG_FOLDER + "curve.pdf", bbox_inches='tight')
    plt.close()
    print(np.shape(models_arr))
    return np.array(models_arr)

