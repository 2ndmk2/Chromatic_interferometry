import numpy as np
import matplotlib.pyplot as plt
import logging 
import sys
import os
logger = logging.getLogger(__name__)

from pathlib import Path
rootdir = Path().resolve()
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../source')))
sys.path.insert(0, os.path.abspath(os.path.join(rootdir , '../config')))
from setting import *
import solver

import plot_make 
import matplotlib as mpl
if not LOCAL_FLAG:
    mpl.use('Agg')




def l2_rep_solver(init_model, loss_f_arr, grad_f, loss_g = solver.zero_func, eta=1.1, L_init = 1, iter_max = 1000, \
    iter_min = 300, mask_positive = True, stop_ratio = 1e-10, answer = None, dirty =None, beam =None, restart = True, outfolder ="../result/l2_rep", file_log ="log.dat",  *args):
    
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)


    file_log_name = os.path.join(outfolder, file_log)
    f_log = open(file_log_name,"w")

    (vis_obs, model_prior, l2_lambda__arr) = args
    chi2_arr =[]
    reg_arr = []
    models_arr = []
    flag_arr = []
    count = 0
    for l2_lambda in l2_lambda__arr:
        model_map, solve_flag = solver.only_fx_mfista(init_model, loss_f_arr, grad_f, loss_g, eta, \
            L_init, iter_max , iter_min , mask_positive,  stop_ratio , restart,  vis_obs, model_prior, l2_lambda)
        args_now = (vis_obs, model_prior, l2_lambda)
        chi2_now,  reg_now = loss_f_arr(model_map, *args_now)
        chi2_arr.append(chi2_now)
        reg_arr.append(reg_now/l2_lambda)
        models_arr.append(model_map)
        flag_arr.append(solve_flag==SOLVED_FLAG_DONE)
        f_log.write("count:%d, chi2:%e, reg_now:%e, l2:%e" % (count, chi2_now, reg_now, l2_lambda))
        count += 1

    flag_arr,chi2_arr, reg_arr = np.array(flag_arr), np.array(chi2_arr), np.array(reg_arr)
    for i in range(len(models_arr)):
        file_map_name = os.path.join(outfolder, "model_%d.npz" % i)
        np.savez(file_map_name, map = models_arr[i])

    ## Rerenrece from answer
    if answer is not None:
        models_arr.append(answer)

    plot_make.plots_comp(models_arr, width_im = 50, fig_size = (10,10), save_folder =outfolder)
    f_log.close()




