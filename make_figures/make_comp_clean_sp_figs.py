import numpy as np
import os
import shutil
from pathlib import Path
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd

##Data Loading
df_none = pd.read_csv('./vis_out/psim_freq350.alma.out20.csv', header=None)
real_350 = df_none[6]
imag = df_none[7]
uv_dist_350 =( df_none[0]**2 + df_none[1]**2  )**0.5
uv_dist_350_arg = np.argsort(uv_dist_350)
uv_dist_350 = uv_dist_350[uv_dist_350_arg]
real_350 = real_350[uv_dist_350_arg]


df_none = pd.read_csv('./vis_out/psim_freq250.alma.out20.csv', header=None)
real_250 = df_none[6]
imag = df_none[7]
uv_dist_250 =( df_none[0]**2 + df_none[1]**2  )**0.5
uv_dist_250_arg = np.argsort(uv_dist_250)
uv_dist_250 = uv_dist_250[uv_dist_250_arg]
real_250 = real_250[uv_dist_250_arg]


folder_origin = "../imaging_color_sp/tests"
file_origin = os.path.join(folder_origin, "vis_mfreq.pk")
vis_obs, num_mat, fft_now, noise, uv_rr = pd.read_pickle(file_origin)
uv_rr = np.ravel(uv_rr)
arg_uv_dist_sim = np.argsort(uv_rr)
uv_rr = uv_rr[arg_uv_dist_sim]
sim_350 = np.ravel(fft_now[0].real)[arg_uv_dist_sim]
sim_250 = np.ravel(fft_now[1].real)[arg_uv_dist_sim]


## Plotting
plt.plot(uv_rr, sim_250,  color ="g", lw=3, alpha = 0.5, label="Sim 250GHz")
plt.plot(uv_dist_250, real_250, color="k", lw=3, alpha = 0.5,label="CASA 250GHz")
plt.plot(uv_rr, sim_350, color="b", lw=3, alpha = 0.5,label="Sim 350GHz")
plt.plot(uv_dist_350, real_350, color ="r", lw=3, alpha = 0.5,label="CASA 350GHz")
plt.scatter(uv_rr, sim_250,  color ="g", s=3, alpha = 1.0)
plt.scatter(uv_rr, sim_350,  color ="b", s=3, alpha = 1.0)
plt.scatter(uv_dist_250, real_250, color="k", s=3, alpha = 1.0, marker= "*", label="CASA 250GHz")
plt.scatter(uv_dist_350, real_350, color="r", s=3, alpha = 1.0, marker= "*", label="CASA 350GHz")
plt.legend()

plt.xlabel("uv dist $(\lambda)$", fontsize = 20)
plt.ylabel("Visibility (real)", fontsize = 20)
plt.xscale("log")
plt.xlim(2*10**4, 2*10**7)
plt.savefig("./plot_images/vis_uvdist_comp.jpg",  dpi=300, tight_layout=True)
plt.close()