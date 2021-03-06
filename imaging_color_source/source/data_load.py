import numpy as np
import os
from scipy.stats import binned_statistic_2d
import pandas as pd

def freqs_nu_file(file_freq):
    freqs = np.load(file_freq)
    return freqs["nu_arr"], freqs["nu0"]

def loader_of_visibility_from_csv(vis_folder, nu_arr):

    obs_vis = []
    u_obs = []
    v_obs = []
    vis_obs = []

    for nu_now in nu_arr:
        file = os.path.join(vis_folder, "psim_freq%d.alma.out20.noisy.csv") % int(nu_now)
        df_none = pd.read_csv(file,\
         header=None)
        u = df_none[0]
        v = df_none[1]
        real = df_none[6]
        imag = -df_none[7] ## NOTE THAT CASA gives - imag
        u_obs.append(u)
        v_obs.append(v)
        vis_obs.append(real + 1j * imag)

    return np.array(u_obs), np.array(v_obs), np.array(vis_obs)


def grided_vis_from_obs(vis_obs, u_obs, v_obs, dx, dy, nfreq, x_len, y_len):

    du = 1/(dx * x_len)
    dv = 1/(dx * y_len)
    u = np.arange(0,x_len * du, du)
    v = np.arange(0,y_len * dv, dv)
    u_shift = u - np.mean(u)
    v_shift = v - np.mean(v)
    for_bins_u = np.append(u_shift, np.max(u_shift)+du) - du/2
    for_bins_v = np.append(v_shift, np.max(v_shift)+dv) - dv/2

    vis_freq = np.zeros((nfreq, x_len, y_len), dtype=np.complex)
    noise_freq = np.zeros((nfreq, x_len, y_len), dtype=np.complex)
    num_mat_freq = np.zeros((nfreq, x_len, y_len), dtype=np.complex)
    print(np.max(u), np.max(u_obs))
    print(np.max(v), np.max(v_obs))

    for i_freq in range(nfreq):

        ret = binned_statistic_2d(\
            v_obs[i_freq], u_obs[i_freq], vis_obs[i_freq].real, statistic="mean", \
            bins=(for_bins_u, for_bins_v))
        mean_bins_real = ret.statistic

        ret = binned_statistic_2d(\
            v_obs[i_freq], u_obs[i_freq], vis_obs[i_freq].imag, statistic="mean", \
            bins=(for_bins_u, for_bins_v))
        mean_bins_imag = ret.statistic
        mean_bins = mean_bins_real + 1j * mean_bins_imag

        ret = binned_statistic_2d(\
            v_obs[i_freq], u_obs[i_freq], vis_obs[i_freq].real, statistic="std", \
            bins=(for_bins_u, for_bins_v))
        std_bins_real = ret.statistic
        ret = binned_statistic_2d(\
            v_obs[i_freq],u_obs[i_freq],  vis_obs[i_freq].imag, statistic="std", \
            bins=(for_bins_u, for_bins_v))
        std_bins_imag = ret.statistic
        std_bins = (std_bins_imag ** 2 + std_bins_real**2)**0.5

        ret = binned_statistic_2d(\
            v_obs[i_freq],u_obs[i_freq],  vis_obs[i_freq].real, statistic="count", \
            bins=(for_bins_u, for_bins_v))
        num_mat = ret.statistic

        flag_at_least_two_counts = num_mat < 3
        mean_bins[flag_at_least_two_counts] = 0
        std_bins[flag_at_least_two_counts] = 0
        num_mat[flag_at_least_two_counts] = 0

        vis_freq[i_freq] = mean_bins
        noise_freq[i_freq] = std_bins
        num_mat_freq[i_freq] = num_mat

    return vis_freq, num_mat_freq, noise_freq


