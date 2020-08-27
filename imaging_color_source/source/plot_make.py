import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import logging
import sys

logger = logging.getLogger(__name__)


def int_div_for_imshow(int_num):
    div_num = int(np.sqrt(int_num))
    if not div_num**2 == int_num:
        div_num += 1
    div_tate = div_num

    while(1):
        if int_num >  div_num * div_tate:
            return div_num, div_tate + 1

        div_tate = div_tate -1



def plots_parallel(args, title, width_im = 10, fig_size = (10,10), save_folder = None, file_name = "image"):

    args = np.array(args)
    x_len, y_len = np.shape(args[0].T)
    len_model = len(args)
    len_fig_side, len_fig_tate = int_div_for_imshow(len_model)
    
    fig_x, fig_y = fig_size
    fig_size = (fig_x, fig_y * float(len_fig_tate /len_fig_side))
    fig, axs = plt.subplots(len_fig_tate, len_fig_side, figsize=fig_size )

    if len_fig_tate == 1:

        for i in range(len_fig_side):
            show_region = args[i].real
            show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
            pcm = axs[i].imshow(args[i].real.T, origin = "lower", vmax = np.max(show_region), vmin = np.min(show_region))
            axs[i].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
            axs[i].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)
            axs[i].set_title(title[i])
            fig.colorbar(pcm, ax=axs[i])

    else:

        for i_args_div in range(len_fig_tate):
            for i_args_mod in range(len_fig_side):
                i = i_args_mod + i_args_div * len_fig_side
                if i == len(args):
                    break
                show_region = args[i].real
                show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
                pcm = axs[i_args_div, i_args_mod].imshow(args[i].real.T, origin = "lower", vmax = np.max(show_region), vmin = np.min(show_region))
                axs[i_args_div, i_args_mod].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
                axs[i_args_div, i_args_mod].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)
                axs[i_args_div, i_args_mod].set_title(title[i])
                fig.colorbar(pcm, ax=axs[i_args_div, i_args_mod])
    if save_folder !=None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, "%s.png" % file_name), dpi = 100, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plots_comparison(nu0_images, nu1_images, image0s, alphas, width_im = 10, fig_size = (10,10), save_folder = None, file_name = "image"):

    len_fig_tate = 4
    len_fig_side, x_len, y_len = np.shape(image0s)
    image_for_plots = np.array([nu0_images, nu1_images, image0s, alphas])

    fig, axs = plt.subplots(len_fig_tate, len_fig_side, figsize=fig_size )

    for i_args_div in range(len_fig_tate):
        vmin_input = np.min(image_for_plots[i_args_div][0])
        vmax_input = np.min(image_for_plots[i_args_div][0])

        for i_args_mod in range(len_fig_side):

            show_region = image_for_plots[i_args_div][i_args_mod]
            show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
            pcm = axs[i_args_div, i_args_mod].imshow(image_for_plots[i_args_div][i_args_mod].T, origin = "lower", vmax = vmax_input, vmin = vmin_input)
            axs[i_args_div, i_args_mod].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
            axs[i_args_div, i_args_mod].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)
            axs[i_args_div, i_args_mod].set_title(ref_yoko[iargs_mod])
            fig.colorbar(pcm, ax=axs[i_args_div, i_args_mod])

    if save_folder !=None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, "%s.png" % file_name), dpi = 100, bbox_inches='tight')
        plt.close()



def plots_comp(*args, width_im = 10, fig_size = (10,10), save_folder = None, file_name = "image"):


    ## if args is array form
    if len(np.shape(args)) == 4:
        args = args[0]

    x_len, y_len = np.shape(args[0].T)
    len_model = len(args)
    len_fig_side = int_div_for_imshow(len_model)
    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )
    answer = args[len(args)-1]

    for (i, dmy) in enumerate(args):
        i_args_div, i_args_mod = int(i/len_fig_side), i % len_fig_side
        show_region = args[i].real
        show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
        axs[i_args_div, i_args_mod].imshow(args[i].real.T, origin = "lower", vmax = np.max(show_region), vmin = np.min(show_region))
        axs[i_args_div, i_args_mod].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
        axs[i_args_div, i_args_mod].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)

    if save_folder !=None:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        plt.savefig(os.path.join(save_folder, "%s.pdf" % file_name), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    num_center = int(x_len * y_len/2)
    len_fig_side = int_div_for_imshow(len_model)
    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )


    for (i, dmy) in enumerate(args):
        i_args_div, i_args_mod = int(i/len_fig_side), i % len_fig_side
        axs[i_args_div, i_args_mod].plot(args[i].real.T.flatten())
        axs[i_args_div, i_args_mod].set_xlim(num_center - width_im, num_center + width_im)
    if save_folder !=None:
        plt.savefig(os.path.join(save_folder, "zoom_view.pdf"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )
    for (i, dmy) in enumerate(args):
        i_args_div, i_args_mod = int(i/len_fig_side), i % len_fig_side
        axs[i_args_div, i_args_mod].hist(args[i].real.flatten(), bins=50)
        axs[i_args_div, i_args_mod].set_yscale('log')
        rms = np.sqrt(np.mean((answer.real - args[i].real)**2))/np.mean(answer.real)
        logger.info("%d model, rms:%e" % (i, rms))

    if save_folder !=None:
        plt.savefig(os.path.join(save_folder,  "hist_val.pdf"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None

def med_bins(r_arr, d_arr,  r_min, r_max, bin_num):
    r_bins = np.linspace(r_min, r_max, num=bin_num)
    d_r = (r_max - r_min)/(1.0 * bin_num - 1.0)
    mean_bins = []
    std_bins = []
    r_bins_return = []
    
    for i in range(len(r_bins)-1):
        flag =( r_arr > r_bins[i] ) * ( r_arr < r_bins[i+1] ) 
        if len(d_arr[flag]) < 2:
            continue
        mean_bins.append(np.mean(d_arr[flag]))
        std_bins.append(np.std(d_arr[flag].real))
        r_bins_return.append(0.5 * r_bins[i] + 0.5 * r_bins[i+1])

    return np.array(mean_bins), np.array(std_bins), np.array(r_bins_return) 

def plots_vis_radial(vis_obs,vis_model, rr,  save_folder, bins_flag = True):
    n_freq, nx, ny = np.shape(vis_obs)
    rr_arr = np.ravel(rr)

    for i in range(n_freq):
        vis_model_arr = np.ravel(vis_model[i])
        vis_obs_arr = np.ravel(vis_obs[i])
        flag_non_zero = vis_obs_arr != 0 
        vis_obs_arr = vis_obs_arr[flag_non_zero]
        rr_arr_obs = rr_arr[flag_non_zero]

        if bins_flag:
            mean_vis_obs, std_vis_obs, r_bins = med_bins(rr_arr_obs, vis_obs_arr, \
                np.min(rr_arr_obs), np.max(rr_arr_obs), 30)
        
        plt.errorbar(r_bins, mean_vis_obs.real, yerr = std_vis_obs, fmt='o', alpha = 0.5)
        plt.scatter(rr_arr, vis_model_arr.real)
        plt.xlim(0, 0.2 * 1e7)

    plt.savefig(os.path.join(save_folder,  "vis_obs_radial.png"), dpi = 200, bbox_inches='tight')
    plt.close()


def plots_vis_radial_model(vis_model, rr,  save_folder, bins_flag = True):
    n_freq, nx, ny = np.shape(vis_model)
    rr_arr = np.ravel(rr)
    for i in range(n_freq):
        vis_model_arr = np.ravel(vis_model[i])
        mean_vis, std_vis, r_bins = med_bins(rr_arr, vis_model_arr, np.min(rr_arr), np.max(rr_arr), 100)
        #plt.plot(rr_arr, vis_model_arr.real)
        plt.plot(r_bins, mean_vis.real)
    plt.savefig(os.path.join(save_folder,  "vis_obs_radial_model.pdf"), bbox_inches='tight')
    plt.close()



def plots_vis(*args, vis_obs, fig_size = (10,10), save_folder = None):
    vis_obs_conv = np.fft.fftshift(vis_obs)

    ## if args is array form
    if len(np.shape(args)) == 4:
        args = args[0]
    


    nx, ny = np.shape(args[0])
    x = np. arange(nx)
    y = np. arange(ny)
    xx, yy= np.meshgrid(x, y, indexing='ij')
    dist = ((xx-nx/2)**2 + (yy-ny/2)**2 )**0.5
    len_model = len(args)


    ##
    mask_non_zero = (vis_obs_conv !=0)
    vis_obs_plt, dist_obs_plt = vis_obs_conv[mask_non_zero].flatten(),dist[mask_non_zero].flatten()


    model_fft = []
    model_fft_all = []
    for dmy in args:
        model_fft_temp = np.fft.fftshift(np.fft.fft2(dmy))
        model_fft_temp_a = model_fft_temp[mask_non_zero].flatten()
        model_fft.append(model_fft_temp_a)
        model_fft_all.append(model_fft_temp.flatten())


    ##
    len_fig_side = int_div_for_imshow(len_model)
    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )

    for (i, dmy) in enumerate(model_fft):
        i_args_div, i_args_mod = int(i/len_fig_side), i % len_fig_side
        ms_size = 5
        axs[i_args_div, i_args_mod].scatter(dist_obs_plt, vis_obs_plt.real, s =ms_size, color = "k")
        axs[i_args_div, i_args_mod].scatter(dist_obs_plt, model_fft[i].real, s =ms_size, color="r")

    if save_folder !=None:
        plt.savefig(os.path.join(save_folder, "vis.pdf"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    len_fig_side = int_div_for_imshow(len_model)
    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )

    for (i, dmy) in enumerate(model_fft_all):
        i_args_div, i_args_mod = int(i/len_fig_side), i % len_fig_side
        ms_size = 5
        axs[i_args_div, i_args_mod].hist(model_fft_all[i].real, bins = 100, color = "k", log=True)
        
    if save_folder !=None:
        plt.savefig(os.path.join(save_folder,  "vis_hist.pdf"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return None

def image_2dplot(image, lim = 20, zoom_f =0.5, show=False):
    
        fig = plt.figure(figsize=(8.0, 8.0))
        shape = np.shape(image)
        plt.axes().set_aspect('equal')  
        plt.xlim(- zoom_f * lim, zoom_f *lim)
        plt.ylim(-zoom_f *lim, zoom_f *lim)
        plt.imshow(image.real, extent = (-lim,  lim, -lim,  lim))
        if show:
            plt.show()
        else:
            plt.pause(0.1)
            plt.close()


        
def plotter_uv_sampling(pos_arr, save_folder = "./test", save_filename = "uv_plot.pdf"):
    
    n_uv, n_freq, ndata = np.shape(pos_arr)
    fig = plt. figure(figsize=(8.0, 8.0))
    lim_max= np.max(np.abs(pos_arr)/1000.0)*1.3
    plt.xlim(-lim_max, lim_max)
    plt.ylim(-lim_max, lim_max)
    plt.xlabel("u ($k\lambda$)", fontsize = 20)
    plt.ylabel("v ($k\lambda$)", fontsize = 20)
    plt.axes().set_aspect('equal') 
    for i_freq in range(n_freq):     
        plt.scatter(pos_arr[0][i_freq]/1000.0, pos_arr[1][i_freq]/1000.0, s = 1)
    plt.savefig(os.path.join(save_folder, save_filename), bbox_inches='tight', dpi =200)
    plt.close()    


def plots_model(*args, width_im = 10,  save_folder = None):

    len_model = len(args)
    nx, ny = np.shape(args[0])
    x = np. arange(nx)
    y = np. arange(ny)
    xx, yy= np.meshgrid(x, y, indexing='ij')
    dist = ((xx-nx/2)**2 + (yy-ny/2)**2 )**0.5

    len_fig_side = int_div_for_imshow(len_model)
    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )
 
    for (i, dmy) in enumerate(args):
        ms_size = 5
        axs[i_args_div, i_args_mod  ].scatter(dist.flatten(), dmy.real.flatten(), s =ms_size)
        axs[i_args_div, i_args_mod].set_xlim(0, width_im)

    if save_folder !=None:
        plt.savefig(os.path.join(save_folder,  "model_dist.pdf"), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return None
