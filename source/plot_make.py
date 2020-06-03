import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os

def int_div_for_imshow(int_num):
    div_num = int(np.sqrt(int_num))
    return div_num+1



def plots_comp(*args, width_im = 10, fig_size = (10,10), save_folder = None):

    x_len, y_len = np.shape(args[0].T)
    len_model = len(args)
    len_fig_side = int_div_for_imshow(len_model)
    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )
    answer = args[len(args)-1]

    for (i, dmy) in enumerate(args):
        i_args_div, i_args_mod = int(i/len_fig_side), i % len_fig_side
        show_region = args[i].real
        show_region = show_region[int(x_len/2) - width_im:int(x_len/2) + width_im, int(y_len/2) - width_im:int(y_len/2) + width_im]
        print(np.shape(show_region))
        print(np.max(show_region))
        axs[i_args_div, i_args_mod].imshow(args[i].real.T, origin = "lower", vmax = np.max(show_region), vmin = np.min(show_region))
        axs[i_args_div, i_args_mod].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
        axs[i_args_div, i_args_mod].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)

    if save_folder !=None:
        plt.savefig(save_folder + "image.pdf", bbox_inches='tight')
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
        plt.savefig(save_folder + "zoom_view.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    fig, axs = plt.subplots(len_fig_side, len_fig_side, figsize=fig_size )
    for (i, dmy) in enumerate(args):
        i_args_div, i_args_mod = int(i/len_fig_side), i % len_fig_side
        axs[i_args_div, i_args_mod].hist(args[i].real.flatten(), bins=50)
        axs[i_args_div, i_args_mod].set_yscale('log')
        rms = np.sqrt(np.mean((answer.real - args[i].real)**2))/np.mean(answer.real)
        print("%d model, rms:%e" % (i, rms))

    if save_folder !=None:
        plt.savefig(save_folder + "hist_val.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None



def plots_vis(*args, vis_obs, fig_size = (10,10), save_folder = None):
    vis_obs_conv = np.fft.fftshift(vis_obs)

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
        plt.savefig(save_folder + "vis.pdf", bbox_inches='tight')
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
        plt.savefig(save_folder + "vis_hist.pdf", bbox_inches='tight')
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
        

def plots_model(*args, width_im = 10, save_folder = None):

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
        axs[i_args_div, i_args_mod].scatter(dist.flatten(), dmy.real.flatten(), s =ms_size)
        axs[i_args_div, i_args_mod].set_xlim(0, width_im)

    if save_folder !=None:
        plt.savefig(save_folder + "model_dist.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return None
