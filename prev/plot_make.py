import numpy as np
import matplotlib.pyplot as plt
import os

def make_dir(dirName):
    try:
    # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")


def plots_comp(*args, width_im = 10, fig_size = (10,3), save_folder = None):

    x_len, y_len = np.shape(args[0].T)
    len_model = len(args)
    fig, axs = plt.subplots(1, len_model, figsize=fig_size )
    answer = args[len(args)-1]

    for (i, dmy) in enumerate(args):
        axs[i].imshow(args[i].real.T, origin = "lower")
        axs[i].set_xlim(int(x_len/2) - width_im,int(x_len/2) + width_im)
        axs[i].set_ylim(int(y_len/2) - width_im,int(y_len/2) + width_im)

    if save_folder !=None:
        plt.savefig(save_folder + "image.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    num_center = int(x_len * y_len/2)
    fig, axs = plt.subplots(1, len(args), figsize=fig_size )
    for (i, dmy) in enumerate(args):
        axs[i].plot(args[i].real.T.flatten())
        axs[i].set_xlim(num_center - width_im, num_center + width_im)
    if save_folder !=None:
        plt.savefig(save_folder + "zoom_view.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    fig, axs = plt.subplots(1, len(args), figsize=fig_size )
    for (i, dmy) in enumerate(args):
        axs[i].hist(args[i].real.flatten(), bins=50)
        axs[i].set_yscale('log')
        rms = np.sqrt(np.mean((answer.real - args[i].real)**2))/np.mean(answer.real)
        print("%d model, rms:%e" % (i, rms))

    if save_folder !=None:
        plt.savefig(save_folder + "hist_val.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    return None



def plots_vis(*args, vis_obs, save_folder = None):
    vis_obs_conv = np.fft.fftshift(vis_obs)

    nx, ny = np.shape(args[0])
    x = np. arange(nx)
    y = np. arange(ny)
    xx, yy= np.meshgrid(x, y, indexing='ij')
    dist = ((xx-nx/2)**2 + (yy-ny/2)**2 )**0.5


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
    fig, axs = plt.subplots(1, len(args), figsize=(10, 3))

    for (i, dmy) in enumerate(model_fft):
        ms_size = 5
        axs[i].scatter(dist_obs_plt, vis_obs_plt.real, s =ms_size, color = "k")
        axs[i].scatter(dist_obs_plt, model_fft[i].real, s =ms_size, color="r")

    if save_folder !=None:
        plt.savefig(save_folder + "vis.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    fig, axs = plt.subplots(1, len(args), figsize=(10, 3))

    for (i, dmy) in enumerate(model_fft_all):
        ms_size = 5
        axs[i].hist(model_fft_all[i].real, bins = 100, color = "k", log=True)
        
    if save_folder !=None:
        plt.savefig(save_folder + "vis_hist.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return None

def plots_model(*args, width_im = 10, save_folder = None):

    len_model = len(args)
    nx, ny = np.shape(args[0])
    x = np. arange(nx)
    y = np. arange(ny)
    xx, yy= np.meshgrid(x, y, indexing='ij')
    dist = ((xx-nx/2)**2 + (yy-ny/2)**2 )**0.5

    fig, axs = plt.subplots(1, len(args), figsize=(10, 3))
 
    for (i, dmy) in enumerate(args):
        ms_size = 5
        axs[i].scatter(dist.flatten(), dmy.real.flatten(), s =ms_size)
        axs[i].set_xlim(0, width_im)

    if save_folder !=None:
        plt.savefig(save_folder + "model_dist.pdf", bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return None
