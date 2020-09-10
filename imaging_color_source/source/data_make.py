import numpy as np
import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from scipy.stats import binned_statistic_2d
import pandas as pd
from scipy.stats import vonmises




def make_dir(dirName):
    try:
    # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

def gaussian_function_2d(x, y, *args):
    
    sigma_x, sigma_y, mu_x, mu_y = args
    x_now = x - mu_x
    y_now = y - mu_y
    return np.exp(-(x_now*x_now)/(2*sigma_x**2) - (y_now*y_now)/(2*sigma_y**2))/( (np.sqrt(2 * np.pi)**2) * sigma_x * sigma_y)


def gaussian_function_1d(x,  *args):
    
    sigma_x, mu_x= args
    x_now = x - mu_x
    return np.exp(-(x_now*x_now)/(2*sigma_x**2)) /( np.sqrt(2 * np.pi) * sigma_x )


def gauss_make(x_len, y_len, x_lim, y_lim, function = gaussian_function_1d, args = (5, 5, 100, 100)):
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    x = np.linspace(xmin, xmax, x_len)
    y = np.linspace(ymin, ymax, y_len)
    yy, xx= np.meshgrid(y, x, indexing='ij')
    return function(xx, yy, *args), xx, yy

def coordinate_make(x_len, y_len, dx, dy):
    x = np.arange(0,x_len * dx, dx)
    y = np.arange(0,y_len * dy, dy)
    x_shift = x - np.mean(x)
    y_shift = y - np.mean(y)
    yy, xx= np.meshgrid(x_shift, y_shift, indexing='ij')

    du = 1/(dx * x_len)
    dv = 1/(dy * y_len)
    u = np.arange(0,x_len * du, du)
    v = np.arange(0,y_len * dv, dv)
    u_shift = u - np.mean(u)
    v_shift = v - np.mean(v)

    vv, uu= np.meshgrid(u_shift, v_shift, indexing='ij')
    return xx, yy, uu, vv
    
def ring_make(x_len, y_len, dx, dy, r_main, width, function = gaussian_function_1d):
    
    xx, yy = coordinate_make(x_len, y_len, dx, dy)
    args = (width, r_main)
    r = (xx**2 + yy**2) **0.5
    
    return function(r, *args), xx, yy

def gauss_make(x_len, y_len, dx, dy, r_main, width, function = gaussian_function_1d):
    xx, yy = coordinate_make(x_len, y_len, dx, dy)


def gauss_2d(mu, mu2, sigma):
    
    x = random.gauss(mu, sigma)
    y = random.gauss(mu2, sigma)
    
    return (x, y)
def image_alpha_to_images(image_nu0,alpha, nu_arr, nu0):
    
    images = []
    for nu_dmy in nu_arr:
        images.append(image_nu0 * (nu_dmy/nu0)**alpha)
    return np.array(images)

## A
def convert_visdash_to_vis(vis, dx, dy):

    x_len, y_len = np.shape(vis)
    x = np.arange(x_len)
    y = np.arange(y_len)
    yy, xx = np.meshgrid(x,y)
    phase_factor = np.exp( (float(x_len-1)/float(x_len)) * np.pi*1j* (xx + yy))
    phase_factor2 = np.exp(-np.pi*1j* (float((x_len-1)**2/float(x_len))))
    
    return phase_factor * phase_factor2 


## A^-1
def convert_vis_to_visdash(vis, dx, dy):

    x_len, y_len = np.shape(vis)
    x = np.arange(x_len)
    y = np.arange(y_len)
    yy, xx = np.meshgrid(x,y)
    phase_factor = np.exp( - (float(x_len-1)/float(x_len)) * np.pi*1j* (xx + yy))
    phase_factor2 = np.exp(np.pi*1j* (float((x_len-1)**2/float(x_len))))

    return phase_factor * phase_factor2 

## B
def convert_Idash_to_Idashdash(image):

    x_len, y_len = np.shape(image)
    x = np.arange(x_len)
    y = np.arange(y_len)
    yy, xx = np.meshgrid(x,y)
    phase_factor = np.exp((float(x_len-1)/float(x_len)) * np.pi*1j* (xx + yy))

    return phase_factor 

## B^-1
def convert_Idashdash_to_Idash(image):

    x_len, y_len = np.shape(image)
    x = np.arange(x_len)
    y = np.arange(y_len)
    yy, xx = np.meshgrid(x,y)
    phase_factor = np.exp(-(float(x_len-1)/float(x_len)) *np.pi*1j* (xx + yy))

    return phase_factor 


class observatory:

    def __init__(self, images, obs_num, period, sn, duration, n_pos, \
        radius, target_elevation, save_folder):
        
        self.images = images
        self.obs_num =obs_num
        self.period = period
        self.sn = sn
        self.duration = duration
        self.n_pos = n_pos
        self.target_elevation = target_elevation
        self.radius = radius
        self.save_folder = save_folder

    def set_antn(self, dim=3):

        posi_obs = []
        count = 0

        while True:
            x = np.random.randn(dim)
            r = np.linalg.norm(x)

            if r != 0.:
                posi_obs.append(x/r)
                count += 1
                if count == self.n_pos:
                    break

        self.antn =  self.radius * np.array(posi_obs)

    def e_unit_set(self, z_theta, y_theta):

        C_z, S_z = np.cos(z_theta), np.sin(z_theta)
        C_y, S_y = np.cos(y_theta), np.sin(y_theta)
        R_z = np.array([[C_z,-S_z,0],[S_z, C_z,0],[0,0,1]])
        R_y = np.array([[C_y, 0, S_y],[0,1,0],[-S_y,0,C_y]])
        R = np.dot(R_z, R_y)
        e_y = np.array([0,1,0])
        e_x = np.array([1,0,0])
        return np.dot(R, e_x), np.dot(R, e_y)



    def time_observation(self,  dim=3):

        position_obs = self.antn
        time_arr = np.linspace(0, self.duration, self.obs_num)
        C_time = np.cos( time_arr * 2 * np.pi/ self.period)
        S_time = np.sin( time_arr * 2 * np.pi/ self.period)
        n_obs, n_pos = np.shape(position_obs )
        pos_time = []
        
        e_u, e_v = self.e_unit_set(self.target_elevation[0], self.target_elevation[1])
        for i in range(n_obs):
            pos_now = position_obs[i]
            pos_time.append([pos_now[0] * C_time - pos_now[1] * S_time, pos_now[0] * S_time + pos_now[1] * C_time, \
                            pos_now[2] * np.ones(len(time_arr))])
        
        pos_time = np.array(pos_time)
        n_obs, n_pos, n_time = np.shape(pos_time)
        uv_arr = []
        for i in range(n_time):
            pos_time_i = pos_time[:,:,i]
            for j in range(n_obs):
                for k in range(n_obs):
                    if j != k:
                        d_pos = pos_time_i[j] - pos_time_i[k]
                        uv_arr.append([np.dot(d_pos, e_u), np.dot(d_pos, e_v)] )
        uv_arr = np.array(uv_arr)
        return uv_arr
    
    def plotter_uv_sampling(self):
        
        fig = plt. figure(figsize=(8.0, 8.0))
        pos_arr = self.time_observation().T
        lim_max= np.max(np.abs(pos_arr))*1.3
        plt.xlim(-lim_max, lim_max)
        plt.ylim(-lim_max, lim_max)
        plt.axes().set_aspect('equal')        
        plt.scatter(pos_arr[0,:], pos_arr[1,:], s = 1)
        plt.savefig(self.save_folder+ "uv_plot.png", bbox_inches='tight', dpi = 200)
        plt.close()
        return pos_arr
        
        
    def obs_make(self, dx, dy, sn, images_load = None):


        if images_load is not None:
            images = images_load
        else:
            images = self.images
        image_dashdash = convert_Idash_to_Idashdash(images) * images
        fft_before_shift = np.fft.fft2(image_dashdash)
        fft_after_shift = convert_visdash_to_vis(fft_before_shift, dx, dy) * fft_before_shift
        
        uv_arr = self.time_observation().T
        x_len, y_len = np.shape(images)
        
        du = 1/(dx * x_len)
        dv = 1/(dx * y_len)
        u = np.arange(0,x_len * du, du)
        v = np.arange(0,y_len * dv, dv)
        u_shift = u - np.mean(u)
        v_shift = v - np.mean(v)
        for_bins_u = np.append(u_shift, np.max(u_shift)+du) - du/2
        for_bins_v = np.append(v_shift, np.max(v_shift)+dv) - dv/2
        
        vis = np.zeros(np.shape(fft_after_shift), dtype=np.complex)
        noise = np.zeros(np.shape(fft_after_shift), dtype=np.complex)

        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(111)        
        num_mat= ax.hist2d(uv_arr[0],uv_arr[1], bins=[for_bins_u, for_bins_v], cmap=cm.jet)
        num_mat = num_mat[0]
        vis_amp_noise = np.max(np.abs(fft_after_shift))/float(np.sqrt(sn))
        plt.close()

        
        for i in range(x_len):
            for j in range(y_len):
                
                if num_mat[i][j] !=0:
                    noise_amp = vis_amp_noise/np.sqrt(num_mat[i][j])
                    real_now, imag_now = fft_after_shift[i,j].real,  fft_after_shift[i,j].imag
                    real_noise, imag_noise = gauss_2d(real_now, imag_now, noise_amp)
                    vis[i][j] = real_noise + imag_noise * 1j
                    noise[i][j] = noise_amp

        noise[noise ==0] = 1
        return vis, num_mat, fft_after_shift, noise

                


## Multi Frequency 

def spectral_power_model(nu_1, nu_0, xx, yy, alpha_func = None):

    if alpha_func is None:
        spectral_indices = (nu_1/nu_0)**(3 * np.ones(np.shape(xx)))
    else:
        spectral_indices = (nu_1/nu_0)** (alpha_func(xx, yy))

    return spectral_indices 


## Jy/pix
def rings_gaps_I0(x_len, y_len, dx, dy, positions, widths, fractions, gaussian_maj, flux_max=1e-3):

    xx, yy, uu, vv = coordinate_make(x_len, y_len, dx, dy)
    major_emission = gaussian_function_2d(xx, yy, gaussian_maj, gaussian_maj, 0, 0)
    flux_returns = major_emission
    r = (xx**2 + yy**2) **0.5

    for i in range(len(positions)):

        fractional_gaps = gaussian_function_1d(r, widths[i],positions[i])
        fractional_gaps_factor = fractions[i]/np.max(fractional_gaps)
        fractional_gaps = 1 - (fractional_gaps*fractional_gaps_factor)
        flux_returns = flux_returns * fractional_gaps

    r_arr = np.ravel(r)
    flux_returns = flux_max * flux_returns/(np.max(flux_returns))
    flux_arr = np.ravel(flux_returns)
    return flux_returns, r_arr, flux_arr

def rings_gaps_I0_components(x_len, y_len, dx, dy, positions, widths, fractions, gaussian_maj):

    yy, xx = coordinate_make(x_len, y_len, dx, dy)
    major_emission = gaussian_function_2d(xx, yy, gaussian_maj, gaussian_maj, 0, 0)
    flux_returns = major_emission
    r = (xx**2 + yy**2) **0.5
    flux_arrs = []
    flux_arrs.append(major_emission)


    for i in range(len(positions)):

        fractional_gaps = gaussian_function_1d(r, widths[i],positions[i])
        fractional_gaps_factor = fractions[i]/np.max(fractional_gaps)
        fractional_gaps = 1 - (fractional_gaps*fractional_gaps_factor)
        flux_returns = flux_returns * fractional_gaps
        flux_arrs.append(fractional_gaps*fractional_gaps_factor * major_emission)

    return flux_arrs

def rings_gaps_alpha(x_len, y_len, dx, dy, positions, widths, height, gaussian_maj, alpha_center=2, alpha_outer = 4):

    xx, yy, uu, vv = coordinate_make(x_len, y_len, dx, dy)
    major_alpha = gaussian_function_2d(xx, yy, gaussian_maj, gaussian_maj, 0, 0)
    alpha_height = alpha_outer - alpha_center  
    major_alpha_factor = alpha_height/np.max(major_alpha)
    major_alpha = alpha_outer - major_alpha * major_alpha_factor
    alpha_returns = major_alpha 
    r = (xx**2 + yy**2) **0.5

    for i in range(len(positions)):
        
        fractional_gaps = gaussian_function_1d(r, widths[i],positions[i])
        fractional_gaps_factor = height[i]/np.max(fractional_gaps)
        fractional_gaps = fractional_gaps*fractional_gaps_factor
        alpha_returns = alpha_returns + fractional_gaps

    r_arr = np.ravel(r)
    alpha_arr = np.ravel(alpha_returns)
    return alpha_returns, r_arr, alpha_arr 

def HD14_like_I0(x_len, y_len, dx, dy, r_position, r_width, theta_position, theta_width, flux_max=1e-3):

    xx, yy, uu, vv = coordinate_make(x_len, y_len, dx, dy)
    r = (xx**2 + yy**2) **0.5
    theta = np.arctan2(yy, xx)
    I0 = np.ones(np.shape(xx))
    I0 = I0 * gaussian_function_1d(r, r_width, r_position)
    I0 = I0 * vonmises.pdf(theta - theta_position, 1/np.sqrt(theta_width))
    I0 = I0 * flux_max/np.max(I0)
    r_arr = np.ravel(r)
    I0_arr = np.ravel(I0)
    return I0, r_arr, I0_arr

def HD14_like_I0_shifted(x_len, y_len, dx, dy, r_position, r_width, theta_position, theta_width, flux_max=1e-3):

    xx, yy, uu, vv = coordinate_make(x_len, y_len, dx, dy)
    r = ((xx-0.3/206265.0)**2 + (yy-0.5/206265.0)**2) **0.5
    theta = np.arctan2(yy, xx)
    I0 = np.ones(np.shape(xx))
    I0 = I0 * gaussian_function_1d(r, r_width, r_position)
    I0 = I0 * vonmises.pdf(theta - theta_position, 1/np.sqrt(theta_width))
    I0 = I0 * flux_max/np.max(I0)
    r_arr = np.ravel(r)
    I0_arr = np.ravel(I0)
    return I0, r_arr, I0_arr


def HD14_like_alpha(x_len, y_len, dx, dy, r_position, r_width, theta_position, theta_width,  alpha_min=2, alpha_max=4):
    xx, yy, uu, vv = coordinate_make(x_len, y_len, dx, dy)
    r = (xx**2 + yy**2) **0.5
    theta = np.arctan2(yy, xx)    
    alpha = np.ones(np.shape(xx))
    alpha = alpha * gaussian_function_1d(r, r_width, r_position)
    alpha = alpha * vonmises.pdf(theta - theta_position, 1/np.sqrt(theta_width))
    alpha = alpha_min + (alpha_max-alpha_min) * \
    (alpha - np.min(alpha))/(np.max(alpha))

    r_arr = np.ravel(r)
    alpha_arr = np.ravel(alpha)
    return  alpha, r_arr, alpha_arr


def radial_make_multi_frequency(x_len, y_len, dx, dy, r_main, width, nu_arr = [], nu0 = 1.00, spectral_alpha_func = None, function = gaussian_function_1d):
    
    
    xx, yy, uu, vv = coordinate_make(x_len, y_len, dx, dy)

    args = (width, r_main)
    r = (xx**2 + yy**2) **0.5
    image_nu0 = function(r, *args)
    images_freqs = []

    for nu_dmy in nu_arr:
        images_freqs.append(image_nu0 * spectral_power_model(nu_dmy, nu0, xx, yy, spectral_alpha_func))

    return images_freqs, image_nu0, xx, yy


def multi_spectral_data_make(I0, alpha, nu_arr, nu0):

    images_freqs = []
    
    for nu_dmy in nu_arr:
        images_freqs.append(I0 * (nu_dmy/nu0)** alpha )

    return images_freqs

class observatory_mfreq(observatory):

    def __init__(self, images, obs_num, period, sn, duration, n_pos, radius, target_elevation, lambda_arr, lambda0, save_folder):
        super().__init__(images, obs_num, period, sn, duration, n_pos, radius, target_elevation, save_folder)
        self.lambda_arr = lambda_arr
        self.lambda0 = lambda0

    def time_observation(self,  dim=3):

        position_obs = self.antn
        time_arr = np.linspace(0, self.duration, self.obs_num)
        C_time = np.cos( time_arr * 2 * np.pi/ self.period)
        S_time = np.sin( time_arr * 2 * np.pi/ self.period)
        n_obs, n_pos = np.shape(position_obs )

        pos_time = []
        e_u, e_v = self.e_unit_set(self.target_elevation[0], self.target_elevation[1])

        for lambda_dmy in self.lambda_arr:
            pos_time_dmy = []
            for i in range(n_obs):
                pos_now = position_obs[i]/lambda_dmy
                pos_time_dmy.append([pos_now[0] * C_time - pos_now[1] * S_time, pos_now[0] * S_time + pos_now[1] * C_time, \
                                pos_now[2] * np.ones(len(time_arr))])
            pos_time.append(pos_time_dmy)

        pos_time = np.array(pos_time)
        n_freq, n_obs, n_pos, n_time = np.shape(pos_time)
        uv_arr_freq = []
        for lam_i in range(n_freq):
            uv_arr = []
            for i in range(n_time):
                pos_time_i = pos_time[lam_i,:,:,i]
                for j in range(n_obs):
                    for k in range(n_obs):
                        if j != k:
                            d_pos = pos_time_i[j] - pos_time_i[k]
                            uv_arr.append([np.dot(d_pos, e_u), np.dot(d_pos, e_v)] )
            uv_arr_freq.append(np.array(uv_arr).T)
        uv_arr_freq = np.array(uv_arr_freq)

        return uv_arr_freq

    def plotter_uv_sampling(self):
        
        fig = plt. figure(figsize=(8.0, 8.0))
        plt.axes().set_aspect('equal')        
        uv_arr_freq = self.time_observation().T
        lim_max= np.max(np.abs(uv_arr_freq))*1.3
        plt.xlim(-lim_max, lim_max)
        plt.ylim(-lim_max, lim_max)
        nx, ny, n_freq = np.shape(uv_arr_freq)
        for i in range(n_freq):
            plt.scatter(uv_arr_freq[:,0,i], uv_arr_freq[:,1,i], s = 1)
        plt.savefig(self.save_folder+ "uv_plot.pdf", bbox_inches='tight')
        plt.close()
        return uv_arr_freq


    def obs_make(self, dx, dy, sn, images_load = None):


        if images_load is not None:
            images = images_load
        else:
            images = self.images

        nfreq, x_len, y_len = np.shape(images)
        fft_images = []
        nu_arr = 1/self.lambda_arr
        nu0 = 1/self.lambda0

        for i in range(nfreq):

            image_dashdash = convert_Idash_to_Idashdash(images[i]) * images[i]
            fft_before_shift = np.fft.fft2(image_dashdash)
            fft_after_shift = convert_visdash_to_vis(fft_before_shift, dx, dy) * fft_before_shift
            fft_images.append(fft_after_shift)

        fft_images = np.array(fft_images)

        uv_arr_freq = self.time_observation()

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
        num_mat_freq = []

        for i_freq in range(nfreq):

            fig = plt.figure(figsize = (12,12))
            ax = fig.add_subplot(111)        
            num_mat = ax.hist2d(uv_arr_freq[i_freq][0],uv_arr_freq[i_freq][1], bins=[ for_bins_u, for_bins_v], cmap=cm.jet)
            num_mat = num_mat[0]
            num_mat_freq.append(num_mat)
            plt.close()
            vis_amp_noise =  np.max(np.abs(fft_images[i_freq]))/float(np.sqrt(sn))

            for i in range(x_len):
                for j in range(y_len):
                    if num_mat[i][j] !=0:
                        noise_amp = vis_amp_noise/np.sqrt(num_mat[i][j])
                        real_now, imag_now = fft_images[i_freq, i,j].real, fft_images[i_freq, i,j].imag
                        real_noise, imag_noise = gauss_2d(real_now, imag_now, noise_amp)
                        vis_freq[i_freq][i][j] = real_noise + imag_noise * 1j
                        noise_freq[i_freq][i][j] = noise_amp
  
        noise_freq[noise_freq ==0] = 1
        num_mat_freq = np.array(num_mat_freq)
        return vis_freq, num_mat_freq, fft_images, noise_freq
## Calculation of TSV
def TSV(mat):
    sum_tsv = 0
    Nx, Ny = np.shape(mat)
    
    # TSV terms from left to right 
    mat_2 = np.roll(mat, shift = 1, axis = 1) 
    sum_tsv += np.sum( (mat_2[:,1:Ny]-mat[:,1:Ny]) * (mat_2[:,1:Ny]-mat[:,1:Ny]) )
    
    # TSV terms from bottom to top  
    mat_3 = np.roll(mat, shift = 1, axis = 0) 
    sum_tsv += np.sum( (mat_3[1:Nx, :]-mat[1:Nx, :]) * (mat_3[1:Nx, :]-mat[1:Nx, :]) )
    
    #Return all TSV terms
    return sum_tsv

def print_chi_L1_TSV_for_inputmodel(vis_obs, vis_model, noise, image_model, alpha_model):
    d_vis = (vis_obs - vis_model)/noise
    d_vis[vis_obs==0] = 0
    chi = np.sum(np.abs(d_vis)*np.abs(d_vis))
    l1 = np.sum(image_model)
    TSV_image = TSV(image_model)
    TSV_alpha = TSV(alpha_model)
    print("Input model:")
    print("chi:%e, l1:%e, TSV:%e, TSV_alpha:%e" % (chi, l1, TSV_image, TSV_alpha))
    print("if chi=1, l1:%e, TSV:%e, TSV_alpha:%e" % (chi/l1, chi/TSV_image, chi/TSV_alpha)) 

## No masking fourier trasnform
def fourier_image(images, dx, dy, lambda_arr, lambda0):
    
    nfreq, x_len, y_len = np.shape(images)
    fft_images = []
    nu_arr = 1/lambda_arr
    nu0 = 1/lambda0

    for i in range(nfreq):

        image_dashdash = convert_Idash_to_Idashdash(images[i]) * images[i]
        fft_before_shift = np.fft.fft2(image_dashdash)
        fft_after_shift = convert_visdash_to_vis(fft_before_shift, dx, dy) * fft_before_shift
        fft_images.append(fft_after_shift)

    fft_images = np.array(fft_images)

    return fft_images




