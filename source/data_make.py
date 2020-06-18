import numpy as np
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os



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
    xx, yy= np.meshgrid(x, y, indexing='ij')
    return function(xx, yy, *args), xx, yy



def coordinate_make(x_len, y_len, dx, dy):
    x = np.arange(0,x_len * dx, dx)
    y = np.arange(0,y_len * dy, dy)
    x_shift = x - np.mean(x)
    y_shift = y - np.mean(y)
    xx, yy= np.meshgrid(x_shift, y_shift, indexing='ij')
    return xx, yy
    
def ring_make(x_len, y_len, dx, dy, r_main, width, function = gaussian_function_1d):
    
    
    xx, yy = coordinate_make(x_len, y_len, dx, dy)

    args = (width, r_main)
    r = (xx**2 + yy**2) **0.5
    return function(r, *args), xx, yy


def gauss_2d(mu, mu2, sigma):
    x = random.gauss(mu, sigma)
    y = random.gauss(mu2, sigma)
    return (x, y)




class observatory:

    def __init__(self, images, obs_num, period, sn, duration, n_pos, radius, target_elevation, save_folder):
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

        self.antn =   self.radius * np.array(posi_obs)

    def e_unit_set(self, z_theta, y_theta):
        C_z, S_z = np.cos(z_theta), np.sin(z_theta)
        C_y, S_y = np.cos(y_theta), np.sin(y_theta)
        R_z = np.array([[C_z,-S_z,0],[S_z, C_z,0],[0,0,1]])
        R_y = np.array([[C_y, 0, S_y],[0,1,0],[-S_y,0,C_y]])
        R = np.dot(R_z, R_y)
        e_y = np.array([0,1,0])
        e_x = np.array([1,0,0])
        #e_z = (Cz Sy, Sz Sy, Cy)
        return np.dot(R, e_x), np.dot(R, e_y)



    def time_observation(self,  dim=3, radius = 6000):
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
        plt.savefig(self.save_folder+ "uv_plot.pdf", bbox_inches='tight')
        plt.close()
        return pos_arr
        
        
    def obs_make(self, dx, dy, sn):
        
        fft_now = np.fft.fft2(self.images)
        fft_now = np.fft.fftshift(fft_now)
        
        uv_arr = self.time_observation().T
        x_len, y_len = np.shape(self.images)
        u_max = 0.5/dx
        v_max = 0.5/dy
        du = 1/(dx * x_len)
        dv = 1/(dx * y_len)
        u = np.arange(0,x_len * du, du)
        v = np.arange(0,y_len * dv, dv)
        u_shift = u - np.mean(u)
        v_shift = v - np.mean(v)
        for_bins_u = np.append(u_shift, np.max(u_shift)+du) - du/2
        for_bins_v = np.append(v_shift, np.max(v_shift)+dv) - dv/2
        
        vis = np.zeros(np.shape(fft_now), dtype=np.complex)
        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(111)        
        num_mat= ax.hist2d(uv_arr[0],uv_arr[1], bins=[ for_bins_u, for_bins_v], cmap=cm.jet)
        num_mat = num_mat[0]
        plt.close()
        
        for i in range(x_len):
            for j in range(y_len):
                if num_mat[i][j] !=0:
                    real_now, imag_now = fft_now[i,j].real,  fft_now[i,j].imag
                    real_noise, imag_noise = gauss_2d(real_now, imag_now, (((real_now**2 + imag_now**2)**0.5)/(sn * num_mat[i][j])**0.5))
                    vis[i][j] = real_noise + imag_noise * 1j
        return np.fft.ifftshift(vis), np.fft.ifftshift(num_mat), fft_now

                


## Multi Frequency 

def spectral_power_model(nu_1, nu_0, xx, yy, beta_func = None):

    if beta_func is None:
        spectral_indices = (nu_1/nu_0)**(3 * np.ones(np.shape(xx)))
    else:
        spectral_indices= (nu_1/nu_0)**(beta_func(xx, yy))

    return spectal_indices



def ring_make_multi_frequency(x_len, y_len, dx, dy, r_main, width, nu_arr = [], nu0 = 1.00, spectral_beta_func = None, function = gaussian_function_1d):
    
    
    xx, yy = coordinate_make(x_len, y_len, dx, dy)

    args = (width, r_main)
    r = (xx**2 + yy**2) **0.5
    image_nu0 = function(r, *args)

    images_freqs = []
    for nu_dmy in nu_arr:
        images_freqs.append(image_nu0 * spectral_power_model(nu_dmy, nu0, xx, yy, spectral_beta_func))

    return images_freqs, xx, yy


class observatory_mfreq(observatory):

    def __init__(self, images, obs_num, period, sn, duration, n_pos, radius, target_elevation, lambda_arr, save_folder):
        super().__init__(images, obs_num, period, sn, duration, n_pos, radius, target_elevation, save_folder)
        self.lambda_arr = nu_arr

    def time_observation(self,  dim=3, radius = 6000):
        position_obs = self.antn
        time_arr = np.linspace(0, self.duration, self.obs_num)
        C_time = np.cos( time_arr * 2 * np.pi/ self.period)
        S_time = np.sin( time_arr * 2 * np.pi/ self.period)
        n_obs, n_pos = np.shape(position_obs )
        pos_time = []
        
        e_u, e_v = self.e_unit_set(self.target_elevation[0], self.target_elevation[1])

        for lambda_dmy in lambda_arr:
            for i in range(n_obs):
                pos_now = position_obs[i]/lambda_dmy
                pos_time.append([pos_now[0] * C_time - pos_now[1] * S_time, pos_now[0] * S_time + pos_now[1] * C_time, \
                                pos_now[2] * np.ones(len(time_arr))])
        
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
            uv_arr_freq.append(uv_arr)
        uv_arr_freq = np.array(uv_arr_freq)
        return uv_arr_freq


    def obs_make(self, dx, dy, sn):
        
        nfreq, x_len, y_len = np.shape(self.images)
        fft_images = []
        for i in range(nfreq):
            fft_images.append(np.fft.fftshift(np.fft.fft2(self.images)))
        fft_images = np.array(ffet_images)
        
        uv_arr_freq = self.time_observation().T
        u_max = 0.5/dx
        v_max = 0.5/dy
        du = 1/(dx * x_len)
        dv = 1/(dx * y_len)
        u = np.arange(0,x_len * du, du)
        v = np.arange(0,y_len * dv, dv)
        u_shift = u - np.mean(u)
        v_shift = v - np.mean(v)
        for_bins_u = np.append(u_shift, np.max(u_shift)+du) - du/2
        for_bins_v = np.append(v_shift, np.max(v_shift)+dv) - dv/2
        
        vis_freq = np.zeros((nfreq, x_len, y_len), dtype=np.complex)
        num_mat_freq = []
        for i_freq in range(nfreq):
            fig = plt.figure(figsize = (12,12))
            ax = fig.add_subplot(111)        
            num_mat = ax.hist2d(uv_arr[i_freq][0],uv_arr[i_freq][1], bins=[ for_bins_u, for_bins_v], cmap=cm.jet)
            num_mat = num_mat[0]
            plt.close()
            
            for i in range(x_len):
                for j in range(y_len):
                    if num_mat[i][j] !=0:
                        real_now, imag_now = fft_images[i_freq, i,j].real, fft_images[i_freq, i,j].imag
                        real_noise, imag_noise = gauss_2d(real_now, imag_now, (((real_now**2 + imag_now**2)**0.5)/(sn * num_mat[i][j])**0.5))
                        vis_freq[i_freq][i][j] = real_noise + imag_noise * 1j
            vis_freq[i_freq] = np.fft.ifftshift(vis_freq[i_freq])
            num_mat_freq.append(np.fft.ifftshift(num_mat))
        num_mat_freq = np.array(num_mat)

        return vis_freq, num_mat_freq, fft_images

