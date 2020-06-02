import numpy as np
import matplotlib.pyplot as plt
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

def ring_make(x_len, y_len, x_lim, y_lim, r_main, width, function = gaussian_function_1d):
    xmin, xmax = x_lim
    ymin, ymax = y_lim
    x = np.linspace(xmin, xmax, x_len)
    y = np.linspace(ymin, ymax, y_len)
    
    x_shift = x - np.mean(x)
    y_shift = y - np.mean(y)
    args = (width, r_main)
    xx, yy= np.meshgrid(x_shift, y_shift, indexing='ij')
    r = (xx**2 + yy**2) **0.5
    return function(r, *args), xx, yy


def obs_make(images, obs_num, width, sn=10):
    fft_now = np.fft.fft2(images)
    fft_now = np.fft.fftshift(fft_now)
    vis = np.zeros(np.shape(fft_now), dtype=np.complex)
    obs = 1000
    x_len, y_len = np.shape(images)
    position_vis = [[i,j] for i in range(int(x_len/2) - width,int(x_len/2) + width) for j in range(int(y_len/2) - width,int(y_len/2) + width)]
    
    Space_Position = np.array(position_vis).reshape(-1, 2)      # make it 2D
    random_indices = np.arange(0, Space_Position.shape[0])    # array of all indices
    np.random.shuffle(random_indices)                         # shuffle the array
    obs_vis_index = Space_Position[random_indices[:obs_num]]
    for i in range(obs_num):
        vix_x, vix_y = obs_vis_index[i][0], obs_vis_index[i][1]
        vis[vix_x, vix_y] = fft_now[vix_x, vix_y] + (np.max(fft_now[vix_x, vix_y])/sn)*np.random.randn(1, 2).view(np.complex128)
    return np.fft.ifftshift(vis)

import numpy as np

class obs_make:

    def __init__(self, images, obs_num, period, sn, cadence, n_pos, target_elevation):
        self.images = images
        self.obs_num =self.obs_num
        self.period = period
        self.sn = sn
        self.cadence = cadence
        self.n_pos = n_pos
        self.target_elevation = target_elevation

    def make_observatory(self, n_pos, dim=3, radius = 6000):

        posi_obs = []

        while True:
            x = np.random.randn(dim)
            r = np.linalg.norm(x)
            if r != 0.:
              posi_obs.append(x/r)

        return  6000 * np.array(posi_obs.append(x/r))

    def target_position(self):
        z_theta = self.target_elevation[0]
        x_theta = self.target_elevation[1]
        C_z, S_z = np.cos(z_theta), np.sin(z_theta)
        C_y, S_y = np.cos(z_theta), np.sin(z_theta)
        R_z = np.array([[C_z,-S_z,0],[S_z, C_z,0],[0,0,1]])
        R_y = np.array([[C_y, 0, -S_y],[0,1,0],[S_y,0,C_y]])
        R = np.dot(R_z, R_y)
        e_y = np.array([0,1,0])
        e_x = np.array([1,0,0])
        return np.dot(R, e_y), np.dot(R, e_x)



    def time_observation(self, n_pos, dim=3, radius = 6000):
        position_obs = make_observatory(n_pos)
        



def obs_make(images, obs_num, width, sn=10, n_pos=6):


    fft_now = np.fft.fft2(images)
    fft_now = np.fft.fftshift(fft_now)
    vis = np.zeros(np.shape(fft_now), dtype=np.complex)
    obs = 1000
    x_len, y_len = np.shape(images)
    position_vis = [[i,j] for i in range(int(x_len/2) - width,int(x_len/2) + width) for j in range(int(y_len/2) - width,int(y_len/2) + width)]
    
    Space_Position = np.array(position_vis).reshape(-1, 2)      # make it 2D
    random_indices = np.arange(0, Space_Position.shape[0])    # array of all indices
    np.random.shuffle(random_indices)                         # shuffle the array
    obs_vis_index = Space_Position[random_indices[:obs_num]]
    for i in range(obs_num):
        vix_x, vix_y = obs_vis_index[i][0], obs_vis_index[i][1]
        vis[vix_x, vix_y] = fft_now[vix_x, vix_y] + (np.max(fft_now[vix_x, vix_y])/sn)*np.random.randn(1, 2).view(np.complex128)
    return np.fft.ifftshift(vis)




