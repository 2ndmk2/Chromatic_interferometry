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

