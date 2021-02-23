import numpy as np
import matplotlib.pyplot as plt
import logging 
import os 
import sys
from scipy import optimize 
import data_make
import matplotlib as mpl
#from multiprocessing import Pool
#import parmap
import time
import solver_mfreq as s_freq
from numba import jit


FLAG_MIN = -100
COUNT = 0
logger = logging.getLogger(__name__)

@jit(nopython=True)
def calc_ave(vec, nx):
    sum_now = 0
    count = 0
    for i in range(nx):
        if vec[i] !=0:
            sum_now+=vec[i]
            count+=1.0
    if sum_now == 0 and count ==0:
        return 0
    else:
        return sum_now/count

@jit(nopython=True)
def constained_tsv_onepix(center, vec, flag, nx):
    sum_now = np.float64(0)
    for i in range(nx):
        if flag[i]:
            sum_now+=0.5*(center - vec[i])**2

    return sum_now

@jit(nopython=True)
def constained_d_tsv_onepix(center, vec, flag, nx):
    sum_now = 0
    for i in range(nx):
        if flag[i]:
            sum_now+=2 * (center - vec[i])
    return sum_now

@jit(nopython=True)
def tsv_constrained(mat, alpha_mask):
    nx, ny = np.shape(mat)
    mat_anot = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros(4)
    flag = np.zeros(4)
    sum_all = np.float64(0)

    for i in range(nx):
        for j in range(ny):
            if not alpha_mask[i][j]:
                sum_all+=0
                
            elif i==0 and j==0:
                v[0], v[1] = mat[1][0], mat[0][1]
                flag[0], flag[1] = alpha_mask[1][0], alpha_mask[0][1]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 2)
 
            elif i==nx-1 and j ==0:
                v[0], v[1] = mat[nx-2][0], mat[nx-1][1]
                flag[0], flag[1] = alpha_mask[nx-2][0], alpha_mask[nx-1][1]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 2)
 
            elif i == 0 and j==ny-1:
                v[0], v[1] = mat[0][ny-2], mat[1][ny-1]
                flag[0], flag[1] = alpha_mask[0][ny-2], alpha_mask[1][ny-1]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 2)
                    
            elif i == nx-1 and j==ny-1:
                v[0], v[1] = mat[nx-1][ny-2], mat[nx-2][ny-1]
                flag[0], flag[1] = alpha_mask[nx-1][ny-2], alpha_mask[nx-2][ny-1]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 2)
            
            elif i==0:
                v[0], v[1], v[2] = mat[0][j-1], mat[0][j+1], mat[1][j]
                flag[0], flag[1], flag[2] = alpha_mask[0][j-1], alpha_mask[0][j+1], alpha_mask[1][j]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 3)
                
            elif i==nx-1:
                v[0], v[1], v[2] = mat[nx-1][j-1], mat[nx-1][j+1], mat[nx-2][j]
                flag[0], flag[1], flag[2] = alpha_mask[nx-1][j-1], alpha_mask[nx-1][j+1], alpha_mask[nx-2][j]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 3)
            
                
            elif j==0:
                v[0], v[1], v[2] = mat[i-1][0], mat[i+1][0], mat[i][1]
                flag[0], flag[1], flag[2] = alpha_mask[i-1][0], alpha_mask[i+1][0], alpha_mask[i][1]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 3)
                        
                
            elif j==ny-1:
                v[0], v[1], v[2] = mat[i-1][ny-1], mat[i+1][ny-1], mat[i][ny-2]
                flag[0], flag[1], flag[2] = alpha_mask[i-1][ny-1], alpha_mask[i+1][ny-1], alpha_mask[i][ny-2]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 3)
                
            elif i>0 and i < nx-1 and j>0 and j<ny-1:
                v[0], v[1], v[2], v[3] = mat[i-1][j], mat[i+1][j], mat[i][j-1], mat[i][j+1]
                flag[0], flag[1], flag[2], flag[3] = alpha_mask[i-1][j], alpha_mask[i+1][j], alpha_mask[i][j-1],  alpha_mask[i][j+1]
                sum_all += constained_tsv_onepix(mat[i][j], v, flag, 4)
    return sum_all

@jit(nopython=True)
def d_tsv_constrained(mat, alpha_mask):
    nx, ny = np.shape(mat)
    d_tsv_mat = np.zeros((nx, ny), dtype=np.float64)
    v = np.zeros(4)
    flag= np.zeros(4)
    sum_all = 0

    for i in range(nx):
        for j in range(ny):
            if not alpha_mask[i][j]:
                d_tsv_mat[i][j] = 0
                
            elif i==0 and j==0:
                v[0], v[1] = mat[1][0], mat[0][1]
                flag[0], flag[1] = alpha_mask[1][0], alpha_mask[0][1]
                d_tsv_mat[i][j] = constained_d_tsv_onepix(mat[i][j], v, flag, 2)
 
            elif i==nx-1 and j ==0:
                v[0], v[1] = mat[nx-2][0], mat[nx-1][1]
                flag[0], flag[1] = alpha_mask[nx-2][0], alpha_mask[nx-1][1]
                d_tsv_mat[i][j] =constained_d_tsv_onepix(mat[i][j], v, flag, 2)
 
                    
            elif i == 0 and j==ny-1:
                v[0], v[1] = mat[0][ny-2], mat[1][ny-1]
                flag[0], flag[1] = alpha_mask[0][ny-2], alpha_mask[1][ny-1]
                d_tsv_mat[i][j] = constained_d_tsv_onepix(mat[i][j], v, flag, 2)
                    
            elif i == nx-1 and j==ny-1:
                v[0], v[1] = mat[nx-1][ny-2], mat[nx-2][ny-1]
                flag[0], flag[1] = alpha_mask[nx-1][ny-2], alpha_mask[nx-2][ny-1]
                d_tsv_mat[i][j] = constained_d_tsv_onepix(mat[i][j], v, flag, 2)
            
            elif i==0:
                v[0], v[1], v[2] = mat[0][j-1], mat[0][j+1], mat[1][j]
                flag[0], flag[1], flag[2] = alpha_mask[0][j-1], alpha_mask[0][j+1], alpha_mask[1][j]
                d_tsv_mat[i][j] = constained_d_tsv_onepix(mat[i][j], v, flag, 3)
                
            elif i==nx-1:
                v[0], v[1], v[2] = mat[nx-1][j-1], mat[nx-1][j+1], mat[nx-2][j]
                flag[0], flag[1], flag[2] = alpha_mask[nx-1][j-1], alpha_mask[nx-1][j+1], alpha_mask[nx-2][j]
                d_tsv_mat[i][j] =constained_d_tsv_onepix(mat[i][j], v, flag, 3)
            
                
            elif j==0:
                v[0], v[1], v[2] = mat[i-1][0], mat[i+1][0], mat[i][1]
                flag[0], flag[1], flag[2] = alpha_mask[i-1][0], alpha_mask[i+1][0], alpha_mask[i][1]
                d_tsv_mat[i][j] = constained_d_tsv_onepix(mat[i][j], v, flag, 3)
                        
                
            elif j==ny-1:
                v[0], v[1], v[2] = mat[i-1][ny-1], mat[i+1][ny-1], mat[i][ny-2]
                flag[0], flag[1], flag[2] = alpha_mask[i-1][ny-1], alpha_mask[i+1][ny-1], alpha_mask[i][ny-2]
                d_tsv_mat[i][j] = constained_d_tsv_onepix(mat[i][j], v, flag, 3)
                
            elif i>0 and i < nx-1 and j>0 and j<ny-1:
                v[0], v[1], v[2], v[3] = mat[i-1][j], mat[i+1][j], mat[i][j-1], mat[i][j+1]
                flag[0], flag[1], flag[2], flag[3] = alpha_mask[i-1][j], alpha_mask[i+1][j], alpha_mask[i][j-1],  alpha_mask[i][j+1]
                d_tsv_mat[i][j] = constained_d_tsv_onepix(mat[i][j], v, flag, 4)
    return d_tsv_mat 
 
def multi_freq_f_I(model_image, model_alpha, obs, noise, nu_arr, nu0, dx, dy, lambda_I):

    ## Load
    n_freq = len(nu_arr)
    nx, ny = np.shape(model_image)
    chi_sum = 0
    #model_alpha[model_image ==0] = 0

    ## Main
    for i_freq in range(n_freq):

        model_freqj = ((nu_arr[i_freq]/nu0)**model_alpha ) * model_image
        model_dash = data_make.convert_Idash_to_Idashdash(model_freqj) * model_freqj
        model_vis = np.fft.fft2(model_dash)
        model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

        obs_mask = (obs[i_freq] == 0)
        obs_mask_2 = ((obs[i_freq] == 0)==False)
        d_vis = np.zeros(np.shape(obs[i_freq]))
        d_vis = (model_vis - obs[i_freq])/noise[i_freq]
        d_vis[obs_mask] = 0

        chi_sum += np.sum(np.abs(d_vis)*np.abs(d_vis))

    return [chi_sum,  lambda_I* s_freq.TSV(model_image)]


def multi_freq_f_alpha(model_alpha, model_image, obs, noise, nu_arr, nu0, dx, dy, lambda_alpha, alpha_mask):

    model_alpha = np.array(model_alpha, dtype = np.float64)
    model_image = np.array(model_image, dtype = np.float64)
    ## Load
    n_freq = len(nu_arr)
    nx, ny = np.shape(model_image)
    chi_sum = 0

    ## Main
    for i_freq in range(n_freq):

        model_freqj = ((nu_arr[i_freq]/nu0)**model_alpha ) * model_image
        model_dash = data_make.convert_Idash_to_Idashdash(model_freqj) * model_freqj
        model_vis = np.fft.fft2(model_dash)
        model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

        obs_mask = (obs[i_freq] == 0)
        obs_mask_2 = ((obs[i_freq] == 0)==False)
        d_vis = np.zeros(np.shape(obs[i_freq]))
        d_vis = (model_vis - obs[i_freq])/noise[i_freq]
        d_vis[obs_mask] = 0
        chi_sum += np.sum(np.abs(d_vis)*np.abs(d_vis))

    #return [chi_sum,  lambda_alpha * TSV(model_alpha)]
    return [chi_sum,  lambda_alpha * tsv_constrained(model_alpha, alpha_mask)]

## gradient for I in multi-freq
def multi_freq_f_grad_I(model_image, model_alpha, obs, noise, nu_arr, nu0, dx, dy, lambda_I):

    ## Load
    n_freq = len(nu_arr)
    nx, ny = np.shape(model_image)
    d_f_d_I = np.zeros(np.shape(model_image))

    ## Main
    for i_freq in range(n_freq):

        model_freqj = ((nu_arr[i_freq]/nu0)**model_alpha ) * model_image
        model_dash = data_make.convert_Idash_to_Idashdash(model_freqj) * model_freqj
        model_vis = np.fft.fft2(model_dash)
        model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

        obs_mask = (obs[i_freq] == 0)
        d_vis = np.zeros(np.shape(obs[i_freq]))
        d_vis = (-model_vis + obs[i_freq])/noise[i_freq]
        d_vis = np.conjugate(data_make.convert_visdash_to_vis(d_vis, dx, dy)) * d_vis/noise[i_freq]
        d_vis[obs_mask] = 0

        ifft_A_d_vis = nx * ny*  np.fft.ifft2(d_vis)
        B_ifft_A_F = np.conjugate(data_make.convert_Idash_to_Idashdash(ifft_A_d_vis)) * ifft_A_d_vis
        sj_model = (nu_arr[i_freq]/nu0)**model_alpha 
        d_f_d_I  += - np.real(2 *sj_model * B_ifft_A_F)


    d_f_d_I += lambda_I * s_freq.d_TSV(model_image)
    return d_f_d_I

## gradient for alpha in multi-freq
def multi_freq_f_grad_alpha(model_alpha, model_image, obs, noise, nu_arr, nu0, dx, dy, lambda_alpha, alpha_mask):

    ## Load
    n_freq = len(nu_arr)
    nx, ny = np.shape(model_image)
    d_f_d_alpha = np.zeros(np.shape(model_image))

    ## Main
    for i_freq in range(n_freq):

        model_freqj = ((nu_arr[i_freq]/nu0)**model_alpha ) * model_image
        model_dash = data_make.convert_Idash_to_Idashdash(model_freqj) * model_freqj
        model_vis = np.fft.fft2(model_dash)
        model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

        obs_mask = (obs[i_freq] == 0)
        d_vis = np.zeros(np.shape(obs[i_freq]))
        d_vis = (-model_vis + obs[i_freq])/noise[i_freq]
        d_vis = np.conjugate(data_make.convert_visdash_to_vis(d_vis, dx, dy)) * d_vis/noise[i_freq]
        d_vis[obs_mask] = 0

        ifft_A_d_vis = nx * ny*  np.fft.ifft2(d_vis)
        B_ifft_A_F = np.conjugate(data_make.convert_Idash_to_Idashdash(ifft_A_d_vis)) * ifft_A_d_vis

        sj_model = (nu_arr[i_freq]/nu0)**model_alpha 
        model_freqj_for_dalpha = np.log(nu_arr[i_freq]/nu0) * ((nu_arr[i_freq]/nu0)**model_alpha ) * model_image

        d_f_d_alpha  += - np.real(2 *model_freqj_for_dalpha * B_ifft_A_F)


    d_f_d_alpha += lambda_alpha * d_tsv_constrained(model_alpha, alpha_mask)
    return d_f_d_alpha
## gradient for alpha in multi-freq
def multi_freq_f_grad_alpha_arr(model_alpha, model_image, obs, noise, nu_arr, nu0, dx, dy, lambda_alpha, alpha_mask):

    ## Load
    n_freq = len(nu_arr)
    nx, ny = np.shape(model_image)
    d_f_d_alpha = np.zeros(np.shape(model_image))

    ## Main
    for i_freq in range(n_freq):

        model_freqj = ((nu_arr[i_freq]/nu0)**model_alpha ) * model_image
        model_dash = data_make.convert_Idash_to_Idashdash(model_freqj) * model_freqj
        model_vis = np.fft.fft2(model_dash)
        model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

        obs_mask = (obs[i_freq] == 0)
        d_vis = np.zeros(np.shape(obs[i_freq]))
        d_vis = (-model_vis + obs[i_freq])/noise[i_freq]
        d_vis = np.conjugate(data_make.convert_visdash_to_vis(d_vis, dx, dy)) * d_vis/noise[i_freq]
        d_vis[obs_mask] = 0

        ifft_A_d_vis = nx * ny*  np.fft.ifft2(d_vis)
        B_ifft_A_F = np.conjugate(data_make.convert_Idash_to_Idashdash(ifft_A_d_vis)) * ifft_A_d_vis

        sj_model = (nu_arr[i_freq]/nu0)**model_alpha 
        model_freqj_for_dalpha = np.log(nu_arr[i_freq]/nu0) * ((nu_arr[i_freq]/nu0)**model_alpha ) * model_image

        d_f_d_alpha  += - np.real(2 *model_freqj_for_dalpha * B_ifft_A_F)


    #d_tsv_d_alpha = lambda_alpha * s_freq.d_TSV(model_alpha) * alpha_mask
    d_tsv_d_alpha = lambda_alpha * d_tsv_constrained(model_alpha, alpha_mask)
    return d_f_d_alpha, d_tsv_d_alpha * alpha_mask

def grad_numerical_alpha_comp(model_alpha, model_image, obs, noise, nu_arr, nu0, dx, dy, lambda_alpha, alpha_mask):
    model_alpha = np.array(model_alpha, dtype = np.float64)
    model_image = np.array(model_image, dtype = np.float64)
    grad_ana_chi, grad_ana_tsv = multi_freq_f_grad_alpha_arr(model_alpha * alpha_mask, model_image, obs, noise, nu_arr, nu0, dx, dy, lambda_alpha, alpha_mask)

    nx, ny = np.shape(model_alpha)
    model_alpha_new = np.copy(model_alpha)
    eps = 1e-5
    grad_new_chi = np.zeros(np.shape(model_alpha))
    grad_new_tsv = np.zeros(np.shape(model_alpha))
    for i in range(nx):
        for j in range(ny):
            model_alpha_new[i][j] += eps
            Q_0 = multi_freq_f_alpha(model_alpha, model_image, obs, noise, nu_arr, nu0, dx, dy, lambda_alpha, alpha_mask)
            Q_1 = multi_freq_f_alpha(model_alpha_new, model_image, obs, noise, nu_arr, nu0, dx, dy, lambda_alpha, alpha_mask)
            grad_new_chi[i][j] = (Q_1[0] - Q_0[0])/eps
            grad_new_tsv[i][j] = (Q_1[1] - Q_0[1])/eps
            model_alpha_new[i][j] -= eps
    print(np.max(model_alpha), np.min(model_alpha))
    print(len(model_alpha[model_alpha==0]))
    plt.scatter(grad_ana_chi, grad_new_chi)
    plt.show()
    plt.scatter(grad_ana_tsv, grad_new_tsv, color ="k")
    plt.scatter(grad_ana_tsv[alpha_mask], grad_new_tsv[alpha_mask], color="r")
    plt.show()

def fx_L1_mfista_bcd_image(init_model, loss_f_arr, grad_f, eta=1.1, L_init = 1, iter_max = 1000, iter_min =300, \
 mask_positive = True,  stop_ratio = 1e-10, plot_solve_curve = False, restart = True, max_l = 1e15, eta_min = 1.05,*args):

    (sub_model, obs,  noise, nu_arr, nu0, dx, dy, lambda_value, lambda_l1) = args
    args = (sub_model, obs,  noise, nu_arr, nu0, dx, dy, lambda_value)
    x_k=init_model
    y_k = init_model
    x_prev = init_model
    L=L_init
    tk = 1
    L1_norm_func = lambda x, *args: lambda_l1 * np.sum(np.abs(x))
    loss_f = lambda x, *args: np.sum(loss_f_arr(x, *args))


    ## Loop for mfista
    itercount = 0
    F_xk_arr = []
    if plot_solve_curve:
        fig, ax = plt.subplots()
    while(1):
        

        ## loop for L 
        f_grad_yk = grad_f(y_k, *args)
        L = L/eta 

        while(1):

            if L > max_l:
                if eta_min < eta:
                    L = L_init
                    eta = eta**0.5
                    iterncount = 0
                    x_k = init_model
                    y_k = init_model
                    x_prev = init_model

                    continue
                else:
                    logger.error("Too large L!! Change l1 or l2 values!!")
                    logger.info("end fitting: itercount: %d" % itercount)
                    chi, l2 =loss_f_arr(y_k, *args)
                    logger.info("chi:%e, L1:%e, Reg:%e " % (chi, np.sum(y_k), l2))
                    sum_statistics = [chi, l2]                    
                    return y_k, sum_statistics , "SOLVE ERROR"                #raise ValueError("fitting error!")
            if mask_positive:
                z_temp = s_freq.positive_soft_threshold(y_k -(1/L) *f_grad_yk,lambda_l1 *(1/L)) 
            else:
                z_temp = s_freq.soft_threshold(y_k -(1/L) *f_grad_yk,lambda_l1 *(1/L)) 
            Q_temp = s_freq.calc_Q(z_temp, y_k, loss_f, L1_norm_func, f_grad_yk, L, *args)
            F_temp = loss_f(z_temp, *args) + L1_norm_func(z_temp)
            logger.debug("itercount: %d, L: %e, eta:%e, Q: %e, F:%e" % (itercount, L, eta,Q_temp, F_temp))
            if F_temp < Q_temp:
                break
            else:
                L = L * eta

        z_k = z_temp
        tk_1 = (1 + np.sqrt(1+4*tk**2))/2
        F_xk = loss_f(x_k, *args) + L1_norm_func(x_k)
        F_zk = loss_f(z_k, *args) + L1_norm_func(z_k)
        F_xk_arr.append(np.log10(F_xk))

        if len(F_xk_arr)>iter_min:
            relative_dif =np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])

            if relative_dif < stop_ratio:
                if plot_solve_curve:
                    plt.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                    plt.savefig("./fitting_curve_l1+%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
                    plt.close()
                break


        if F_xk > F_zk:
            x_k = z_k
            y_k = x_k + ((tk-1)/tk_1) * (x_k - x_prev)
            x_prev = x_k
            tk = tk_1
        else:
            x_k = x_prev
            y_k = x_k + (tk/tk_1)*(z_k-x_k)
            tk = tk_1
            if restart:
                tk = 1
        itercount+=1

        if itercount%50==0:

            chi, l2 =loss_f_arr(x_k, *args)
            logger.debug("iternum:%d, L: %e, F_xk: %e, chi:%e, reg: %e, rela_err:%e"  % (itercount, L, np.log10(F_xk), chi, l2, np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])))
            if plot_solve_curve:
                line, = ax.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                plt.pause(0.01)
                line.remove()


        if itercount > iter_max:
            if plot_solve_curve:
                plt.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                plt.savefig("./fitting_curve_l1+%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
                plt.close()
            break
    relative_dif =np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])
    logger.info("end; fitting. iternum:%d, relative dif of F: %e\n" % (itercount, relative_dif))
    chi, l2 =loss_f_arr(y_k, *args)
    logger.info("chi:%e, L1:%e, Reg:%e " % (chi, np.sum(y_k), l2))
    sum_statistics = [chi, l2]
    return y_k, sum_statistics, "SOlVED"



def only_fx_mfista_bcd_alpha(init_model, loss_f_arr, grad_f, eta=1.1, L_init = 1, iter_max = 1000, \
    iter_min = 300, mask_positive = True, stop_ratio = 1e-10, plot_solve_curve = False, restart = True, max_l =1e15, eta_min = 1.05,  *args):

    (sub_model, obs, noise, nu_arr, nu0, dx, dy, lambda_value, alpha_mask) = args
    tk = 1
    x_k = init_model
    y_k = init_model
    x_prev = init_model
    L=L_init
    
    loss_f = lambda x, *args: np.sum(loss_f_arr(x, *args))
    zero_func_temp = lambda x, *args: 0

    F_xk_arr = [] 

    ## Loop for mfista
    itercount = 0
    if plot_solve_curve:
        fig, ax = plt.subplots()

    while(1):
        

        ## loop for L
        f_grad_yk = grad_f(y_k, *args)
        loss_f_yk = loss_f(y_k, *args)
        L = L/eta 
        #print(itercount, tk)
        while(1):
            logger.debug("itercount: %d, L: %e, eta:%e, tk%f" % (itercount, L, eta, tk))

            if L > max_l:
                if eta_min < eta:
                    L = L_init
                    eta = eta**0.5
                    iterncount = 0
                    continue
                    
                else:
                    logger.error("Too large L!! Make l1 or l2 regularization parameters small!!")
                    logger.info("end fitting: itercount: %d" % itercount)
                    chi, l2 =loss_f_arr(y_k, *args)
                    logger.info("chi:%e, L1:%e, Reg:%e " % (chi, np.sum(y_k), l2))
                    sum_statistics = [chi, l2]     
                    return y_k, sum_statistics, "SOLVE ERROR" 


            z_temp = y_k -(1/L) *f_grad_yk
            if mask_positive:
                z_temp[z_temp<0] = 0

            Q_temp = s_freq.calc_Q(z_temp, y_k, loss_f,zero_func_temp, f_grad_yk, L, *args)
            F_temp = loss_f(z_temp, *args)
            if F_temp < Q_temp:
                break
            else:
                L = L * eta

        z_k = y_k -(1/L) *f_grad_yk
        if mask_positive:
            z_k[z_k<0] = 0        
        tk_1 = (1 + np.sqrt(1+4*tk**2))/2 
        F_xk = loss_f(x_k, *args) 
        F_zk = loss_f(z_k, *args) 
        F_xk_arr.append(np.log10(F_xk)) 

        if len(F_xk_arr)>iter_min:
            relative_dif =np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - \
                np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])

            if relative_dif < stop_ratio:
                plt.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                plt.savefig("./fitting_curve_%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
                plt.close()
                break

        if F_xk > F_zk:
            x_k = z_k
            y_k = x_k + ((tk-1)/tk_1) * (x_k - x_prev)
            x_prev = x_k
            tk = tk_1
        else:
            x_k = x_prev
            y_k = x_k + (tk/tk_1)*(z_k-x_k)
            tk = tk_1
            if restart:
                tk = 1


        itercount+=1
        if itercount%50==0:
            chi, l2 =loss_f_arr(x_k, *args)
            logger.debug("iternum:%d, L: %e, F_xk: %e, chi:%e, reg: %e, rela_err:%e"  % (itercount, L, np.log10(F_xk), chi, l2, np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])))
            if plot_solve_curve:
                line, = ax.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                plt.pause(0.001)
                line.remove()

        if itercount > iter_max:
            if plot_solve_curve:
                plt.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                plt.savefig("./fitting_curve_%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
                plt.close()
            break

    relative_dif =np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])
    logger.info("end; fitting. iternum:%d, relative dif of F: %e" % (itercount, relative_dif))
    chi, l2 =loss_f_arr(y_k, *args)
    logger.info("chi:%e, L1:%e, Reg:%e " % (chi, np.sum(y_k), l2))
    sum_statistics = [chi, l2]
    return y_k, sum_statistics, "SOlVED"

def mask_alpha(image_now, image_res):
    return image_now > image_res

def bcd_mfreq(lambda1, lambda2, lambda_alpha_2, x_init, obs, noise, nu_arr, nu0, dx,dy, image_res, iter_max = 5, eta_init =1.1, \
    l_init = 1e-3, maxite = 100, minite = 40, positive_solve = True, stop_ratio = 1e-7, \
    plot_solve_curve = False, restart = True,  max_l = 1e15, eta_min = 1.05, grad_comp = False):

    x_init = s_freq.edge_zero(x_init, flag_2d =False)
    model_image, model_alpha= s_freq.x_to_I_alpha(x_init)
    mask_alpha_now = mask_alpha(model_image,  image_res)

    ## If you need to check analytical gradiant values againt numerical ones
    if grad_comp:
        grad_numerical_alpha_comp(model_alpha, model_image, obs, noise, nu_arr, nu0, dx, dy, \
         lambda_alpha_2, mask_alpha_now)    

    ## Prepare saving arrays
    chi_arr = []
    l1_I_arr = []
    tsv_I_arr = []
    tsv_alpha_arr = []
    tsv_alpha_now = lambda_alpha_2 * tsv_constrained(model_alpha, mask_alpha_now)
    image_arr = []
    alpha_arr = []
    for iter_count in range(iter_max):
        print("%d/%d iterations" % (iter_count, iter_max))

        ## Optimization for image
        I_new, sum_stat_now, result = fx_L1_mfista_bcd_image(model_image, multi_freq_f_I, multi_freq_f_grad_I, \
            eta_init, l_init, maxite, minite, positive_solve, stop_ratio, plot_solve_curve, restart, \
            max_l, eta_min,  model_alpha, obs, noise, nu_arr, nu0, dx, dy, \
            lambda2, lambda1)


        ## Do jobs after optimization
        model_image = I_new
        chi_now, tsv_I_now = sum_stat_now[0], sum_stat_now[1]
        chi_arr.append(chi_now)
        l1_I_arr.append(lambda1 * np.sum(model_image))
        tsv_alpha_arr.append(tsv_alpha_now)
        tsv_I_arr.append(tsv_I_now)
        image_arr.append(model_image)
        alpha_arr.append(model_alpha)


        ## Optimization for alpha
        mask_alpha_now = mask_alpha(model_image,  image_res)        
        alpha_new, sum_stat_now, result = only_fx_mfista_bcd_alpha(model_alpha, multi_freq_f_alpha, multi_freq_f_grad_alpha, \
            eta_init, l_init, maxite, minite, positive_solve, stop_ratio, plot_solve_curve, restart, \
            max_l, eta_min,  model_image, obs, noise, nu_arr, nu0, dx, dy, \
            lambda_alpha_2, mask_alpha_now)

        ## Do jobs after optimization
        model_alpha = alpha_new
        chi_now, tsv_alpha_now = sum_stat_now[0], sum_stat_now[1]
        chi_arr.append(chi_now)
        l1_I_arr.append(lambda1 * np.sum(model_image))
        tsv_alpha_arr.append(tsv_alpha_now)
        tsv_I_arr.append(tsv_I_now)
        image_arr.append(model_image)
        alpha_arr.append(model_alpha)


    sum_statistics = (np.array(chi_arr), np.array(l1_I_arr), np.array(tsv_I_arr), np.array(tsv_alpha_arr))
    sum_images = (np.array(image_arr), np.array(alpha_arr))
    return model_image, model_alpha, sum_statistics, sum_images

def solver_mfreq_for_loop_bcd(lambda1, lambda2, lambda_alpha_2, x_init, obs, noise, nu_arr, nu0, dx,dy, image_res):
    nx = int(np.sqrt(len(x_init)/2.0))

    alpha_prior = 0 + np.zeros((nx, nx)) ## no meaning if TSV for alpha
    x_init = s_freq.edge_zero(x_init, flag_2d =False)
    args = (obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, dx, dy)
    start = time.time()
    image, alpha, sum_statistics, sum_images = bcd_mfreq(lambda1, lambda2, lambda_alpha_2, x_init, obs, noise, nu_arr, nu0, dx, \
        dy, image_res)
    time_consumed = time.time() - start
    print()
    print("fittting w/")
    print("time:", time_consumed)
    print("Parameters: l1%f. l2%f. lalpha%f" % ( np.log10(lambda1), np.log10(lambda2), np.log10(lambda_alpha_2)))

    return image, alpha, sum_statistics, sum_images 

def solver_mfreq_several_reg_params_bcd_testruns(folder_name, x_init, image_res, obs, noise, nu_arr, nu0, log10_lam_l1_arr, log10_lam_ltsv_arr, log10_lam_alpha_arr, dx,dy):
    nx = int(np.sqrt(len(x_init)/2.0))
    image_I_arrs = np.zeros((len(log10_lam_l1_arr), len(log10_lam_ltsv_arr), len(log10_lam_alpha_arr), nx ,nx))
    image_alpha_arrs = np.zeros((len(log10_lam_l1_arr), len(log10_lam_ltsv_arr), len(log10_lam_alpha_arr), nx ,nx))
    log_10_lam_l1_all, log10_lam_ltsv_all,log10_lam_alpha_all =  s_freq.make_list_combinations(log10_lam_l1_arr, log10_lam_ltsv_arr, log10_lam_alpha_arr)
    image_arr = []
    alpha_arr = []

    for log_10_lam_l1 in log10_lam_l1_arr:
    	for log10_lam_ltsv in log10_lam_ltsv_arr:
    		for log10_lam_alpha in log10_lam_alpha_arr:


    			print(log_10_lam_l1, log10_lam_ltsv, log10_lam_alpha)
    			file_name = os.path.join(folder_name, "l1%d_ltsv%d_lalpha%d" \
    				% (log_10_lam_l1, log10_lam_ltsv, log10_lam_alpha))

    			image, alpha, sum_statistics, sum_images  = solver_mfreq_for_loop_bcd (10**log_10_lam_l1, 10**log10_lam_ltsv, 10**log10_lam_alpha, x_init, obs, noise, nu_arr, nu0, dx,dy, image_res)
    			chi_arr, l1_arr, tsv_arr, tsv_al_arr = sum_statistics
    			image_arr_temp, alpha_arr_temp = sum_images
    			np.savez(file_name, chi = chi_arr, l1 = l1_arr, tsv = tsv_arr, tsv_al = tsv_al_arr, \
    				image = image_arr_temp, alpha = alpha_arr_temp)
    			image_arr.append(image)
    			alpha_arr.append(alpha)
    return log_10_lam_l1_all, log10_lam_ltsv_all,log10_lam_alpha_all, np.array([image_arr, alpha_arr])

def solver_mfreq_several_reg_params_bcd(x_init, image_res, obs, noise, nu_arr, nu0, log10_lam_l1_arr, log10_lam_ltsv_arr, log10_lam_alpha_arr, dx,dy):

    image_I_arrs = np.zeros((len(log10_lam_l1_arr), len(log10_lam_ltsv_arr), len(log10_lam_alpha_arr), nx ,nx))
    image_alpha_arrs = np.zeros((len(log10_lam_l1_arr), len(log10_lam_ltsv_arr), len(log10_lam_alpha_arr), nx ,nx))
    log_10_lam_l1_all, log10_lam_ltsv_all,log10_lam_alpha_all =  s_freq.make_list_combinations(log10_lam_l1_arr, log10_lam_ltsv_arr, log10_lam_alpha_arr)
    image_arr = []
    alpha_arr = []
    for log_10_lam_l1 in log_10_lam_l1_all:
    	for log10_lam_ltsv in log10_lam_ltsv_all:
    		for log10_lam_alpha in log10_lam_alpha_all:
    			print(log_10_lam_l1, log10_lam_ltsv, log10_lam_alpha)
    			image, alpha, sum_statistics, sum_images  = solver_mfreq_for_loop_bcd (10**log_10_lam_l1, 10**log10_lam_ltsv, 10**log10_lam_alpha, x_init, obs, noise, nu_arr, nu0, dx,dy, image_res)
    			image_arr.append(image)
    			alpha_arr.append(alpha)
    return log_10_lam_l1_all, log10_lam_ltsv_all,log10_lam_alpha_all, np.array([image_arr, alpha_arr])

