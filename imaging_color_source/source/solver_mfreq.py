import numpy as np
import matplotlib.pyplot as plt
import logging 
import os 
import sys
from scipy import optimize 
import data_make
import matplotlib as mpl
COUNT = 0

logger = logging.getLogger(__name__)

sys.path.insert(0,'../../config')
from setting_freq_image import *
from setting_freq_common import *




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

## Calculation of d_TSV
def d_TSV(mat):
    Nx, Ny = np.shape(mat)
    d_TSV_mat = np.zeros(np.shape(mat))
    mat_1 = np.roll(mat, shift = 1, axis = 1)
    mat_2 = np.roll(mat, shift = -1, axis = 1)
    mat_3 = np.roll(mat, shift = 1, axis = 0)
    mat_4 = np.roll(mat, shift = -1, axis = 0)
    dif_1 = 2 * (mat[:,1:Ny] - mat_1[:,1:Ny])
    dif_1 = np.pad(dif_1, [(0,0),(1,0)], mode = 'constant')
    dif_2 = 2 * (mat[:,0:Ny-1] - mat_2[:,0:Ny-1])
    dif_2 = np.pad(dif_2, [(0,0),(0,1)], mode = 'constant')
    dif_3 = 2 * (mat[1:Nx, :] - mat_3[1:Nx, :])
    dif_3 = np.pad(dif_3, [(1,0),(0,0)], mode = 'constant')
    dif_4 = 2 * (mat[0:Nx-1, :] - mat_4[0:Nx-1, :])
    dif_4 = np.pad(dif_4, [(0,1),(0,0)], mode = 'constant')

    return dif_1 + dif_2 + dif_3 + dif_4

def loss_function_l2(model, *args):
    (obs, noise, model_prior, lambda_l2, dx, dy) = args
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis
    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis = (model_vis- obs)/noise
    d_vis[obs_make] = 0


    return np.sum(np.abs(d_vis)**2) +  lambda_l2 * np.sum((model-model_prior)**2)

def loss_function_arr_l2(model, *args):
    (obs, noise, model_prior, lambda_l2, dx, dy) = args
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis = (model_vis- obs)/noise
    d_vis[obs_make] = 0

    return [np.sum(np.abs(d_vis)**2), lambda_l2 * np.sum((model-model_prior)**2)]


def grad_loss_l2(model, *args):
    
    (obs, noise, model_prior, lambda_l2, dx, dy) = args

    ## Gradient for L2
    dl2_dmodel =2 * lambda_l2 * (model-model_prior)
    nx, ny = np.shape(model)
    
    ## Gradient for chi^2
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis
    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis = (model_vis- obs)/noise
    d_vis[obs_make] = 0


    d_vis = np.conjugate(data_make.convert_visdash_to_vis(d_vis, dx, dy)) * d_vis
    ifft_F = np.fft.ifft2(d_vis)
    ifft_F = np.conjugate(data_make.convert_Idash_to_Idashdash(ifft_F)) * ifft_F
    dF_dmodel =2* nx * ny* ifft_F.real 
    return dF_dmodel + dl2_dmodel

def grad_loss_arr_l2(model, *args):
    
    (obs, noise, model_prior, lambda_l2, dx, dy) = args

    ## Gradient for L2
    dl2_dmodel =2 * lambda_l2 * (model-model_prior)
    nx, ny = np.shape(model)
    
    ## Gradient for chi^2
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis = (model_vis- obs)/noise
    d_vis[obs_make] = 0

    d_vis = np.conjugate(data_make.convert_visdash_to_vis(d_vis, dx, dy)) * d_vis
    ifft_F = np.fft.ifft2(d_vis)
    ifft_F = np.conjugate(data_make.convert_Idash_to_Idashdash(ifft_F)) * ifft_F
    dF_dmodel =2* nx * ny* ifft_F.real

    return dF_dmodel, dl2_dmodel 

## For comparison with above analytical expressions
def grad_loss_numerical_l2(model,i,j, *args):
    (obs,noise,  model_prior, lambda_l2, dx, dy) = args

    model_new = np.copy(model) 
    eps = 1e-5
    model_new[i][j] += eps
    chi_new, l2_new = loss_function_arr_l2(model_new, *args)
    chi_, l2_ = loss_function_arr_l2(model, *args)
    return (chi_new-chi_)/eps, (l2_new-l2_)/eps



def loss_function_TSV(model, *args):
    (obs, noise, model_prior, lambda_ltsv, dx, dy) = args
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis
    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis = (model_vis- obs)/noise
    d_vis[obs_make] = 0


    return np.sum(np.abs(d_vis)**2) +  lambda_ltsv * TSV(model)


def loss_function_arr_TSV(model, *args):
    (obs, noise,  model_prior, lambda_ltsv, dx, dy) = args
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis,dx, dy) * model_vis
    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis = (model_vis- obs)/noise
    d_vis[obs_mask] = 0

    return [np.sum(np.abs(d_vis)**2), lambda_ltsv * TSV(model)]


def grad_loss_tsv(model, *args):
    
    (obs, noise, model_prior, lambda_ltsv, dx, dy) = args

    ## Gradient for L2
    nx, ny = np.shape(model)
    
    
    ## Gradient for chi^2
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis = (model_vis- obs)/(noise*noise)
    d_vis[obs_mask] = 0

    d_vis = np.conjugate(data_make.convert_visdash_to_vis(d_vis,dx, dy)) * d_vis
    ifft_F = nx * ny* np.fft.ifft2(d_vis)
    ifft_F = np.conjugate(data_make.convert_Idash_to_Idashdash(ifft_F)) * ifft_F
    dF_dmodel =2*  ifft_F.real 

    return dF_dmodel + lambda_ltsv * d_TSV(model)

def grad_loss_arr_TSV(model, *args):
    
    (obs, noise,  model_prior, lambda_ltsv, dx, dy) = args

    ## Gradient for L2
    nx, ny = np.shape(model)
    
    ## Gradient for chi^2
    model_dash = data_make.convert_Idash_to_Idashdash(model) * model
    model_vis = np.fft.fft2(model_dash)
    model_vis = data_make.convert_visdash_to_vis(model_vis, dx, dy) * model_vis

    obs_mask = (obs == 0)
    d_vis = np.zeros(np.shape(obs))
    d_vis= (model_vis - obs)/(noise*noise)
    d_vis[obs_make] = 0

    d_vis = np.conjugate(data_make.convert_visdash_to_vis(d_vis,dx, dy)) * d_vis
    ifft_F = nx * ny*  np.fft.ifft2(d_vis)
    ifft_F = np.conjugate(data_make.convert_Idash_to_Idashdash(ifft_F)) * ifft_F
    dF_dmodel = 2* ifft_F.real 

    return dF_dmodel, lambda_ltsv * d_TSV(model)

def grad_loss_numerical_TSV(model, *args):
    (obs, noise, model_prior, lambda_l2, dx, dy) = args

    grad_tot = np.zeros(np.shape(model))
    nx, ny = np.shape(model)
    for i in range(nx):
        for j in range(ny):
            model_new = np.copy(model) 
            eps = 1e-6
            model_new[i][j] += eps
            chi_new, l2_new = loss_function_arr_TSV(model_new, *args)
            chi_, l2_ = loss_function_arr_TSV(model, *args)
            grad_tot[i][j] = (chi_new-chi_)/eps + (l2_new-l2_)/eps
    return grad_tot

## dummy function
def zero_func(model, *args):
    return 0


    ## For fx_L1_mfista
def L1_norm(model, *args):
    if len(args) ==0:
        return np.sum(np.abs(model))
    else:
        (obs, noise, lambda_l1, lambda_l2, dx, dy) = args
        return lambda_l1 * np.sum(np.abs(model))


## For FISTA algorithm
def calc_Q(x, y, f, g, grad_f, L, *args):
    term1 = f(y, *args)
    term2 = np.sum((x-y)*grad_f)
    term3 = (L/2) * np.sum((x-y)**2)
    term4 = g(x, *args)
    return term1 + term2 + term3 + term4

## For proximal mapping
def soft_threshold(model, lambda_t, print_flag=False):

    mask_temp1 = model> lambda_t
    mask_temp2 = (model> -lambda_t) * (model< lambda_t)
    mask_temp3 = model< - lambda_t
    model[mask_temp1] -= lambda_t
    model[mask_temp2] =0
    model[mask_temp3] += lambda_t
    if print_flag:
        logger.debug(len(model[mask_temp1]), len(model[mask_temp2]),len(model[mask_temp3]),lambda_t)
    return model

## For proximal mapping under positive condition
def positive_soft_threshold(model, lambda_t, print_flag=False):

    mask_temp1 = model> lambda_t
    mask_temp2 = model< lambda_t
    model[mask_temp1] -= lambda_t
    model[mask_temp2] =0
    if print_flag:
        logger.debug("%d %d %e" % (len(model[mask_temp1]), len(model[mask_temp2]),lambda_t))
    return model


def fx_L1_mfista(init_model, loss_f_arr, grad_f, loss_g = zero_func, eta=1.1, L_init = 1, iter_max = 1000, iter_min =300, \
 mask_positive = True,  stop_ratio = 1e-10, restart = True, *args):

    (obs, noise, lambda_l1, lambda_l2, dx, dy) = args
    tk = 1
    x_k=init_model
    y_k = init_model
    x_prev = init_model
    L=L_init

    loss_f = lambda x, *args: np.sum(loss_f_arr(x, *args))

    ## Loop for mfista
    itercount = 0
    F_xk_arr = []
    if PLOT_SOLVE_CURVE:
        fig, ax = plt.subplots()
    while(1):
        

        ## loop for L 
        f_grad_yk = grad_f(y_k, *args)
        L = L/eta 

        while(1):

            if L > MAX_L:
                if ETA_MIN < eta:
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
                    return y_k, SOLVED_FLAG_ER 
                #raise ValueError("fitting error!")
            if mask_positive:
                z_temp = positive_soft_threshold(y_k -(1/L) *f_grad_yk,lambda_l1 *(1/L)) 
            else:
                z_temp = soft_threshold(y_k -(1/L) *f_grad_yk,lambda_l1 *(1/L)) 
            Q_temp = calc_Q(z_temp, y_k, loss_f, L1_norm, f_grad_yk, L, *args)
            F_temp = loss_f(z_temp, *args) + lambda_l1 * L1_norm(z_temp)
            logger.debug("itercount: %d, L: %e, eta:%e, Q: %e, F:%e" % (itercount, L, eta,Q_temp, F_temp))
            if F_temp < Q_temp:
                break
            else:
                L = L * eta

        z_k = z_temp
        tk_1 = (1 + np.sqrt(1+4*tk**2))/2
        F_xk = loss_f(x_k, *args) + lambda_l1 * L1_norm(x_k)
        F_zk = loss_f(z_k, *args) + lambda_l1 * L1_norm(z_k)
        F_xk_arr.append(np.log10(F_xk))

        if len(F_xk_arr)>iter_min:
            relative_dif =np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])

            if relative_dif < stop_ratio:
                if PLOT_SOLVE_CURVE:
                    plt.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                    plt.savefig(FIG_FOLDER + "fitting_curve_l1+%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
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
            if PLOT_SOLVE_CURVE:
                line, = ax.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                plt.pause(0.01)
                line.remove()


        if itercount > iter_max:
            if PLOT_SOLVE_CURVE:
                plt.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
                plt.savefig(FIG_FOLDER + "fitting_curve_l1+%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
                plt.close()
            break
    relative_dif =np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])
    logger.info("end; fitting. iternum:%d, relative dif of F: %e\n" % (itercount, relative_dif))
    chi, l2 =loss_f_arr(y_k, *args)
    logger.info("chi:%e, L1:%e, Reg:%e " % (chi,np.sum(y_k), l2))
    return y_k, SOLVED_FLAG_DONE



def only_fx_mfista(init_model, loss_f_arr, grad_f, loss_g = zero_func, eta=1.1, L_init = 1, iter_max = 1000, \
    iter_min = 300, mask_positive = True, stop_ratio = 1e-10, restart = True,  *args):

    (obs, noise, model_prior, lambda_l2, dx, dy) = args
    tk = 1
    x_k = init_model
    y_k = init_model
    x_prev = init_model
    L=L_init
    
    loss_f = lambda x, *args: np.sum(loss_f_arr(x, *args))
    F_xk_arr = [] 
    fig, ax = plt.subplots(1, 1)

    ## Loop for mfista
    itercount = 0

    while(1):
        

        ## loop for L
        f_grad_yk = grad_f(y_k, *args)
        loss_f_yk = loss_f(y_k, *args)
        L = L/eta 
        #print(itercount, tk)
        while(1):
            logger.debug("itercount: %d, L: %e, eta:%e, tk%f" % (itercount, L, eta, tk))

            if L > MAX_L:
                if ETA_MIN < eta:
                    L = L_init
                    eta = eta**0.5
                    iterncount = 0
                    continue
                    
                else:
                    logger.error("Too large L!! Make l1 or l2 regularization parameters small!!")
                    logger.info("end fitting: itercount: %d" % itercount)
                    return y_k, SOLVED_FLAG_ER 


            z_temp = y_k -(1/L) *f_grad_yk
            if mask_positive:
                z_temp[z_temp<0] = 0

            Q_temp = calc_Q(z_temp, y_k, loss_f,loss_g, f_grad_yk, L, *args)
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
                plt.savefig(FIG_FOLDER + "fitting_curve_%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
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
            line, = ax.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
            plt.pause(0.001)
            line.remove()
        if itercount > iter_max:
            plt.plot(range(len(F_xk_arr)), F_xk_arr, color='blue')
            plt.savefig(FIG_FOLDER + "fitting_curve_%s.pdf" % (loss_f_arr.__name__), bbox_inches='tight')
            plt.close()
            break
    relative_dif =np.abs(np.log10(F_xk_arr[len(F_xk_arr)-2]) - np.log10(F_xk_arr[len(F_xk_arr)-1]))/np.log10(F_xk_arr[len(F_xk_arr)-1])
    logger.info("end; fitting. iternum:%d, relative dif of F: %e" % (itercount, relative_dif))


    return y_k, SOLVED_FLAG_DONE

## Optimizers

def adam(init_model, loss_f, grad_f, alpha = 0.001, alpha_1 = 0.9, alpha_2 = 0.999, epsilon = 1e-10, iter_max = 1000, *args):

    M_prev = np.zeros(np.shape(init_model))
    R_prev = np.zeros(np.shape(init_model))
    model_prev = init_model

    for i in range(1, iter_max):
        M = alpha_1 * M_prev + ( 1-alpha_1) * grad_f(model_prev)
        R = alpha_2 * R_prev + (1- alpha_2) * grad_f(model_prev)**2
        M_hat = M/(1- alpha_1 ** i)
        R_hat = R/(1-alpha_2 **i)
        model_prev = model_prev -alpha * M_hat/(np.sqrt(np.sum(R_hat**2)) + epsilon)
    return model_prev


    
## Optimization method insetad of FISTA
def steepest_method(init_model, init_alpha, ryo = 0.7,  c1=0.7, eps_stop = 1e-10, iter_max = 1000, loss_name = "tsv", *args):
    (obs, model_prior, lambda_l2, dx, dy) = args
    model = init_model
    alpha = init_alpha
    next_start_alpha = init_alpha
    
    iter_num = 0

    if loss_name == "tsv":
        gradient_function = grad_loss_tsv
        loss_function = loss_function_TSV
        loss_function_arr = loss_function_arr_TSV
        grad_loss_arr = grad_loss_arr_TSV

    if loss_name == "l2":
        gradient_function = grad_loss_l2
        loss_function = loss_function_l2
        loss_function_arr = loss_function_arr_l2
        grad_loss_arr = grad_loss_arr_l2

    
    while(1):
        gradient = gradient_function(model, *args)
        loss_1, loss_2 = loss_function_arr(model, *args)
        grad_1, grad_2 = grad_loss_arr(model, *args)
        alpha = next_start_alpha

            
        if np.linalg.norm(gradient) < eps_stop:
            logger.info("Iteration_stop")
            break


        while (1):
            model_new = model - alpha * gradient/np.linalg.norm(gradient)
            if alpha < init_alpha*1e-10:
                break
            if loss_function(model_new, *args) < loss_function(model, *args) - c1 *alpha* np.linalg.norm(gradient):
                break
            else:
                alpha = alpha * ryo
                
        model_prev =model
        model = model_new
        next_start_alpha = alpha /(ryo**3)

        iter_num+=1

        if np.sqrt(np.mean( (model - model_prev)**2))/np.sqrt(np.mean( (model)**2))< eps_stop:
            break
        if iter_max < iter_num:
            break
    
    return model

def grad_check( grad_loss_numerical, graidient_function_arr, *args):
    (model_prior, model_prior2, vis_obs, l2_lambada, i_test, j_test) = args
    i_test, j_test = 2, 3
    dF_dmodel, dl2_dmodel  = gradient_function_arr(model_prior, vis_obs, model_prior2,l2_lambada)
    dF_dmodel_num, dl2_dmodel_num  = grad_loss_numerical(model_prior,i_test, j_test, vis_obs, model_prior2,l2_lambada)




## Multi-frequency 

def d_L1_norm(model):

    d = np.ones(np.shape(model))
    d[d==0] = 0
    d[d<0] = 0

    return d

def bound_make(ndim, positive = True):

    bounds = []

    for i in range(ndim):
        if positive:
            bounds.append([0, np.inf])

    for i in range(ndim):
            bounds.append([0, 5])

    return np.array(bounds)

def x_to_I_alpha(x_vec,reverse = False):

    if not reverse:

        n1 = np.shape(x_vec)[0]
        model_image = x_vec[0:int(n1/2)]
        model_alpha = x_vec[int(n1/2):n1]
        model_image = np.reshape(model_image,(int(np.sqrt(n1/2)), int(np.sqrt(n1/2))))
        model_alpha = np.reshape(model_alpha,(int(np.sqrt(n1/2)), int(np.sqrt(n1/2))))

    else:

        x_image = np.ravel(x_vec[0])
        x_alpha = np.ravel(x_vec[1])
        n1 = 2 * np.shape(x_image)[0]

        return np.append(x_image, x_alpha)

    return model_image, model_alpha

def set_bounds(Nx, alpha_max=4, set_alpha_zero_at_edge = True, flux_max=np.inf):

    image_bd = []
    alpha_bd = [] 
    nx, ny = int(Nx**0.5),int(Nx**0.5) 

    for i in range(nx):
        for j in range(ny):
            image_bd.append([-1*10**(-9), + flux_max])

            if set_alpha_zero_at_edge and (i==0 or j == 0 or i ==nx-1 or j == ny-1):
                alpha_bd.append([0,0])
            else:
                alpha_bd.append([-1,alpha_max])

    return np.append(image_bd, alpha_bd, axis =0)

def edge_zero(image, flag_2d = True):

    if flag_2d:
        nx, ny = np.shape(image)
        image[0,:] = 0
        image[nx-1,:] = 0
        image[:,0] = 0
        image[0,ny-1] = 0
    else:
        nx = int((len(image)**0.5))

        for i in range(nx):
            for j in range(nx):

                if i==0 or j == 0 or i ==nx-1 or j == nx-1:
                    image[i + nx * j] = 0

    return image


def call_back(x_vec):
    global COUNT
    COUNT += 1
    #print(COUNT)
    logger.debug(COUNT)
    return None


def alpha_to_zero_w_I(x_vec):

    N = int(len(x_vec)/2)
    x_vec[N:2 * N][x_vec[0:N] ==0] = 0

    return x_vec

def alpha_to_zero_w_I_df(df_dx, image):

    N = int(len(image)/2)
    df_dx[N:2 * N][image[0:N] ==0] = 0

    return df_dx

def multi_freq_grad(x_vec, *args):

    #x_vec = alpha_to_zero_w_I(x_vec)

    ## Load
    obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, alpha_reg, alpha_prior, dx, dy  = args 
    n_freq = len(nu_arr)
    model_image, model_alpha= x_to_I_alpha(x_vec)
    nx, ny = np.shape(model_image)

    ## Main
    d_TSV_I = lambda2 * d_TSV(model_image)
    d_L1_norm_I = lambda1 * d_L1_norm(model_image)

    model_alpha[model_image ==0] = 0
    if lambda_alpha_2 is None:
        d_alpha_reg = np.zeros(np.shape(d_L1_norm_I))

    elif alpha_reg == "TSV":
        d_alpha_reg = lambda_alpha_2 * d_TSV(model_alpha)
        #d_alpha_reg[model_image==0] = 0

    elif alpha_reg == "L2":
        d_alpha_reg = 2 * lambda_alpha_2 * (model_alpha-alpha_prior)
        #d_alpha_reg[model_image==0] = 0

    d_reg_sum = x_to_I_alpha([d_L1_norm_I  + d_TSV_I, d_alpha_reg], reverse = True)
    d_chi_sum = multi_freq_chi2_grad(x_vec, obs, noise, nu_arr, nu0, dx, dy)
    d_tot = d_chi_sum + d_reg_sum
    #d_tot = alpha_to_zero_w_I_df(d_tot, x_vec)

    return d_tot


def multi_freq_chi2_grad(x_vec, obs, noise, nu_arr, nu0,dx, dy):

    ## Load
    n_freq = len(nu_arr)
    model_image, model_alpha= x_to_I_alpha(x_vec)
    nx, ny = np.shape(model_image)
    d_chi_d_I = np.zeros(np.shape(model_image))
    d_chi_d_alpha = np.zeros(np.shape(model_image))

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

        d_chi_d_alpha  += - np.real(2 *model_freqj_for_dalpha * B_ifft_A_F)
        d_chi_d_I  += - np.real(2 *sj_model * B_ifft_A_F)

    #d_chi_d_alpha[model_image==0] = 0
    return x_to_I_alpha([d_chi_d_I, d_chi_d_alpha], reverse = True)



def multi_freq_chi2(x_vec, obs, noise, nu_arr, nu0, dx, dy):

    ## Load
    n_freq = len(nu_arr)
    model_image, model_alpha= x_to_I_alpha(x_vec)
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

    return chi_sum


def multi_freq_cost_l1_tsv(x_vec, *args):

    #x_vec = alpha_to_zero_w_I(x_vec)
    obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, alpha_reg, alpha_prior, dx, dy  = args 
    model_image, model_alpha = x_to_I_alpha(x_vec)
    chi2 = multi_freq_chi2(x_vec, obs, noise, nu_arr, nu0, dx, dy)
    model_alpha[model_image ==0] = 0

    if lambda_alpha_2 is None:
        alpha_cost = 0

    elif alpha_reg == "TSV":
        alpha_cost = lambda_alpha_2 * TSV(model_alpha)

    elif alpha_reg == "L2":
        alpha_cost = lambda_alpha_2 * np.sum((model_alpha-alpha_prior)**2)

    cost_sum = chi2 + lambda1 * L1_norm(model_image) + lambda2* TSV(model_image) + alpha_cost
    logger.debug("mfreq solving: chi2:%e, l1:%e, TSV:%e, alpha:%e, sum:%e" % \
        (chi2,  L1_norm(model_image), TSV(model_image), alpha_cost, cost_sum))
    return cost_sum



def grad_mfreq_numerical(x_vec,  *args):

    obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, dx, dy   = args
    delta = 1e-10
    model_image, model_alpha = x_to_I_alpha(x_vec)
    chi2 = multi_freq_chi2(x_vec, obs, noise, nu_arr, nu0, dx, dy)
    chi_sum = chi2 + lambda1 * L1_norm(model_image) + lambda2* TSV(model_image) + lambda_alpha_2 * TSV(model_alpha)  
    num_grad = np.ones(len(x_vec))

    for i in range(len(x_vec)):
        x_vec_delta = np.copy(x_vec)
        x_vec_delta[i] += delta
        model_image, model_alpha = x_to_I_alpha(x_vec_delta)
        chi2 = multi_freq_chi2(x_vec_delta, obs, noise, nu_arr, nu0, dx, dy)
        chi_sum_delta = chi2 + lambda1 * L1_norm(model_image) + lambda2* TSV(model_image)+ lambda_alpha_2 * TSV(model_alpha)
        num_grad[i] = (chi_sum_delta - chi_sum)/delta 

    return num_grad 


def solver_mfreq(x_init, obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, dx,dy, func = multi_freq_cost_l1_tsv, \
    f_grad= multi_freq_grad, alpha_reg="TSV", maxiter = 15000, factr = 10000000.0, call_back = call_back):

    nx = int(np.sqrt(len(x_init)/2.0))
    alpha_prior = 0 + np.zeros((nx, nx))
    x_init = edge_zero(x_init, flag_2d =False)
    bounds = set_bounds(int(nx * nx), alpha_max=np.inf , set_alpha_zero_at_edge =False)
    args = (obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, alpha_reg, alpha_prior, dx, dy)
    print(func(x_init, *args))

    result = optimize.fmin_l_bfgs_b(func, x_init, args = args, fprime = f_grad, \
            bounds = bounds, maxiter = maxiter, factr = factr, callback = call_back)

    image, alpha = x_to_I_alpha(result[0])
    print("chi:%e,L1:%e, TSV:%e, TSV(alpha):%e" %(multi_freq_chi2(result[0], obs, noise, nu_arr, nu0, dx, dy), \
        np.sum(image), TSV(image), TSV(alpha)))
    return image, alpha

def lambda_arrs_make(lambda1, lambda2, lambda_alpha_2, nterms=3, spacing = 1):
    log10_lambda1 =  round(np.log10(lambda1))
    log10_lambda2 =  round(np.log10(lambda2))
    log10_lambda_alpha_2 =  round(np.log10(lambda_alpha_2))
    if nterms %2 == 0:
        nterms = nterms+1
    del_mins_plus = int((nterms-1)/2.0)
    lambda1_arr = 10**np.linspace(log10_lambda1-del_mins_plus*spacing, log10_lambda1+del_mins_plus*spacing, num = nterms, endpoint = True) 
    lambda2_arr = 10**np.linspace(log10_lambda2-del_mins_plus*spacing, log10_lambda2+del_mins_plus*spacing, num = nterms, endpoint = True) 
    lambda_alpha_2_arr = 10**np.linspace(log10_lambda_alpha_2-del_mins_plus*spacing, log10_lambda_alpha_2+del_mins_plus*spacing, num = nterms, endpoint = True) 

    return lambda1_arr, lambda2_arr,lambda_alpha_2_arr

def solver_mfreq_for_loop(x_init, obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, dx,dy, f = multi_freq_cost_l1_tsv, \
    f_grad= multi_freq_grad, alpha_reg="TSV", maxiter = 15000, factr = 10000000.0, call_back = call_back):

    alpha_prior = 0 + np.zeros(np.shape(alpha))
    x_init = edge_zero(x_init, flag_2d =False)
    bounds = set_bounds(N_tot, alpha_max=np.inf , set_alpha_zero_at_edge =False)
    args = (obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, alpha_reg, alpha_prior, dx, dy)
    

    result = optimize.fmin_l_bfgs_b(f, x_init, args = args, fprime = f_grad, \
            bounds = bounds, maxiter = maxiter, factr = factr, callback = call_back)

    image, alpha = x_to_I_alpha(result[0])
    print("chi:%e,L1:%e, TSV:%e, TSV(alpha):%e" %(multi_freq_chi2(result[0], obs, noise, nu_arr, nu0, dx, dy), \
        np.sum(image), TSV(image), TSV(alpha)))
    return image, alpha
def solver_mfreq_for_wrapper(args):
    return solver_mfreq_for_loop(*args)

def solver_mfreq_several_reg_params(x_init, obs, noise, nu_arr, nu0, lambda1_pre, lambda2_pre, lambda_alpha_2_pre, dx,dy, f = multi_freq_cost_l1_tsv, \
    f_grad= multi_freq_grad, alpha_reg="TSV", maxiter = 15000, factr = 10000000.0, call_back = call_back):
    
    lambda1_arr, lambda2_arr,lambda_alpha_2_arr = lambda_arrs_make(lambda1_pre, lambda2_pre, lambda_alpha_2_pre, nterms=3)
    nx = int(np.sqrt(len(x_init)/2.0))

    image_I_arrs = np.zeros((len(lambda1_arr), len(lambda2_arr), len(lambda_alpha_2_arr), nx ,nx))
    image_alpha_arrs = np.zeros((len(lambda1_arr), len(lambda2_arr), len(lambda_alpha_2_arr), nx ,nx))
    print(lambda1_arr, lambda2_arr,lambda_alpha_2_arr)
    p = Pool(processes=2)

    for (i, lambda1) in enumerate(lambda1_arr):
        for (j, lambda2) in enumerate(lambda2_arr):
            for (k, lambda_alpha_2) in enumerate(lambda_alpha_2_arr):
                print(i,j,k)
                #solver_mfreq_for_loop
                #p.map(wrapper_kakezan, tutumimono) 
                """
                alpha_prior = 0 + np.zeros((nx, nx))
                x_init = edge_zero(x_init, flag_2d =False)
                bounds = set_bounds(int(nx * nx), alpha_max=np.inf , set_alpha_zero_at_edge =False)
                args = (obs, noise, nu_arr, nu0, lambda1, lambda2, lambda_alpha_2, alpha_reg, alpha_prior, dx, dy)
                result = optimize.fmin_l_bfgs_b(f, x_init, args = args, fprime = f_grad, \
                bounds = bounds, maxiter = maxiter, factr = factr, callback = call_back)
                image, alpha = x_to_I_alpha(result[0])
                """
                image_I_arrs[i,j,k] = image
                image_alpha_arrs[i,j,k] = alpha

    return image_I_arrs, image_alpha_arrs 


def solver_mfreq_independent(loss, grad, l1_func, vis_obs,  noise, nu_arr, n0, l1_lambda, l2_lambda, dx, dy, xnum, ynum, alpha_def = None, positive_solve =True, percentile = 10):

    nfreq, nx, ny = np.shape(vis_obs)
    model_freqs = np.zeros((nfreq, xnum, ynum))

    for i in range(nfreq):
        init_model = np.zeros((xnum, ynum))
        image, solved = fx_L1_mfista(init_model, loss, 
            grad, l1_func, ETA_INIT, L_INIT, MAXITE, MINITE, positive_solve ,\
            STOP_RATIO,RESTART, vis_obs[i], noise[i], l1_lambda, l2_lambda, dx, dy)

        model_freqs[i] = image


    freq_log = np.log(nu_arr/n0)
    alpha = np.zeros((xnum, ynum))
    image_0 = np.zeros((xnum, ynum))
    max_emission = np.max(model_freqs)


    sum_flux = np.sum(model_freqs, axis=(1,2))
    alpha_rough_est, alpha_rough_flux = np.polyfit(freq_log, np.log(sum_flux), 1)

    if alpha_def is not None:
        alpha_rough_est =  alpha_def

    for i in range(xnum):
        for j in range(ynum):
            int_freq = model_freqs[:,i,j]

            if int_freq.prod() > 0:

                b, a = np.polyfit(freq_log, np.log(int_freq), 1)

                if not (np.isfinite(b) and np.isfinite(a)):
                    image_0[i,j] = 0
                    alpha[i,j] = 0
                    continue

                ## Remove Unrealsitic high I_0

                if np.exp(a) > 1 * max_emission:
                    continue
                image_0[i,j] = np.exp(a)
                if b > 2 * alpha_rough_est:
                    b= 2*alpha_rough_est
                if b<0:
                    b = 0

                alpha[i,j] = b

    return image_0, alpha, model_freqs


def determine_regularization_scaling_from_clean(num_mat_freq, clean_image0, clean_alpha):
    n_data = len(num_mat_freq[num_mat_freq!=0])
    image_max = np.max(clean_image0)
    factor = 1/image_max
    clean_image0_normlized = clean_image0 * factor
    l1_norm = L1_norm(clean_image0_normlized)
    TSV_term = TSV(clean_image0_normlized)
    alpha_TSV = TSV(clean_alpha)
    lambda_l0 = n_data/l1_norm
    lambda_tsv = n_data/TSV_term
    lambda_alpha = n_data/alpha_TSV
    return lambda_l0, lambda_tsv, lambda_alpha, factor


