import numpy as np
import matplotlib.pyplot as plt

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
    (obs, model_prior, lambda_l2) = args
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis- obs
    d_vis[obs_mask] = 0
    return np.sum(np.abs(d_vis)**2) +  lambda_l2 * np.sum((model-model_prior)**2)

def loss_function_TSV(model, *args):
    (obs, model_prior, lambda_ltsv) = args
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis- obs
    d_vis[obs_mask] = 0
    return np.sum(np.abs(d_vis)**2) +  lambda_ltsv * TSV(model)

def loss_function_arr_l2(model, *args):
    (obs, model_prior, lambda_l2) = args
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis- obs
    d_vis[obs_mask] = 0
    return np.sum(np.abs(d_vis)**2), lambda_l2 * np.sum((model-model_prior)**2)

def loss_function_arr_TSV(model, *args):
    (obs, model_prior, lambda_ltsv) = args
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis- obs
    d_vis[obs_mask] = 0
    return np.sum(np.abs(d_vis)**2), lambda_ltsv * TSV(model)


def grad_loss_arr_l2(model, *args):
    
    (obs, model_prior, lambda_l2) = args

    ## Gradient for L2
    dl2_dmodel =2 * lambda_l2 * (model-model_prior)
    nx, ny = np.shape(model)
    
    ## Gradient for chi^2
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis - obs
    d_vis[obs_mask] = 0
    #vis_F = np.fft.ifftshift(d_vis)
    ifft_F = np.fft.ifft2(d_vis)
    dF_dmodel =2*    ifft_F.real 
    return dF_dmodel, dl2_dmodel 

def grad_loss_l2(model, *args):
    
    (obs, model_prior, lambda_l2) = args

    ## Gradient for L2
    dl2_dmodel =2 * lambda_l2 * (model-model_prior)
    nx, ny = np.shape(model)
    
    
    ## Gradient for chi^2
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis - obs
    d_vis[obs_mask] = 0
    #vis_F = np.fft.ifftshift(d_vis)
    ifft_F = np.fft.ifft2(d_vis)
    dF_dmodel =2* nx * ny* ifft_F.real 
    return dF_dmodel + dl2_dmodel

def grad_loss_tsv(model, *args):
    
    (obs, model_prior, lambda_ltsv) = args

    ## Gradient for L2
    nx, ny = np.shape(model)
    
    
    ## Gradient for chi^2
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis - obs
    d_vis[obs_mask] = 0
    #vis_F = np.fft.ifftshift(d_vis)
    ifft_F = np.fft.ifft2(d_vis)
    dF_dmodel =2* nx * ny* ifft_F.real 
    return dF_dmodel + lambda_ltsv * d_TSV(model)

def grad_loss_arr_TSV(model, *args):
    
    (obs, model_prior, lambda_ltsv) = args

    ## Gradient for L2
    nx, ny = np.shape(model)
    
    
    ## Gradient for chi^2
    #model_vis = np.fft.fftshift(np.fft.fft2(model))
    model_vis = np.fft.fft2(model)
    obs_mask = (obs == 0)
    d_vis = model_vis - obs
    d_vis[obs_mask] = 0
    #vis_F = np.fft.ifftshift(d_vis)
    ifft_F = np.fft.ifft2(d_vis)
    dF_dmodel =2* nx * ny * ifft_F.real 
    return dF_dmodel, lambda_ltsv * d_TSV(model)

def grad_loss_numerical_l2(model,i,j, *args):
    (obs, model_prior, lambda_l2) = args

    model_new = np.copy(model) 
    eps = 1e-5
    model_new[i][j] += eps
    chi_new, l2_new = loss_function_arr_l2(model_new, *args)
    chi_, l2_ = loss_function_arr_l2(model, *args)
    return (chi_new-chi_)/eps, (l2_new-l2_)/eps

    

def steepest_method(init_model, init_alpha, ryo = 0.7,  c1=0.7, eps_stop = 1e-10, iter_max = 1000, loss_name = "tsv", *args):
    (obs, model_prior, lambda_l2) = args
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
            print("Iteration_stop")
            break


        while (1):
            model_new = model - alpha * gradient/np.linalg.norm(gradient)
            if alpha < init_alpha*1e-10:
                break
            #print(alpha)
            #print(loss_function(model, *args) , loss_function(model_new, *args) , loss_function(model, *args) - c1 *alpha* np.linalg.norm(gradient))
            if loss_function(model_new, *args) < loss_function(model, *args) - c1 *alpha* np.linalg.norm(gradient):
                break
            else:
                alpha = alpha * ryo
                
        model_prev =model
        model = model_new
        next_start_alpha = alpha /(ryo**3)
        if iter_num %10==0:
            print(iter_num, np.linalg.norm(grad_1), np.linalg.norm(grad_2),alpha)
            print(iter_num, loss_1, loss_2/lambda_l2, loss_1+ loss_2, alpha)
        iter_num+=1
        #print(iter_num, np.sqrt(np.mean( (model - model_prev)**2))/np.sqrt(np.mean( (model)**2)))
        #print(iter_num, loss_function_arr(model_new, *args))

        if np.sqrt(np.mean( (model - model_prev)**2))/np.sqrt(np.mean( (model)**2))< eps_stop:
            break
        if iter_max < iter_num:
            break
    
    return model

def zero_func(model, *args):
    
    return 0

def L1_norm(model, *args):
    (obs, lambda_l1, lambda_l2) = args

    return lambda_l1 * np.sum(np.abs(model))

def calc_Q(x, y, f, g, grad_f, L, *args):
    term1 = f(y, *args)
    term2 = np.sum((x-y)*grad_f)
    term3 = (L/2) * np.sum((x-y)**2)
    term4 = g(x, *args)

    return term1 + term2 + term3 + term4

def soft_threshold(model, lambda_t, print_flag=False):

    mask_temp1 = model> lambda_t
    mask_temp2 = (model> -lambda_t) * (model< lambda_t)
    mask_temp3 = model< - lambda_t
    model[mask_temp1] -= lambda_t
    model[mask_temp2] =0
    model[mask_temp3] += lambda_t
    if print_flag:
        print(len(model[mask_temp1]), len(model[mask_temp2]),len(model[mask_temp3]),lambda_t)
    return model

def positive_soft_threshold(model, lambda_t, print_flag=False):

    mask_temp1 = model> lambda_t
    mask_temp2 = model< lambda_t
    model[mask_temp1] -= lambda_t
    model[mask_temp2] =0
    if print_flag:
        print(len(model[mask_temp1]), len(model[mask_temp2]),len(model[mask_temp3]),lambda_t)
    return model


def fx_L1_mfista(init_model, loss_f, grad_f, loss_g = zero_func, eta=1.1, L_init = 1, iter_max = 1000, mask_positive = True, *args):

    (obs, lambda_l1, lambda_l2) = args
    tk = 1
    x_k=init_model
    y_k = init_model
    x_prev = init_model
    L=L_init

    ## Loop for mfista
    itercount = 0
    while(1):
        

        ## loop for L
        f_grad_yk = grad_f(y_k, *args)
        L = L/eta

        while(1):
            if mask_positive:
                z_temp = positive_soft_threshold(y_k -(1/L) *f_grad_yk,lambda_l1 *(1/L)) 
            else:
                z_temp = soft_threshold(y_k -(1/L) *f_grad_yk,lambda_l1 *(1/L)) 
            Q_temp = calc_Q(z_temp, y_k, loss_f,L1_norm, f_grad_yk, L, *args)
            F_temp = loss_f(z_temp, *args) + L1_norm(z_temp, *args)
            if F_temp < Q_temp:
                break
            else:
                L = L * eta


        if itercount%50==0:
            soft_threshold(z_temp, lambda_l1 *(1/L), print_flag=True)


        z_k = z_temp
        tk_1 = (1 + np.sqrt(1+4*tk**2))/2
        F_xk = loss_f(x_k, *args) + L1_norm(x_k, *args)
        F_zk = loss_f(z_k, *args) + L1_norm(z_k, *args)

        if F_xk > F_zk:
            x_k = z_k
            y_k = x_k + ((tk-1)/tk_1) * (x_k - x_prev)
            x_prev = x_k
        else:
            x_k = x_prev
            y_k = x_k + (tk/tk_1)*(z_k-x_k)
        itercount+=1
        if itercount%50==0:
            print("iternum:",itercount, ", L:", L, "F_zk:", np.log10(F_xk))
        if itercount > iter_max:
            break

    return y_k



def only_fx_mfista(init_model, loss_f, grad_f, loss_g = zero_func, eta=1.1, L_init = 1, iter_max = 1000, mask_positive = True, *args):

    (obs, model_prior, lambda_l2) = args
    tk = 1
    x_k=init_model
    y_k = init_model
    x_prev = init_model
    L=L_init

    ## Loop for mfista
    itercount = 0
    while(1):
        

        ## loop for L
        f_grad_yk = grad_f(y_k, *args)
        loss_f_yk = loss_f(y_k, *args)

        while(1):

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

        if F_xk > F_zk:
            x_k = z_k
            y_k = x_k + ((tk-1)/tk_1) * (x_k - x_prev)
            x_prev = x_k
        else:
            x_k = x_prev
            y_k = x_k + (tk/tk_1)*(z_k-x_k)
        itercount+=1
        if itercount%50==0:
            print("iternum:",itercount, ", L:", L, "F_zk:", np.log10(F_xk))
        if itercount > iter_max:
            break

    return y_k



def adam(init_model, loss_f, grad_f, alpha = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-10, iter_max = 1000, *args):

    M_prev = np.zeros(np.shape(init_model))
    R_prev = np.zeros(np.shape(init_model))
    model_prev = init_model

    for i in range(1, iter_max):
        M = beta_1 * M_prev + ( 1-beta_1) * grad_f(model_prev)
        R = beta_2 * R_prev + (1- beta_2) * grad_f(model_prev)**2
        M_hat = M/(1- beta_1 ** i)
        R_hat = R/(1-beta_2 **i)
        model_prev = model_prev -alpha * M_hat/(np.sqrt(np.sum(R_hat**2)) + epsilon)
    return model_prev



def grad_check( grad_loss_numerical, graidient_function_arr, *args):
    (model_prior, model_prior2, vis_obs, l2_lambada, i_test, j_test) = args
    i_test, j_test = 2, 3
    dF_dmodel, dl2_dmodel  = gradient_function_arr(model_prior, vis_obs, model_prior2,l2_lambada)
    dF_dmodel_num, dl2_dmodel_num  = grad_loss_numerical(model_prior,i_test, j_test, vis_obs, model_prior2,l2_lambada)
    print(dF_dmodel[i_test, j_test], dl2_dmodel[i_test, j_test])
    print(dF_dmodel_num, dl2_dmodel_num)
    print(dF_dmodel_num/dF_dmodel[i_test, j_test])

