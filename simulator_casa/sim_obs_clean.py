import shutil
import numpy as np
import os
import sys
sys.path.insert(0,'../config')
from setting_freq_common import *
from setting_freq_image import *
import utility_clean

def exportfits_images(clean_folder,clean_name, nterm = 2):
	exportfits(imagename="./%s/%s.alpha" % (clean_folder, clean_name), fitsimage="./%s/%s.alpha.fits" % (clean_folder, clean_name),
	overwrite = True, history=False) 
	for i in range(nterm):
		exportfits(imagename="./%s/%s.image.tt%d" % (clean_folder, clean_name,i), \
			fitsimage="./%s/%s.image.tt%d.fits" % (clean_folder, clean_name,i), overwrite = True, history=False) 

def ms_to_visfile(files, out_folder, nu_arr, nu0 = 0):
	
	for (i, file_name) in enumerate(files):

		file_name_freq = file_name + "/SPECTRAL_WINDOW"
		tb.open(file_name_freq)
		c_const = 299792458.0 * 1e3/1e9#mm Ghz
		nu_now = tb.getcol("CHAN_FREQ")[0][0]/1e9
		lambda_now = c_const/nu_now

		tb.open(file_name)
 
		sigma=tb.getcol('SIGMA');  
		sigma_x = sigma[0]
		sigma_y = sigma[1]

		data=tb.getcol('DATA');  
		data = data[:,0,:]
		real_data_x = data[0].real ##pol 1
		imag_data_x = data[0].imag
		real_data_y = data[1].real ## pol 2
		imag_data_y = data[1].imag

		uvw=tb.getcol('UVW');  
		u = 1e3 * uvw[0]/lambda_now
		v = 1e3 * uvw[1]/lambda_now

		len_uv = len(u)
		file_name = file_name.split("/")
		file_name = file_name[-1]
		file_name = file_name.replace("ms", "csv")
		file_name = os.path.join(out_folder, file_name)
		file_out = open(file_name, "w")
		for i in range(len_uv):
		    file_out.write("%.7f, %.7f, %.7f, %.7f, %.7f, %.7f, %.7f, %.7f, %.7f, %.7f\n" % (u[i], v[i], real_data_x[i], \
		                                                                   imag_data_x[i], real_data_y[i], imag_data_y[i], 
		                                                                   0.5 * real_data_x[i] + 0.5 * real_data_y[i], 
		                                                                   0.5 * imag_data_x[i] + 0.5 * imag_data_y[i], 
		                                                                   sigma[0][i], sigma[1][i]))
		file_out.close()
		tb.close() 

	file_name = os.path.join(out_folder, "vis_file_freqs")
	np.savez(file_name, nu_arr = nu_arr, nu0 = nu0)

	return None

out_folder ="./vis_sim"
clean_folder ="./clean_result"
clean_name = "try"
nu_arr = NU_OBS
nu0 = NU0
folder_names = utility_clean.make_folder_names(nu_arr, clean_folder, out_folder)
fits_files = utility_clean.make_fits_names(nu_arr)
utility_clean.rm_folders(folder_names)

if os.path.exists(out_folder):
	shutil.rmtree(out_folder)
	os.mkdir(out_folder)
else:
	os.mkdir(out_folder)

## open files
for i in range(len(fits_files)):
	default("simobserve")
	file_name_fit = fits_files[i]

	ia.open(file_name_fit)
	axesLength = ia.shape()
	# Divide the first two elements of axesLength by 2.
	center_pixel = [ x / 2.0 for x in axesLength[:2] ]
	# Feed center_pixel to ia.toworld and and save the RA and Dec to ra_radians and dec_radians
	(ra_radians, dec_radians) = ia.toworld( center_pixel )['numeric'][:2]
	ia.close()

	ra_hms  = qa.formxxx(str(ra_radians)+"rad",format='hms',prec=5)
	dec_dms = qa.formxxx(str(dec_radians)+"rad",format='dms',prec=5)
	noise_mod = "tsys-atm"
	cell = "%.3farcsec" % DX_PIX
	direction_now = "J2000 %s %s" % (ra_hms, dec_dms)

	project_name = "psim_freq%d" % nu_arr[i]

	simobserve(project = project_name, \
		skymodel = fits_files[i], \
		direction = direction_now, \
		mapsize =  "10.0arcsec", \
		obsmode = "int", \
		totaltime = "3600s", \
		incenter="%dGHz" % nu_arr[i], \
		inwidth="200MHz", \
		incell=cell, \
		integration = "60s", \
		antennalist = "alma.out20.cfg",\
		pointingspacing  =  "100arcsec", \
		thermalnoise = noise_mod)

vis_files = utility_clean.make_vis_noise_names(nu_arr)

tclean(vis = vis_files, 
	imagename = "./%s/%s" % (clean_folder,clean_name),
	imsize = [axesLength[0], axesLength[1]],
	cell=cell,
	niter = 2000,
	threshold = "1e-7Jy", 
	weighting = "natural",
	specmode="mfs", 
	deconvolver="mtmfs",
	nterms = len(nu_arr),
	reffreq="%dGHz" % nu0 
	)

exportfits_images(clean_folder,clean_name, nterm = len(nu_arr))
vis_Files = utility_clean.make_vis_names(nu_arr)
ms_to_visfile(vis_Files, out_folder, nu0=nu0, nu_arr = nu_arr)





