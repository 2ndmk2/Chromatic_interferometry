import shutil
import numpy as np
import os
import sys
sys.path.insert(0,'../config')
from setting_freq_common import *
from setting_freq_image import *
import utility

out_folder ="./vis_sim"
clean_folder ="./clean_result"
clean_name = "try"
nu_arr = NU_OBS
nu0 = NU0
folder_names = utility.make_folder_names(nu_arr, clean_folder, out_folder)
fits_files = utility.make_fits_names(nu_arr)

utility.rm_folders(folder_names)
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

vis_files = utility.make_vis_noise_names(nu_arr)

tclean(vis = vis_files, 
	imagename = "./%s/%s" % (clean_folder,clean_name)
	imsize = [axesLength[0], axesLength[1]],
	cell=cell,
	niter = 2000,
	threshold = "1e-7Jy", 
	weighting = "natural",
	specmode="mfs", 
	deconvolver="mtmfs",
	nterms = 2,
	reffreq="%dGHz" % nu0 
	)

utility.exportfits_images(clean_folder,clean_name, nterm = len(nu_arr))
utility.clean_to_images_and_save(clean_folder, clean_name, nterm, nu_arr, nu0):
vis_Files = utility.make_vis_names(nu_arr)
utility.ms_to_visfile(vis_Files, out_folder, nu0=nu0, nu_arr = nu_arr)
