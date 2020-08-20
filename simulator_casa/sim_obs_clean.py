import shutil
import numpy as np
import os
import sys
sys.path.insert(0,'../config')
from setting_freq_common import *

def ms_to_visfile(files, out_folder):
	
	nu_arr = []
	for (i, file_name) in enumerate(files):

		file_name_freq = file_name + "/SPECTRAL_WINDOW"
		tb.open(file_name_freq)
		c_const = 299792458.0 * 1e3/1e9#mm Ghz
		nu_now = tb.getcol("CHAN_FREQ")[0][0]/1e9
		nu_arr.append(nu_now)
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
	np.save(file_name, nu_arr)

	return None

def mv_folders(folder_names, move_to_place):
	for name in folder_names:
		shutil.move(name, move_to_place)

def rm_folders(folder_names):
	for name in folder_names:
		if os.path.exists(name):
			shutil.rmtree(name)

out_folder ="./vis_sim"
clean_folder ="./clean_result"
folder_names =[clean_folder, "psim_freq350", "psim_freq250", out_folder, "plot_images"]

rm_folders(folder_names)


default("simobserve")




if os.path.exists(out_folder):
	shutil.rmtree(out_folder)
	os.mkdir(out_folder)
else:
	os.mkdir(out_folder)


# This reports image header parameters in the Log Messages window
fileimage_freq0 = "./fits_modified/my_image_freq0.fits"
fileimage_freq1 = "./fits_modified/my_image_freq1.fits"

## open files
ia.open(fileimage_freq0)

axesLength = ia.shape()
print(axesLength)
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

simobserve(project = "psim_freq350", \
	skymodel = fileimage_freq0, \
	#setpointings = True, \
	#direction = "J2000 18h00m00.031s -22d59m59.6s", \
	#direction = direction_now, \
	# mapsize =  "20.6arcsec", \
	obsmode = "int", \
	totaltime = "3600s", \
	incenter="350.0GHz", \
	inwidth="200MHz", \
	incell=cell, \
	#inbright="0.000055", \
	integration = "60s", \
	antennalist = "alma.out20.cfg",\
	thermalnoise = noise_mod)

simobserve(project = "psim_freq250", \
	skymodel = fileimage_freq1, \
	setpointings = True, \
	#direction = "J2000 18h00m00.031s -22d59m59.6s", \
	#direction = direction_now, \
	#mapsize =  "0.76arcsec", \
	obsmode = "int", \
	totaltime = "3600s", \
	incenter="250.0GHz", \
	inwidth="200MHz", \
	incell=cell, \
	#inbright="0.000055", \
	integration = "60s", \
	antennalist = "alma.out20.cfg",\
	thermalnoise = noise_mod)


vis_files = ["./psim_freq350/psim_freq350.alma.out20.noisy.ms",\
"./psim_freq250/psim_freq250.alma.out20.noisy.ms"]

tclean(vis = vis_files, 
	imagename = "./%s/try" % clean_folder,
	imsize = [axesLength[0], axesLength[1]],
	cell=cell,
	niter = 2000,
	threshold = "1e-7Jy", 
	weighting = "natural",
	specmode="mfs", 
	deconvolver="mtmfs",
	nterms = 2,
	reffreq="300GHz" 
	)

exportfits(imagename="./%s/try.alpha" % clean_folder, fitsimage="./%s/try.alpha.fits" % clean_folder,
	overwrite = True, history=False)  
exportfits(imagename="./%s/try.image.tt0" % clean_folder, fitsimage="./%s/try.image.tt0.fits" % clean_folder, 
	overwrite = True, history=False)  
exportfits(imagename="./%s/try.image.tt1" % clean_folder, fitsimage="./%s/try.image.tt1.fits" % clean_folder, 
	overwrite = True, history=False)  

vis_Files = ["./psim_freq250/psim_freq250.alma.out20.ms", "./psim_freq350/psim_freq350.alma.out20.ms", \
"./psim_freq250/psim_freq250.alma.out20.noisy.ms", "./psim_freq350/psim_freq350.alma.out20.noisy.ms"]

ms_to_visfile(vis_Files, out_folder)
