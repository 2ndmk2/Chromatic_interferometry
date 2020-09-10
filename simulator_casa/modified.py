import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.insert(0,'../config')
from setting_freq_common import *
from setting_freq_image import *


hdulist=pyfits.open('./ppdisk672_GHz_50pc.fits')
hdu=hdulist[0]
data=hdu.data
header = hdu.header
hdu.header["NAXIS1"] = NX
hdu.header["NAXIS2"] = NY
hdu.header["CDELT1"] = DX_PIX/206265.0
hdu.header["CDELT2"] = DX_PIX/206265.0
if not os.path.exists("fits_modified"):
	os.makedirs("fits_modified")


input_images = np.load(os.path.join(FOLDER_pre,"input_model.npz"))
nu_arr = input_images["nu_arr"]
images = input_images["model"]

for i in range(len(nu_arr)):
	input_model_now = images[i]
	NX, NY = np.shape(input_model_now)
	new_model= flux_factor * np.fliplr(input_model_now)
	new_model = new_model[np.newaxis,np.newaxis,:,:]

	hdu.data= new_model
	hdu.writeto('./fits_modified/my_image_freq%d.fits' % nu_arr[i], overwrite=True)



