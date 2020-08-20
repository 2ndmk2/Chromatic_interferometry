import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np
import os

hdulist=pyfits.open('./ppdisk672_GHz_50pc.fits')
hdu=hdulist[0]
data=hdu.data

header = hdu.header
#flux_max = np.max(data)
flux_factor = 1
data2 = np.load("../imaging_color_source/tests/input_model.npz")
input_model = data2["model"][0]
input_model2 = data2["model"][1]
NX, NY = np.shape(input_model)
print(NX, NY)
flux_max_input = np.max(input_model)
new_model= flux_factor * input_model
new_model = new_model[np.newaxis,np.newaxis,:,:]
new_model2= flux_factor * input_model2
new_model2 = new_model2[np.newaxis,np.newaxis,:,:]


hdu.header["NAXIS1"] = NX
hdu.header["NAXIS2"] = NY
hdu.header["CDELT1"] = 0.02/206265.0
hdu.header["CDELT2"] = 0.02/206265.0

if not os.path.exists("fits_modified"):
	os.makedirs("fits_modified")

header=hdu.header
hdu.data= new_model
#if not os.path.exists("my_image.fits"):
hdu.writeto('./fits_modified/my_image_freq0.fits', overwrite=True)

hdu.data= new_model2
#if not os.path.exists("my_image.fits"):
hdu.writeto('./fits_modified/my_image_freq1.fits', overwrite=True)


