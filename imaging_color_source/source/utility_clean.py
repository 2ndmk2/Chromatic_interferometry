import numpy as np
import os
import shutil


def mv_folders(folder_names, move_to_place):
	if not os.path.exists(move_to_place):
		os.makedirs(move_to_place)
		
	for name in folder_names:
		path_new = os.path.join(move_to_place, name)
		if os.path.exists(path_new):
			shutil.rmtree(path_new)
		shutil.move(name, path_new)

def rm_folders(folder_names):
	for name in folder_names:
		if os.path.exists(name):
			shutil.rmtree(name)
			
def rm_files(names):
	for name in names:
		os.remove(name)

def make_folder_names(nu_arr, clean_folder, out_folder):
	folder_names = []
	for nu in nu_arr:
		folder_names.append("psim_freq%d" % nu)
	folder_names.append(clean_folder)
	folder_names.append(out_folder)
	return folder_names

def make_fits_names(nu_arr):
	file_names = []
	for nu in nu_arr:
		file_names.append("./fits_modified/my_image_freq%d.fits" % nu)
	return file_names

def make_vis_noise_names(nu_arr):

	file_names = []
	for nu in nu_arr:
		project_name = "psim_freq%d" % nu
		file_names.append("./%s/%s.alma.out20.noisy.ms" % (project_name, project_name))
	return file_names

def make_vis_names(nu_arr):

	file_names = []
	for nu in nu_arr:
		project_name = "psim_freq%d" % nu
		file_names.append("./%s/%s.alma.out20.noisy.ms" % (project_name, project_name))
		file_names.append("./%s/%s.alma.out20.ms" % (project_name, project_name))
	return file_names


def load_image_npfile(folder, file_name ="input_model.npz"):
	path = os.path.join(folder, file_name)
	input_models = np.load(path)
	model_nu0 = input_models["model_nu0"]
	alpha_model = input_models["alpha"]
	nu_arr = np.array(input_models["nu_arr"])
	nu_0 = float(input_models["nu0"])
	model_freqs = input_models["model"]
	return model_freqs, model_nu0, alpha_model, nu_arr, nu_0

def title_makes(nu_arr, nu0):
	titles = []
	for nu in nu_arr:
		titles.append("image at %d GHz" % nu)
	titles.append("image at %d GHz" % nu0)
	titles.append("alpha")
	return titles

def clean_to_images_and_save(clean_folder, clean_name, nterm, nu_arr, nu0):


	fileimage_taylor0 = "./%s/%s.image.tt0.fits" % (clean_folder, clean_name)
	hdul = fits.open(fileimage_taylor0)
	image_taylor0 = np.array(hdul[0].data[0][0])
	nx, ny = np.shape(image_taylor0)
	header = hdul[0].header
	Bmaj = header["BMAJ"]
	Bmin = header["BMIN"]
	d_pixel = np.abs(header["CDELT1"])
	beam_div_pixel2 = np.pi * Bmaj * Bmin/(4 * np.log(2) * d_pixel **2)
	image_taylor0 = np.fliplr(image_taylor0/beam_div_pixel2) ##CLEANed image is flipped
	hdul.close()
	images_taylor0 = np.ones(nterm, nx, ny)
	images_taylor[0] = image_taylor0

	fileimage_alpha = "./%s/%s.alpha.fits"  % (clean_folder, clean_name)
	hdul = fits.open(fileimage_alpha)
	image_alpha = np.fliplr(np.array(hdul[0].data[0][0]))
	hdul.close()

	for i in range(nterm-1):
		i_now = i+1
		fileimage_taylor = "./%s/%s.image.tt%d.fits" % (clean_folder, clean_name, i_now)
		hdul = fits.open(fileimage_taylor)
		image_taylor_now = np.fliplr(np.array(hdul[0].data[0][0]))
		image_taylor_now = image_taylor1/beam_div_pixel2
		hdul.close()
		images_taylor[i_now] = image_taylor_now

	images_freqs=[]
	for nu in nu_arr:
		image_for_freq = np.zeros(nx, ny)
		for i in range(nterm):
			image_for_freq += images_taylor[i] * ((nu - nu0)/nu0)**i
		images_freqs.append(image_for_freq)
	images_freqs = np.array(images_freqs)

	np.savez("./%s/I0_alpha_clean" % clean_folder, I0 = image_taylor0, alpha = image_alpha, \
		image_nu = image_freqs)

