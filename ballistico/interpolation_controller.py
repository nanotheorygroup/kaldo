import numpy as np
from scipy import ndimage


def resample_fourier(observable, increase_factor):
	matrix = np.fft.fftn (observable, axes=(0, 1, 2))
	bigger_matrix = np.zeros ((increase_factor * matrix.shape[0], increase_factor * matrix.shape[1],increase_factor * matrix.shape[2])).astype (complex)
	half = int(matrix.shape[0] / 2)
	bigger_matrix[0:half, 0:half, 0:half] = matrix[0:half, 0:half, 0:half]
	bigger_matrix[-half:, 0:half, 0:half] = matrix[-half:, 0:half, 0:half]
	bigger_matrix[0:half, -half:, 0:half] = matrix[0:half, -half:, 0:half]
	bigger_matrix[-half:, -half:, 0:half] = matrix[-half:, -half:, 0:half]
	bigger_matrix[0:half, 0:half, -half:] = matrix[0:half, 0:half, -half:]
	bigger_matrix[-half:, 0:half, -half:] = matrix[-half:, 0:half, -half:]
	bigger_matrix[0:half, -half:, -half:] = matrix[0:half, -half:, -half:]
	bigger_matrix[-half:, -half:, -half:] = matrix[-half:, -half:, -half:]
	bigger_matrix = (np.fft.ifftn (bigger_matrix, axes=(0, 1, 2)))
	bigger_matrix *= increase_factor ** 3
	return bigger_matrix

	
def map_interpolator(k_list, observable):
	k_size = np.array(observable.shape)
	return ndimage.map_coordinates (observable, (k_list * k_size).T, order=0, mode='wrap')


def fourier_interpolator(k_list, observable, increase_factor=8):
	try:
		increase_factor = 2 * int(k_list.shape[0] / observable.shape[0])
	except AttributeError:
		increase_factor=increase_factor
	return resample_fourier (observable, increase_factor).real
	
	
def interpolator(k_list, observable):
	# Here we can put a pipeline of several interpolator
	observable = fourier_interpolator(k_list, observable)
	observable = map_interpolator(k_list, observable)
	return observable