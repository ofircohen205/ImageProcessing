# Name: Ofir Cohen
# ID: 312255847
# Date: 5/12/2019

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter


def create_rectangle(size, start_x, start_y, end_x, end_y, rgb, thickness):
	'''
	This function creates a black square and a white rectangle inside it
	'''
	img = np.zeros(size, dtype=np.int8)
	cv2.rectangle(img, (start_x, start_y), (end_x, end_y), rgb, thickness)
	return img


def add_gaussian_noise(img, noise_to_add):
	'''
	This function adds to the image a gaussian noise
	'''
	tempImg = np.float64(np.copy(img))
	rows = tempImg.shape[0]
	cols = tempImg.shape[1]
	noise = np.random.randn(rows,cols) * (noise_to_add * 10)
	noisyImg = np.zeros(tempImg.shape, np.float64)
	if len(tempImg.shape) == 2:
		noisyImg =  tempImg + noise
	else:
		noisyImg[:,:,0] =  tempImg[:,:,0] + noise
		noisyImg[:,:,1] =  tempImg[:,:,1] + noise
		noisyImg[:,:,2] =  tempImg[:,:,2] + noise

	return  noisyImg


def find_edges_with_canny(img, sigma, t, T):
	img1 = gs_filter(img, sigma)
	img2, D = gradient_intensity(img1)
	img3 = suppression(np.copy(img2), D)
	img4, weak = threshold(np.copy(img3), t, T)
	img5 = tracking(np.copy(img4), weak)
	return img5


def plot_images(noised, edged, percentage):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.subplot(121)
	plt.imshow(noised, cmap='gray')
	plt.title('Noised')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(edged, cmap='gray')
	plt.title('Edged {}'.format(percentage))
	plt.xticks([])
	plt.yticks([])
	plt.show()


# Private functions
def gs_filter(img, sigma):
	return gaussian_filter(img, sigma)


########################################################


def gradient_intensity(img):
	Kx = np.array(
		[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
	)
	# Kernel for Gradient in y-direction
	Ky = np.array(
		[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
	)
	# Apply kernels to the image
	Ix = ndimage.filters.convolve(img, Kx)
	Iy = ndimage.filters.convolve(img, Ky)

	# return the hypothenuse of (Ix, Iy)
	G = np.hypot(Ix, Iy)
	D = np.arctan2(Iy, Ix)
	return (G, D)


########################################################


def suppression(img, D):
	M, N = img.shape
	Z = np.zeros((M,N), dtype=np.int32)

	for i in range(M):
		for j in range(N):
			# find neighbour pixels to visit from the gradient directions
			where = round_angle(D[i, j])
			try:
				if where == 0:
					if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
						Z[i,j] = img[i,j]
				elif where == 90:
					if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
						Z[i,j] = img[i,j]
				elif where == 135:
					if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
						Z[i,j] = img[i,j]
				elif where == 45:
					if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
						Z[i,j] = img[i,j]
			except IndexError as e:
				""" Todo: Deal with pixels at the image boundaries. """
				pass
	return Z


########################################################


def threshold(img, t, T):
	# define gray value of a WEAK and a STRONG pixel
	cf = {
		'WEAK': np.int32(50),
		'STRONG': np.int32(255),
	}

	# get strong pixel indices
	strong_i, strong_j = np.where(img > T)

	# get weak pixel indices
	weak_i, weak_j = np.where((img >= t) & (img <= T))

	# get pixel indices set to be zero
	zero_i, zero_j = np.where(img < t)

	# set values
	img[strong_i, strong_j] = cf.get('STRONG')
	img[weak_i, weak_j] = cf.get('WEAK')
	img[zero_i, zero_j] = np.int32(0)

	return (img, cf.get('WEAK'))


########################################################


def tracking(img, weak, strong=255):
	M, N = img.shape
	for i in range(M):
		for j in range(N):
			if img[i, j] == weak:
				# check if one of the neighbours is strong (=255 by default)
				try:
					if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
						or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
						or (img[i+1, j + 1] == strong) or (img[i-1, j - 1] == strong)):
						img[i, j] = strong
					else:
						img[i, j] = 0
				except IndexError as e:
					pass
	return img


########################################################


def round_angle(angle):
	""" Input angle must be \in [0,180) """
	angle = np.rad2deg(angle) % 180
	if (0 <= angle < 22.5) or (157.5 <= angle < 180):
		angle = 0
	elif (22.5 <= angle < 67.5):
		angle = 45
	elif (67.5 <= angle < 112.5):
		angle = 90
	elif (112.5 <= angle < 157.5):
		angle = 135
	return angle


def main():
	rectangles = []
	noise = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	for i in range(9):
		rect = create_rectangle(size=(256, 256), start_x=60, start_y=60, end_x=200, end_y=220, rgb=(255, 255, 255), thickness=-1)
		noised = add_gaussian_noise(img=rect, noise_to_add=noise[i])
		rectangles.append({
			"original": rect,
			"noised": noised,
			"edged": None
		})
	
	for i in range(9):
		rectangles[i]['edged'] = find_edges_with_canny(rectangles[i]['noised'], np.sqrt(2), 75, 200)


	for i in range(9):
		plot_images(rectangles[i].get('noised'), rectangles[i].get('edged'), noise[i])



if __name__ == "__main__":
	main()