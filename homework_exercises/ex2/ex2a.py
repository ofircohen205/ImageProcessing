# Name: Ofir Cohen
# ID: 312255847
# Date: 5/1/2020


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from scipy.ndimage.filters import convolve, gaussian_filter


def Canny(img, sigma, t, T):
	img1 = gs_filter(img, sigma)
	img2, D = gradient_intensity(img1)
	img3 = suppression(np.copy(img2), D)
	img4, weak = threshold(np.copy(img3), t, T)
	img5 = tracking(np.copy(img4), weak)
	return img5


def create_rgb_img(width, height, img):
	output_img = np.zeros((height, width, 3), dtype=np.uint8)
	output_img[:, :, 0] = img
	output_img[:, :, 1] = img
	output_img[:, :, 2] = img
	return output_img


def draw_red_lines(width, height, img, edges):
	sum = 0
	lines = []
	col_sums = [0 for _ in range(width)]
	for col in range(width):
		for row in range(60,height):
			col_sums[col] += edges[row,col]

	while len(lines) < 4:
		max_val = max(col_sums)
		index = col_sums.index(max_val)
		lines.append(index)
		for i in range(30):
			col_sums[index-i] = 0
			col_sums[index+i] = 0
		col_sums[index] = 0

	for col in lines:
		for row in range(height):
			for i in range(-1,3):
				img[row,col+i,1] = 0
				img[row,col+i,2] = 0


def threshold_img(img):
	thr1 = img.copy()
	height, width = thr1.shape
	for col in range(width):
		for row in range(height):
			if thr1[row,col] < 40:
				thr1[row,col] = 0
			else:
				thr1[row,col] = 255
	return thr1


def plot_image(canny):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	plt.imshow(canny)
	plt.title('Canny Output')
	plt.xticks([])
	plt.yticks([])
	plt.show()


################### Private functions ##################

def gs_filter(img, sigma):
	return gaussian_filter(img, sigma)


########################################################


def gradient_intensity(img):
	Kx = np.array(
		[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
	)
	# Kernel for Gradient in col-direction
	# Ky = np.array(
	# 	[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
	# )
	# Apply kernels to the image
	Ix = convolve(img, Kx)
	# Iy = convolve(img, Ky)
	
	# return the hypothenuse of (Ix, Iy)
	G = np.hypot(Ix, img)
	D = np.arctan2(img, Ix)
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



########################################################
######################## MAIN ##########################
########################################################


def main():
	img = cv2.imread('sudoku-original.jpg', 0)
	thr1 = threshold_img(img)
	height, width = thr1.shape
	edges = Canny(thr1, np.sqrt(0.9), 70, 200)
	edges_inv = (edges * -1) + 255
	output_img = create_rgb_img(width, height, img)
	draw_red_lines(width, height, output_img, edges_inv)
	plot_image(output_img)



if __name__ == "__main__":
	main()