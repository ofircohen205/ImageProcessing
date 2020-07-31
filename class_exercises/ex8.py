# Name: Ofir Cohen
# ID: 312255847
# Date: 15/12/2019

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter


def create_lines(size, rgb, start, end):
	'''
	This function creates a two crossed lines
	'''
	img1 = np.zeros(size, dtype=np.int8)
	img1[start:end, start:end] = np.eye(end-start)
	img2 = np.zeros(size, dtype=np.int8)
	img2[start:end, start:end] = np.eye(end-start)
	img1 = np.fliplr(img1)
	res = img1 + img2
	return res



def hough_line(img):
	# Rho and Theta ranges
	thetas = np.deg2rad(np.arange(-90.0, 90.0))
	width, height = img.shape
	diag_len = int( np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
	rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)

	# Cache some resuable values
	cos_t = np.cos(thetas)
	sin_t = np.sin(thetas)
	num_thetas = len(thetas)

	# Hough accumulator array of theta vs rho
	accumulator = np.zeros((2 * diag_len, num_thetas))
	y_idxs, x_idxs = np.nonzero(img) # (row, col) indexes to edges

	# Vote in the hough accumulator
	for i in range(len(x_idxs)):
		x = x_idxs[i]
		y = y_idxs[i]
		for t_idx in range(num_thetas):
			# Calculate rho. diag_len is added for a positive index
			rho = int( round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len )
			accumulator[rho, t_idx] += 1
	
	return accumulator, thetas, rhos


def plot_images(original, crossed):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	plt.subplot(121)
	plt.imshow(crossed, cmap='gray')
	plt.title('Crossed')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(original, cmap='gray')
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def main():
	lines = create_lines(size=(256,256), rgb=(255,255,255), start=60, end=200)
	lines_crossed = create_lines(size=(256,256), rgb=(255,255,255), start=60, end=200)
	accumulator, thetas, rhos = hough_line(lines)
	
	# Easiest peak finding based on max votes
	idx = np.argmax(accumulator)
	rho = rhos[idx // accumulator.shape[1]]
	theta = thetas[idx % accumulator.shape[1]]
	
	accumulator1 = accumulator
	accumulator1[idx // accumulator.shape[1], idx % accumulator.shape[1]] = 0

	idx2 = np.argmax(accumulator1)
	rho2 = rhos[idx2 // accumulator1.shape[1]]
	theta2 = thetas[idx2 % accumulator1.shape[1]]

	m1 = int(-np.cos(theta) / np.sin(theta))
	b1 = int(rho / np.sin(theta))
	m2 = int(-np.cos(theta2) / np.sin(theta2))
	b2 = int(rho2 / np.sin(theta2))

	x = int((b2-b1)/(m1-m2))
	y = int(m2*x + b2) 

	lines_rgb = np.zeros((256, 256, 3))
	lines_rgb[:, :, 0] = lines
	lines_rgb[:, :, 1] = lines
	lines_rgb[:, :, 2] = lines
	lines_rgb[x, y, 1] = 0
	lines_rgb[x, y, 2] = 0


	plot_images(lines, lines_rgb)


if __name__ == "__main__":
	main()