# Name: Ofir Cohen
# ID: 312255847
# Date: 5/12/2019

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_rectangle(size, start_x, start_y, end_x, end_y, rgb, thickness):
	'''
	This function creates a black square and a white rectangle inside it
	'''
	img = np.zeros(size, dtype=np.int8)
	cv2.rectangle(img, (start_x, start_y), (end_x, end_y), rgb, thickness)
	return img


def add_gaussian_noise(img):
	'''
	This function adds to the image a gaussian noise
	'''
	size = img.shape
	mean = 0
	var = 1
	sigma = var**0.5
	noise = np.random.normal(mean, sigma, size)
	noise = noise.reshape(img.shape[0], img.shape[1]).astype(np.int8)
	noise = img + img * noise * 0.05
	return noise


def convolution2d(image, kernel):
	'''
	Convolution between the image and the kernel
	'''	
	image_height, image_width= image.shape
	kernel_height, kernel_width = kernel.shape
	image_padded = np.zeros(shape=(image_height + kernel_height, image_width + kernel_width), dtype="float32")    
	image_padded[kernel_height//2:-kernel_height//2, kernel_width//2:-kernel_width//2] = image
	output = np.zeros(shape=image.shape)
	for row in range(image_height):
		for col in range(image_width):
			for i in range(kernel_height):
				for j in range(kernel_width):
					output[row, col] += image_padded[row + i, col + j]*kernel[i, j]
	
	output /= 255.0/output.max()
	return np.absolute(output).astype(np.int8)


def plot_images(original, edged):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.subplot(121)
	plt.imshow(original, cmap='gray')
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(edged, cmap='gray')
	plt.title('Edged')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def main():
	rect = create_rectangle(size=(256, 256), start_x=60, start_y=60, end_x=200, end_y=220, rgb=(255, 255, 255), thickness=-1)
	rect_noised = add_gaussian_noise(img=rect)
	
	binomial_filter = np.array(( [1, 4, 6, 4, 1],
									[4, 16, 24, 16, 4], 
									[6, 24, 36, 24, 6],
									[4, 16, 24, 16, 4],
									[1, 4, 6, 4, 1]), dtype=np.int8) / 256.0

	laplacian = np.array((  [1, 1, 1], 
							[1, -8, 1], 
							[1, 1, 1]), dtype=np.int8)

	removed_noise_rect = convolution2d(rect_noised, binomial_filter)
	only_edges_rect = convolution2d(removed_noise_rect, laplacian)
	
	plot_images(rect, only_edges_rect)

if __name__ == "__main__":
	main()