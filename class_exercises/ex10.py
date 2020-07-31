# Name: Ofir Cohen
# ID: 312255847
# Date: 5/1/2020

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_images(imgs, rows, columns, explain):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	for index in range(rows*columns):
		plt.subplot(rows, columns,index+1)
		plt.imshow(imgs[index], cmap='gray')
		if index == 0:
			plt.title('Original image')
		elif index == 1:
			plt.title('Mask')
		else:
			plt.title('FFT Output')
		plt.xticks([])
		plt.yticks([])
	plt.figtext(0.5, 0.1, explain, ha="center", fontsize=7, bbox={"facecolor":"orange", "alpha":0.5, "pad":5}) 
	plt.show()


def create_masks(img):
	masks = []
	height, width = img.shape

	mask1 = np.full(img.shape, 255, dtype=np.uint8)
	mask1[:, (height // 2) - 5:(height // 2) + 5] = 0
	masks.append(mask1)

	mask2 = np.full(img.shape, 255, dtype=np.uint8)
	mask2[(width // 2) - 5:(width // 2) + 5, :] = 0
	masks.append(mask2)

	mask3 = np.zeros(img.shape, dtype=np.uint8)
	mask3[:, (height // 2) - 5:(height // 2) + 5] = 255
	masks.append(mask3)

	mask4 = np.zeros(img.shape, dtype=np.uint8)
	mask4[(width // 2) - 5:(width // 2) + 5, :] = 255
	masks.append(mask4)

	mask5 = np.zeros(img.shape, dtype=np.uint8)
	mask5[(width // 2) - 5:(width // 2) + 5, (height // 2) - 5:(height // 2) + 5] = 255
	masks.append(mask5)

	mask6 = np.full(img.shape, 255, dtype=np.uint8)
	mask6[(width // 2) - 5:(width // 2) + 5, (height // 2) - 5:(height // 2) + 5] = 0
	masks.append(mask6)

	mask7 = np.full(img.shape, 255, dtype=np.uint8)
	mask7[:, (height // 2) - 5:(height // 2) + 5] = 0
	mask7[(width // 2) - 5:(width // 2) + 5, :] = 0
	masks.append(mask7)

	mask8 = np.zeros(img.shape, dtype=np.uint8)
	mask8[:, (height // 2) - 5:(height // 2) + 5] = 255
	mask8[(width // 2) - 5:(width // 2) + 5, :] = 255
	masks.append(mask8)

	return masks


def main():
	img = cv2.imread('ex10.png', 0)
	all_images = []
	imageps = []

	masks = create_masks(img)
	
	ftimage = np.fft.fft2(img) # transform image with fourier
	ftimage = np.fft.fftshift(ftimage)
	
	for mask in masks:
		ftimagep = ftimage * mask
		imagep = np.fft.ifft2(ftimagep)
		imageps.append(np.abs(imagep))
		all_images.append([img, mask])
	
	for i in range(len(imageps)):
		all_images[i].append(imageps[i])

	explain = ['Removes the horizontal lines', 'Removes the vertical lines', 'Highlights the horizontal lines', 'Highlights the vertical lines'  , 'Removes the high frequencies', 'Removes the low frequencies', 'Shows only the connections between the horizontal and vertical lines', 'Removes the connections between the horizontal and vertical lines']
	for i in range(len(all_images)):
		plot_images(all_images[i], 1, 3, explain[i])


if __name__ == "__main__":
	main()