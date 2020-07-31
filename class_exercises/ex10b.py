# Name: Ofir Cohen
# ID: 312255847
# Date: 12/1/2020

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_images(imgs, rows, columns):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	for index in range(rows*columns):
		plt.subplot(rows, columns,index+1)
		plt.imshow(imgs[index], cmap='gray')
		plt.xticks([])
		plt.yticks([])
	plt.show()


def create_mask(img):
	height, width = img.shape

	mask = np.full(img.shape, 255, dtype=np.uint8)
	mask[:, (height // 2) - 5:(height // 2) + 5] = 0
	mask[(width // 2) - 5:(width // 2) + 5, :] = 0

	return mask


def check_dots(image):
    hei,wid = image.shape
    temp = np.zeros((hei, wid, 3), dtype=np.uint8)
    for i in range(hei):
        for j in range(wid):
            if(image[i][j] >= 34000 and i < 115):
                temp[i, j, 0] = 255
    return temp


def main():
	img = cv2.imread('ex10.png', 0)
	mask = create_mask(img)
	
	ftimage = np.fft.fft2(img) # transform image with fourier
	ftimage = np.fft.fftshift(ftimage)
	
	ftimagep = ftimage * mask
	imagep = np.fft.ifft2(ftimagep)
	imagep_abs = np.abs(imagep)

	dots = check_dots(imagep_abs)
	
	final_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) + dots
	plot_images([img, final_img], 1, 2)


if __name__ == "__main__":
	main()