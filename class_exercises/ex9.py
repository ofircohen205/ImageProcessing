# Name: Ofir Cohen
# ID: 312255847
# Date: 15/12/2019

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_images(original, lines):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	plt.subplot(121)
	plt.imshow(lines, cmap='gray')
	plt.title('Thick lines')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(original, cmap='gray')
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def keep_thick_lines(img):
	kernel = np.ones((5,5), np.uint8)
	dilate = cv2.dilate(img, kernel)
	erode = cv2.erode(dilate, kernel)
	return erode


def main():
	img = cv2.imread('lines.png')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	thick_lines = keep_thick_lines(gray)
	
	plot_images(img, thick_lines)


if __name__ == "__main__":
	main()