# Name: Ofir Cohen
# ID: 312255847
# Date: 12/1/2020

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_rectangle(size, start, end, rgb):
	'''
	This function creates a black square and a white rectangle inside it
	'''
	img = np.zeros(size, dtype=np.uint8)
	cv2.rectangle(img, start, end, rgb, 1)
	return img


def plot_images(original, corners):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	plt.subplot(121)
	plt.imshow(original, cmap='gray')
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(corners, cmap='gray')
	plt.title('Corners')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def find_corners(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst > 0.01*dst.max()] = [255,0,0]

	return img


def main():
	rect = create_rectangle(size=(256, 256, 3), start=(60,60), end=(200,220), rgb=(255, 255, 255))
	corners = find_corners(rect.copy())
	plot_images(rect, corners)


if __name__ == "__main__":
	main()