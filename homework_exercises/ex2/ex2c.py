# Name: Ofir Cohen
# ID: 312255847
# Date: 5/1/2020

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_images(original, lines, filling):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	plt.subplot(131)
	plt.imshow(original, cmap='gray')
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(132)
	plt.imshow(lines, cmap='gray')
	plt.title('Objects Boundry')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(133)
	plt.imshow(filling, cmap='gray')
	plt.title('Objects Filling')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def find_obj_boundry(img, kernel1, kernel2):

	objects_boundry = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel1)
	dilate = cv2.dilate(objects_boundry, kernel1)
	dilate = cv2.dilate(dilate, kernel2)
	subtract = dilate - objects_boundry
	subtract = cv2.morphologyEx(subtract,cv2.MORPH_CLOSE,kernel1)
	return subtract


def region_filling(img, x, y, new_color):
	kernel = np.array((	[0, 1, 0],
						[1, 1, 1],
						[0, 1, 0]), dtype=np.uint8)

	A_c = img
	objects_filling = np.zeros((img.shape[0],img.shape[1]))
	objects_filling[0][0] = 255

	objects_filling = threshold_img(objects_filling)
	A_c = threshold_img(A_c)
	objects_filling = objects_filling.astype(np.uint8)
	A_c = A_c.astype(np.uint8)
	while True:
		prev_img = objects_filling.copy()
		objects_filling = cv2.dilate(objects_filling, kernel)
		objects_filling = np.bitwise_and(objects_filling, A_c)
		if np.array_equal(prev_img, objects_filling):
			break
	
	return objects_filling


def threshold_img(img):
	thr1 = img.copy()
	height, width = thr1.shape
	for col in range(width):
		for row in range(height):
			if thr1[row,col] < 120:
				thr1[row,col] = 0
			else:
				thr1[row,col] = 255
	return thr1


def main():
	img = cv2.imread('rice.jpg', 0)
	thresh1 = threshold_img(img)
	height, width = thresh1.shape
	
	kernel1 = np.array((	[0,0,1,1,0,0],
							[0,1,1,1,1,0],
							[1,1,1,1,1,1],
							[1,1,1,1,1,1],
							[0,1,1,1,1,0],
							[0,1,1,1,1,0],
							[0,0,1,1,0,0]), dtype=np.uint8)
	
	kernel2 = np.array((	[0,1,0],
							[1,1,1],
							[0,1,0]), dtype=np.uint8)

	objects_boundry = find_obj_boundry(thresh1, kernel1, kernel2)

	objects_copy = (objects_boundry.copy() * -1) + 255

	objects_filling = region_filling(objects_copy, 0, 0, 0)
	objects_filling = cv2.dilate(objects_filling, kernel1)
	
	output_img = (objects_filling.copy() * -1) + 255
	
	plot_images(img, objects_boundry, output_img)


if __name__ == "__main__":
	main()