# Name: Ofir Cohen
# ID: 312255847
# Date: 22/12/2019

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
	plt.title('objects_boundry')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(122)
	plt.imshow(original, cmap='gray')
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	plt.show()


def find_obj_boundry(img, operation, kernel):
	objects_boundry = cv2.morphologyEx(img, operation, kernel)
	dilate = cv2.dilate(objects_boundry, kernel)
	subtract = dilate - objects_boundry
	return subtract


def main():
	img = cv2.imread('ex9b.jpg')
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	ret,thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
	
	kernel = np.array((	[0,0,1,1,0,0],
                    	[0,1,1,1,1,0],
                    	[1,1,1,1,1,1],
                    	[1,1,1,1,1,1],
                		[0,1,1,1,1,0],
                		[0,1,1,1,1,0],
                		[0,0,1,1,0,0]), dtype=np.uint8)

	objects_boundry = find_obj_boundry(thresh1, cv2.MORPH_OPEN, kernel)
	
	plot_images(img, objects_boundry)


if __name__ == "__main__":
	main()