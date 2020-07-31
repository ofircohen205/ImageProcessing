# Name: Ofir Cohen
# ID: 312255847
# Date: 12/1/2020

import cv2
import numpy as np
import matplotlib.pyplot as plt


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels = 9):
	# assume mask is float32 [0,1]
	# generate Gaussian pyramid for A,B and mask
	GA = A.copy()
	GB = B.copy()
	GM = m.copy()
	gpA = [GA]
	gpB = [GB]
	gpM = [GM]
	for i in range(num_levels):
		GA = cv2.pyrDown(GA)
		GB = cv2.pyrDown(GB)
		GM = cv2.pyrDown(GM)
		gpA.append(np.float64(GA))
		gpB.append(np.float64(GB))
		gpM.append(np.float64(GM))

	# generate Laplacian Pyramids for A,B and masks
	# the bottom of the Lap-pyr holds the last (smallest) Gauss level
	lpA = [gpA[num_levels-1]]
	lpB = [gpB[num_levels-1]]
	gpMr = [gpM[num_levels-1]]
	for i in range(num_levels-1,0,-1):
		# Laplacian: subtarct upscaled version of lower
		# level from current level
		# to get the high frequencies
		LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
		LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
		lpA.append(LA)
		lpB.append(LB)
		gpMr.append(gpM[i-1]) # also reverse the masks

	# Now blend images according to mask in each level
	LS = []
	for la,lb,gm in zip(lpA,lpB,gpMr):
		ls = la * gm + lb * (1.0 - gm)
		ls.dtype = np.float64
		LS.append(ls)

	# now reconstruct
	ls_ = LS[0]
	#ls_.dtype = np.float64
	for i in range(1,num_levels):
		print("LS" +str(i))
		ls_ = cv2.pyrUp(ls_)
		ls_ = cv2.add(ls_, LS[i])
	return ls_


def create_mask(img):
	for row in range(img.shape[0]):
		for col in range(img.shape[1]):
			if img[row, col] < 255:
				img[row, col] = 0
			else:
				img[row, col] = 1

	return img.astype(np.float32)


def main():
	A = cv2.imread('inputA.jpg', 0)
	B = cv2.imread('inputB.jpg', 0)
	m = cv2.imread('mask.jpg', 0)
	m = create_mask(m)

	lpb = Laplacian_Pyramid_Blending_with_mask(A, B, m)
	plt.imshow(lpb, cmap='gray')
	plt.show()

if __name__ == "__main__":
	main()