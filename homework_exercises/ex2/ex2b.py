# Name: Ofir Cohen
# ID: 312255847
# Date: 5/1/2020


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage as ndi
from scipy.ndimage.filters import convolve, gaussian_filter
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter


def create_rectangle(size, start, end, rgb, thickness):
	'''
	This function creates a black square and a white rectangle inside it
	'''
	img = np.zeros(size, dtype=np.int8)
	cv2.rectangle(img, start, end, rgb, thickness)
	return img



def hough_lines(img):
	accumulator, thetas, rhos = hough_lines_helper(img)
	height, width = img.shape
	lines = []

	while len(lines) < 4:
		# Easiest peak finding based on max votes
		idx1 = np.argmax(accumulator)
		rho = rhos[idx1 // accumulator.shape[1]] 
		theta = thetas[idx1 % accumulator.shape[1]]
		accumulator[idx1 // accumulator.shape[1], idx1 % accumulator.shape[1]] = 0 

		x0 = 0
		x1 = height

		if theta == 0.0:
			x0 = round_to_closeset(rho)
			x1 = round_to_closeset(rho)
			y0 = 0
			y1 = width
			line_to_add = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
			if line_to_add not in lines:
				lines.append(line_to_add)
		else:
			y0 = int((-np.cos(theta) / np.sin(theta))*x0 + (rho / np.sin(theta)))
			y0 = round_to_closeset(y0)
			y1 = int((-np.cos(theta) / np.sin(theta))*x1 + (rho / np.sin(theta)))
			y1 = round_to_closeset(y1)
			line_to_add = {"x0": x0, "x1": x1, "y0": y0, "y1": y1}
			if y0 > height or y0 < 0 or y1 > height or y1 < 0:
				continue
			
			if line_to_add not in lines:
				lines.append(line_to_add)
		
	return lines


def hough_lines_helper(img):
	# Rho and Theta ranges
	thetas = np.deg2rad(np.arange(-90.0, 90.0))
	width, height = img.shape
	diag_len = int(np.ceil(np.sqrt(width * width + height * height)) ) # max_dist
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
			rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
			accumulator[rho, t_idx] += 1

	return accumulator, thetas, rhos


def draw_rect(lines, height, width):
	output = np.zeros((height, width), dtype=np.uint8)
	
	x_lines = []
	y_lines = []

	for index in range(len(lines)):
		for key in lines[index]:
			if key is 'x0' or key is 'x1':
				if lines[index][key] not in x_lines and lines[index][key] != width and lines[index][key] != 0:
					x_lines.append(lines[index][key])
			if key is 'y0' or key is 'y1':
				if lines[index][key] not in y_lines and lines[index][key] != height and lines[index][key] != 0:
					y_lines.append(lines[index][key])

	for x in x_lines:
		cv2.line(output, (x, y_lines[0]), (x, y_lines[1]), (1,1,1), 1)

	for y in y_lines:
		cv2.line(output, (x_lines[0], y), (x_lines[1], y), (1,1,1), 1)

	return output


def round_to_closeset(num):	
	rem = num % 10
	if rem < 5:
		num = int(num / 10) * 10
	else:
		num = int((num + 10) / 10) * 10

	return num


def add_gaussian_noise(img, noise_to_add):
	'''
	This function adds to the image a gaussian noise
	'''
	tempImg = np.float64(np.copy(img))
	rows = tempImg.shape[0]
	cols = tempImg.shape[1]
	noise = np.random.randn(rows,cols) * noise_to_add
	noisyImg = np.zeros(tempImg.shape, np.uint8)
	if len(tempImg.shape) == 2:
		noisyImg =  tempImg + noise
	else:
		noisyImg[:,:,0] =  tempImg[:,:,0] + noise
		noisyImg[:,:,1] =  tempImg[:,:,1] + noise
		noisyImg[:,:,2] =  tempImg[:,:,2] + noise

	return noisyImg


def Canny(img, sigma, t, T):
	img1 = gs_filter(img, sigma)
	img2, D = gradient_intensity(img1)
	img3 = suppression(np.copy(img2), D)
	img4, weak = threshold(np.copy(img3), t, T)
	img5 = tracking(np.copy(img4), weak)
	return img5


def threshold_img(img):
	thr1 = img.copy()
	height, width = thr1.shape
	for col in range(width):
		for row in range(height):
			if thr1[row,col] <= 50:
				thr1[row,col] = 0
			else:
				thr1[row,col] = 1
	return thr1


def plot_images(original, noised, edges, edges_hl):
	'''
	Plotting the Original image and the Edged image
	'''
	plt.figure(figsize=(20,20))
	plt.subplot(221)
	plt.imshow(original, cmap='gray')
	plt.title('Original')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(222)
	plt.imshow(noised, cmap='gray')
	plt.title('Noised')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(223)
	plt.imshow(edges, cmap='gray')
	plt.title('Canny Edges')
	plt.xticks([])
	plt.yticks([])
	plt.subplot(224)
	plt.imshow(edges_hl, cmap='gray')
	plt.title('Hough Lines Edges')
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
	Ky = np.array(
		[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
	)
	# Apply kernels to the image
	Ix = convolve(img, Kx)
	Iy = convolve(img, Ky)
	
	# return the hypothenuse of (Ix, Iy)
	G = np.hypot(Ix, Iy)
	D = np.arctan2(Iy, Ix)
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
	rect = create_rectangle(size=(256, 256), start=(60,60), end=(200,220), rgb=(255, 255, 255), thickness=-1)
	rect_noised = add_gaussian_noise(rect, 10)
	
	edges = Canny(rect_noised, np.sqrt(2), 75, 200)
	thresh = threshold_img(edges)
	height, width = thresh.shape
	
	lines = hough_lines(thresh)
	output_img = draw_rect(lines, 256, 256)
	plot_images(rect, rect_noised, edges, output_img)

if __name__ == "__main__":
	main()