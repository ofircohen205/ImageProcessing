# Name: Ofir Cohen
# ID: 312255847
# Date: 24/11/2019

import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_image(name):
    return cv2.imread(name, 0)


def blur_image(img):
    return cv2.GaussianBlur(img, (25,25), 0)


def sharpen_image(img):
    kernel=np.array((   [-1, -1, -1, -1, -1, -1, -1],
                        [-1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, 49, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1], 
                        [-1, -1, -1, -1, -1, -1, -1]), dtype=np.int8)
    return cv2.filter2D(img, -1, kernel=kernel)


def plot_images(original, blurred, sharpened, diff):
    plt.subplot(141)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(142)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(143)
    plt.imshow(sharpened, cmap='gray')
    plt.title('Sharpened')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(144)
    plt.imshow(diff, cmap='gray')
    plt.title('Difference between Original and Sharpen Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    img = read_image("image2.jpeg")
    blurred_img = blur_image(img)
    sharpened_img = sharpen_image(blurred_img)
    diff = blurred_img - sharpened_img
    plot_images(img, blurred_img, sharpened_img, diff)

if __name__ == "__main__":
    main()