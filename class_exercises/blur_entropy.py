# Name: Ofir Cohen
# ID: 312255847
# Date: 17/11/2019

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import entropy
from PIL import Image


def blur_img_gaussian(img, size):
    return cv2.GaussianBlur(img, size, 0)


def blur_img(img, size):
    return cv2.blur(img, size)


def plot_images(original, blurred, image_entropy, original_entropy, plot_type):
    plt.subplot(121)
    plt.imshow(original)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Entropy: {}'.format(original_entropy))
    plt.subplot(122)
    plt.imshow(blurred)
    plt.title('Blurred {}'.format(plot_type))
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('Entropy: {}'.format(image_entropy))
    plt.show()


def calc_entropy(img):
    return entropy(img)


def add_gaussian_noise(img, alpha):
    filterd_blur = cv2.GaussianBlur(img,(7,7),0)
    sharppened = img + (alpha*(img - filterd_blur))
    
    return sharppened



def main():
    img = cv2.imread('image2.jpeg', 0)
    
    blur_gaussian = blur_img_gaussian(img, (25, 25))
    blur_1 = blur_img(img, (250, 250))
    blur_2 = blur_img(img, (100, 100))
    
    res_original = calc_entropy(img)
    res_gaussian = calc_entropy(blur_gaussian)
    res_1 = calc_entropy(blur_1)
    res_2 = calc_entropy(blur_2)
    
    plot_images(img, blur_gaussian, res_gaussian, res_original, "Gaussian")
    plot_images(img, blur_1, res_1, res_original, "Blur 250")
    plot_images(img, blur_2, res_2, res_original, "Blur 100")
    
    noised_img = add_gaussian_noise(img, 0.5)
    res_noised = calc_entropy(noised_img)
    plot_images(img, noised_img, res_noised, res_original, "Gaussian noise")


if __name__ == "__main__":
    main()