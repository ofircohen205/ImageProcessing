# Name: Ofir Cohen
# ID: 312255847
# Date: 24/11/2019

import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_rectangle(start_x, start_y, end_x, end_y, rgb, thickness):
    img = np.zeros((256, 256, 3), np.int8)
    cv2.rectangle(img, (start_x, start_y), (end_x, end_y), rgb, thickness)
    return img


def first_derivative(img, img_filter):
    new_img = cv2.filter2D(img, ddepth=-1, kernel=img_filter)
    return new_img


def plot_images(original, filtered):
    plt.subplot(121)
    plt.imshow(original)
    plt.title('Original')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(filtered)
    plt.title('Filtered')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def main():
    rect = create_rectangle(60, 60, 200, 220, (255, 255, 255), -1)

    rect_filter_degree_1 = np.array((   [-2, -1, 0],
                                        [-1, 0, 1],
                                        [0, 1, 2]))
    
    rect_filter_degree_2 = np.array((   [2, 1, 0],
                                        [1, 0, -1],
                                        [0, -1, -2]))
    
    new_rect_degree1 = first_derivative(rect, rect_filter_degree_1)
    new_rect_degree2 = first_derivative(rect, rect_filter_degree_2)
    
    new_rect = np.add(new_rect_degree1, new_rect_degree2)
    
    plot_images(rect, new_rect)

if __name__ == "__main__":
    main()