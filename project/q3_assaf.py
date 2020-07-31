#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import glob
import cv2

names = []

def read_images(path):
    filenames = [img for img in glob.glob(path)]
    images = []
    for filename in filenames:
        names.append(filename)
        images.append(cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
    return images

def show_image(org, img):
    plt.subplot(121)
    plt.imshow(org, cmap='gray')
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()

def search_engine(database, search):
    working_copy = search.copy()
    working_copy = cv2.GaussianBlur(working_copy, (3,3), 0)
    maximum = 0
    match = []
    sift = cv2.xfeatures2d.SIFT_create()
    for i, candidate in enumerate(database):
        kp1, des1 = sift.detectAndCompute(working_copy, None)
        kp2, des2 = sift.detectAndCompute(candidate, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
        if len(good) > 10:
            if len(good) > maximum:
                maximum = len(good)
                match = candidate
        else:
            print("Not enough matches are found - %d !> %d" % (len(good), 10))
    show_image(search, match)

images = read_images("./images/q3/*.png")
db = read_images("./images/q3/DB/*.png")
for img in images:
    search_engine(db, img)