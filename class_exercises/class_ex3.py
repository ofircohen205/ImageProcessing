# Name: Ofir Cohen
# ID: 312255847
# Date: 10/11/2019

import cv2
import numpy as np
from matplotlib import pyplot as plt

def class_histogram(file):
    img = cv2.imread(file,0)
    if (img is None):
        print("faild to read the image")
        exit

    plt.imshow(img,cmap='gray')
    plt.title("Orig Image")
    plt.show()

    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.plot(cdf_normalized, color = 'b')
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()

    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    plt.imshow(img2,cmap='gray')
    plt.show()
    
    
def main():
    class_histogram('good_image.jpg')
    class_histogram('bad_image.jpg')
    

if __name__ == "__main__":
    main()