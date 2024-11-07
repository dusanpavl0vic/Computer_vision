import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def thinning(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False

    plt.figure(figsize=(6, 6))
    plt.ion()

    while not done:

        eroded = cv.erode(img, kernel)
        plt.imshow(eroded, cmap='gray')
        plt.title("After Erosion")
        plt.pause(0.5)

        temp = cv.dilate(eroded, kernel)
        plt.imshow(temp, cmap='gray')
        plt.title("After Dilation")
        plt.pause(0.5)

        temp = cv.subtract(img, temp)
        plt.imshow(temp, cmap='gray')
        plt.title("Contour Extraction")
        plt.pause(0.5)

        skel = cv.bitwise_or(skel, temp)
        plt.imshow(skel, cmap='gray')
        plt.title("Updated Skeleton")
        plt.pause(0.5)

        img = eroded.copy()
        zeros = size - cv.countNonZero(img)
        if zeros == size:
            done = True

    plt.ioff()

    return skel


img = cv.imread('Fingerprint.jpg', 0)
_, imgTh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

skeleton = thinning(imgTh)

plt.imshow(skeleton, cmap='gray')
plt.title("Skeleton")
plt.show()