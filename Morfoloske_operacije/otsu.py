import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('buildings.jpg', cv2.IMREAD_GRAYSCALE)

t, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


plt.hist(image.ravel(), bins=256, range=(0, 256))
plt.title("Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()


cv2.imshow("Otsu Thresholding", thresholded_image)



