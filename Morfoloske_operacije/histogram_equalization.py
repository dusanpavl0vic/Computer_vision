import cv2
import numpy as np

def histogram_equalization(image):
    
    return cv2.equalizeHist(image)

image = cv2.imread('buildings.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

equalized_image = histogram_equalization(gray_image)


cv2.imshow('Equalized Image', equalized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
