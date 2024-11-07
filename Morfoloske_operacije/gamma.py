import cv2
import numpy as np

def gamma_correction(image, gamma):
    image_normalized = image / 255.0
    corrected_image = np.power(image_normalized, 1 / gamma)
    corrected_image = np.uint8(corrected_image * 255)
    return corrected_image

image = cv2.imread('buildings.jpg')


gamma = 3
corrected_image = gamma_correction(image, 50)

cv2.imshow('Original Image', image)
cv2.imshow('Gamma Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
