import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('buildings.jpg', cv2.IMREAD_GRAYSCALE)


t, thresholded_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
erosion = cv2.erode(thresholded_image, kernel, iterations=1)
dilation = cv2.dilate(thresholded_image, kernel, iterations=1)
opening = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)


cv2.imshow("Erosion", erosion)
cv2.imshow("Dilation", dilation)
cv2.imshow("Opening", opening)
cv2.imshow("Closing", closing)

cv2.waitKey(0)
cv2.destroyAllWindows()

imgOpen = cv2.erode(img, kernel=kernel)
imgOpen = cv2.dilate(imgOpen, kernel=kernel)
cv2.imshow("Open", imgOpen)

imgOpen2 = cv2.morphologyEx(img, op=cv2.MORPH_OPEN, kernel=kernel)
cv2.imshow("Open2", imgOpen2)

imgClose = cv2.dilate(img, kernel=kernel)
imgClose = cv2.erode(imgClose, kernel=kernel)
cv2.imshow("Close", imgClose)

imgClose2 = cv2.morphologyEx(img, op=cv2.MORPH_CLOSE, kernel=kernel)
cv2.imshow("Close2", imgClose2)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
imgEdge = cv2.dilate(img, kernel=kernel)
imgEdge = imgEdge - img
cv2.imshow("Edge", imgEdge)