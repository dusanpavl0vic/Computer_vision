import cv2
import numpy as np
import matplotlib.pyplot as plt
from threshold_rucno import findThreshold
from skimage.morphology import reconstruction

def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded

img = cv2.imread('coins.png')
img2 = img.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

thresh = findThreshold(img)

_, img_binary = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)

plt.imshow(img, cmap='gray')
plt.title("Pocetna slika")
plt.show()

plt.imshow(img_binary, cmap='gray')
plt.title("Binarna slika")
plt.show()

rotate_img_binary = 255 - img_binary

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5 ,5)) #idealna velicina kernela za ovu sliku

#probao sam samo dilataciju
#imgDilate = cv2.dilate(rotate_img_binary, kernel=kernel)
#probao sam closing
imgClose = cv2.morphologyEx(rotate_img_binary, op=cv2.MORPH_CLOSE, kernel=kernel)

plt.imshow(imgClose, cmap='gray')
plt.title("Posle zatvaranja")
plt.show()



img2HSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
img2Sat = img2HSV[:, :, 1]

plt.imshow(img2HSV, cmap='gray')
plt.title("HSV slika")
plt.show()

plt.imshow(img2Sat, cmap='gray')
plt.title("Saturacioni kanal")
plt.show()


thresh2 = findThreshold(img2Sat)

# mogao sam da uvecam thresh2 za jos 70 dobio bih tacke na
# mestu bakarnog novcia i rekonstrukcija bi tada imala smisla ovako ne vidim poentu
_, imgHSV_binary = cv2.threshold(img2Sat, thresh2, 255, cv2.THRESH_BINARY)

plt.imshow(imgHSV_binary, cmap='gray')
plt.title("Binarna slika HSV saturacije")
plt.show()

rotate_img_binary2 = 255 - imgHSV_binary


kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(5 ,5)) #idealna velicina kernela za ovu sliku

imgHSVClose2 = cv2.morphologyEx(imgHSV_binary, op=cv2.MORPH_CLOSE, kernel=kernel2)

plt.imshow(imgHSVClose2, cmap='gray')
plt.title("HSV saturacija posle zatvaranja")
plt.show()

kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(7 ,7))

imgHSVOpen2 = cv2.morphologyEx(imgHSVClose2, op=cv2.MORPH_OPEN, kernel=kernel3)

plt.imshow(imgHSVOpen2, cmap='gray')
plt.title("HSV saturacija posle zatvaranja pa otvaranja")
plt.show()

reconstructed = morphological_reconstruction(imgHSVOpen2, imgClose)


plt.imshow(reconstructed)
plt.title("Nakon rekonstrukcije")
plt.show()


plt.imshow(img * reconstructed)
plt.title("Samo bakarni novcic")
plt.show()

#ovo je cak i bolje
plt.imshow(img * imgHSVOpen2)
plt.title("Samo bakarni novcic")
plt.show()









