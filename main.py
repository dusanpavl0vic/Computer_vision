import cv2

# Učitaj sliku
img = cv2.imread("imagecopy.png")

# Izdvoji region F: [y1:y2, x1:x2] = [50:150, 0:50]
region_F = img[50:150, 0:50]

# Sačuvaj kao posebnu sliku
cv2.imwrite("region_F.png", region_F)