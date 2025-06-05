import cv2
import glob
import matplotlib.pyplot as plt
import math

imagefiles = glob.glob("./res/*")
imagefiles.sort()

if not imagefiles:
    print("Nema slika u folderu ./res/")
    exit()

images = []

for filename in imagefiles:
    img = cv2.imread(filename)
    if img is None:
        print(f"Slika {filename} nije učitana.")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)



stitcher = cv2.Stitcher_create()
status, result = stitcher.stitch(images)

if status == 0:
    plt.figure(figsize=[30,10])
    plt.imshow(result)
    plt.show()
else:
    print(f"Greška pri stitchovanju: status {status}")

