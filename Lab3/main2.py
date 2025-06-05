import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt

image_files = glob.glob("./res/*")
image_files.sort()

if not image_files:
    print("Nema slika u folderu!")
    exit()


images = []
for img_path in image_files:
    img = cv.imread(img_path)
    cv.cvtColor(img, cv.COLOR_BGR2YUV)
    if img is None:
        print(f"Slika '{img_path}' nije uspešno učitana.")
    else:
        images.append(img)

if not images:
    print("Nema validnih slika za obradu.")
    exit()

plt.figure(figsize=(20, 10))
for i, img in enumerate(images):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(img)
    plt.title(f"Slika {i + 1}")
    plt.axis('off')
plt.show()

def napraviPanoramu():
    img = NapraviPanoramuOdDveSlike(images[1], images[2])
    img = NapraviPanoramuOdDveSlike(images[0], img)
    return img

def NapraviPanoramuOdDveSlike(imgL, imgR):
    detector = cv.SIFT_create()
   
    kp1, des1 = detector.detectAndCompute(imgR, None) 
    kp2, des2 = detector.detectAndCompute(imgL, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=1)
    search_params = dict(checks=50) 

    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > 30:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 30))
        return None
    
    prikaziSpojeneTacke(imgL, imgR, kp1, kp2, matches, good)


    width = imgL.shape[1] + imgR.shape[1]
    height = imgL.shape[0] + int(imgR.shape[0] / 2)
    outimg = cv.warpPerspective(imgR, M, (width, height))
    outimg[0:imgL.shape[0], 0:imgL.shape[1]] = imgL

    return outimg

def prikaziSpojeneTacke(imgL, imgR, kp1, kp2, matches, good_matches):
    match_img = cv.drawMatches(imgR, kp1, imgL, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(15, 8))
    plt.imshow(match_img)
    plt.title("Spojene karakteristične tačke")
    plt.axis('off')
    plt.show()


plt.figure(figsize=(20, 10))
for i, img in enumerate(images):
    plt.subplot(1, len(images), i + 1)
    plt.imshow(img)
    plt.title(f"Slika {i + 1}")
    plt.axis('off')
plt.show()

panorama = napraviPanoramu()

if panorama is not None:
    plt.figure(figsize=(15, 8))
    plt.imshow(cv.cvtColor(panorama, cv.COLOR_BGR2RGB))
    plt.title("Panorama")
    plt.axis('off')
    plt.show()

    cv.imwrite("output.jpg", panorama)
else:
    print("Panorama nije uspešno napravljena.")