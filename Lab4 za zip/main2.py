import cv2
import numpy as np
import argparse
import time
import imutils

image = cv2.imread("input.png")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w > 1440 and h > 720:
        cropped_image = image[y:y+h, x:x+w]
        break



# cv2.imshow("Input image", image)
# cv2.imshow("Crop image", cropped_image)
# cv2.imwrite("crop_input.png", cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------------------

def scale(image, scale=1.5, minSize=(30, 30)):
    scaled_images = [image]
    
    while True:
        new_width = int(image.shape[1] / scale)
        image = imutils.resize(image, width=new_width)
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        scaled_images.append(image)
    
    return scaled_images

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, 	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# load the input image from disk
image = cropped_image

# Sliding Window
def sliding_window(image, window_size, stride):
    for y in range(0, image.shape[0] - window_size[1] + 1, stride):
        for x in range(0, image.shape[1] - window_size[0] + 1, stride):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# load the class labels from disk

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Sliding Window size
window_size = 180
step_size = 180
width = 1440
height = 720

net = cv2.dnn.readNetFromCaffe('./bvlc_googlenet.prototxt', './bvlc_googlenet.caffemodel')

image_scale = scale(image, scale=2.0, minSize=(350, 100))

for resized in image_scale:
    for y in range(0, resized.shape[0], step_size):
        for x in range(0, resized.shape[1], step_size):

            croppedImage = resized[y:y + window_size, x:x + window_size]
            blob = cv2.dnn.blobFromImage(croppedImage, 1.0, (224, 224), (104, 117, 123), swapRB = True)
            net.setInput(blob)
            preds = net.forward()       

            idxs = np.argsort(preds[0])[::-1][:1]
            idx = idxs[0]
            if preds[0][idx] > 0.5:

                odnos = int(image.shape[1] / resized.shape[1])
                x1 = x * odnos
                y1 = y * odnos
                wSw = window_size * odnos

                if "dog" in classes[idx]:
                    color = (0, 255, 255)
                    text = "DOG"
                elif "cat" in classes[idx]:
                    color = (0, 0, 255)
                    text = "CAT"
                else:
                    continue
                cv2.putText(image, text, (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(image, (x1 + 2, y1 + 2), (x1 + wSw - 2, y1 + wSw - 2), color, 2)

cv2.imwrite("output.jpg", image)

cv2.imshow("Image", image)
cv2.waitKey(0)