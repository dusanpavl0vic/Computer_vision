import cv2
import numpy as np
import argparse
import time

image = cv2.imread("input.png")

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, binary = cv2.threshold(image_gray, 240, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    if w > 1440 and h > 720:
        cropped_image = image[y:y+h, x:x+w]
        break



cv2.imshow("Input image", image)
cv2.imshow("Crop image", cropped_image)
cv2.imwrite("crop_input.png", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, 	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True, help="path to ImageNet labels (i.e., syn-sets)")
args = vars(ap.parse_args())

# load the input image from disk
image = cv2.imread(args["image"])
 


# Sliding Window
def sliding_window(image, window_size, stride):
    for y in range(0, image.shape[0] - window_size[1] + 1, stride):
        for x in range(0, image.shape[1] - window_size[0] + 1, stride):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# load the class labels from disk

rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

# Sliding Window size
window_size = (180, 180)
#offset
stride = 180 

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe('./bvlc_googlenet.prototxt', './bvlc_googlenet.caffemodel')

# iteretion Sliding Window
for (x, y, window) in sliding_window(image, window_size, stride):
    if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
        continue

    # Priprema prozora za model
    blob = cv2.dnn.blobFromImage(window, 1, (224, 224), (104, 117, 123))
    net.setInput(blob)
    preds = net.forward()

    # Detekcija klase sa najvećom verovatnoćom
    class_id = np.argmax(preds[0])
    class_label = classes[class_id]

    # Obeležavanje pasa i mačaka
    if "dog" in class_label.lower():
        cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 255, 255), 2)
        cv2.putText(image, "DOG", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    elif "cat" in class_label.lower():
        cv2.rectangle(image, (x, y), (x + window_size[0], y + window_size[1]), (0, 0, 255), 2)
        cv2.putText(image, "CAT", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Prikaz rezultata
cv2.imshow("Result", image)
cv2.imwrite("output.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()