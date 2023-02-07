from cProfile import label

import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initializing the Capture object, the Hand Detector and the Classifier
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/asl_keras_model.h5", "Model/labels.txt")
# Cropping offset and blank image size
crop_offset = 15
imgSize = 300

# Data folder location
data_folder = "Data/F"
data_counter = 0

# Data labels
data_labels = ["A", "B", "C", "D", "E", "F", "G"]

# Displaying the camera output
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # Bounding box for the hand (x, y, width, height)
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Creating a blank white image
        imgCrop = img[y - crop_offset: y + h + crop_offset, x - crop_offset: x + w + crop_offset]

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h  # Aspect ratio constant
            widthCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (widthCal, imgSize))
            imgResizeShape = imgResize.shape
            # Centering the image
            widthGap = math.ceil((imgSize - widthCal) / 2)
            imgWhite[:, widthGap: widthCal + widthGap] = imgResize
            # Starting the prediction
            prediction, index = classifier.getPrediction(imgWhite)
        else:
            k = imgSize / w  # Aspect ratio constant
            heightCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, heightCal))
            imgResizeShape = imgResize.shape
            # Centering the image
            heightGap = math.ceil((imgSize - heightCal) / 2)
            imgWhite[heightGap: heightCal + heightGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite)

        cv2.putText(imgOutput, data_labels[index], (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 3, (240, 128, 0), 2)

        cv2.imshow("Img Crop", imgCrop)
        cv2.imshow("Img White", imgWhite)
        cv2.waitKey(5)

    cv2.imshow("ASL Visualizer", imgOutput)
    cv2.waitKey(1)
