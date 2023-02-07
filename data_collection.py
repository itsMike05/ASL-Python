import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Initializing the capture object and the hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Cropping offset and blank image size
crop_offset = 15
imgSize = 300

# Data folder location
data_folder = "Data/F"
data_counter = 0

# Displaying the camera output
while True:
    success, img = cap.read()
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
        else:
            k = imgSize / w  # Aspect ratio constant
            heightCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, heightCal))
            imgResizeShape = imgResize.shape
            # Centering the image
            heightGap = math.ceil((imgSize - heightCal) / 2)
            imgWhite[heightGap: heightCal + heightGap, :] = imgResize

        cv2.imshow("Img Crop", imgCrop)
        cv2.imshow("Img White", imgWhite)
        cv2.waitKey(30)

    cv2.imshow("ASL", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        data_counter += 1
        try:
            cv2.imwrite(f"{data_folder}/Image_{time.time()}.jpg", imgWhite)
        except Exception as e:
            print(f"An error occurred: {e}")
        print(f"Images taken: {data_counter}")
