import cv2
from cvzone.HandTrackingModule import HandDetector

# Initializing the capture object and the hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Displaying the camera output
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    cv2.imshow("ASL", img)
    cv2.waitKey(1)