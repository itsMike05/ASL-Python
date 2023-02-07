import cv2
from cvzone.HandTrackingModule import HandDetector

# Initializing the capture object and the hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Cropping offset
crop_offset = 15
# Displaying the camera output
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]  # Bounding box for the hand (x, y, width, height)
        imgCrop = img[y - crop_offset : y + h + crop_offset, x - crop_offset : x + w + crop_offset]
        cv2.imshow("Img Crop", imgCrop)
        cv2.waitKey(30)

    cv2.imshow("ASL", img)
    cv2.waitKey(1)
