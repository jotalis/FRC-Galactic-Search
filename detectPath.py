import numpy as np
import time
import cv2

webcam = cv2.VideoCapture(0)
webcam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

# Allow time for the cam to warmup
time.sleep(1)
ret, imageFrame = webcam.read()
# Image manipulations to find yellow contours:
blurredFrame = cv2.GaussianBlur(imageFrame, (15, 15), 0)
hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

# HSV Value Range to detect
yellow_lower = np.array([20, 98, 125], np.uint8)
yellow_upper = np.array([28, 255, 204], np.uint8)

# Further Image manipulation + Detection
yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)
res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                             mask=yellow_mask)
_, contours, _ = cv2.findContours(yellow_mask,
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)

redA = False

for pic, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    imageFrame = cv2.rectangle(imageFrame, (x, y),
                               (x + w, y + h),
                               (0, 255, 255), 2)

    if w >= 14:
        redA = True

date = time.strftime("%Y%m%d-%H%M%S")
cv2.imwrite("{}.jpg".format(date), imageFrame)

if redA:
    print("redA")
else:
    print("blueA")

webcam.release()