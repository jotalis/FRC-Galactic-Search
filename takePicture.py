import numpy as np
import time
import cv2

webcam = cv2.VideoCapture(1)
time.sleep(1)  # Jay, you may have to change this number depending on how long it takes for the rpi cam to warm up

bounding_boxes = {}  # dictionary of coordinates and sizes of each bounding box corresponding with the bounding box's number

ret, imageFrame = webcam.read()

# Image manipulations to find yellow contours:
blurredFrame = cv2.GaussianBlur(imageFrame, (15, 15), 0)
hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)

yellow_lower = np.array([20, 50, 100], np.uint8)
yellow_upper = np.array([50, 200, 255], np.uint8)

yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)
res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                             mask=yellow_mask)

contours, hierarchy = cv2.findContours(yellow_mask,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)

# Find and draw bounding boxes:
box_counter = 0

for pic, contour in enumerate(contours):

    area = cv2.contourArea(contour)

    if area > 300:  # may have to be changed to a larger number to detect large powercells vs small yellow patches

        box_counter += 1

        x, y, w, h = cv2.boundingRect(contour)
        imageFrame = cv2.rectangle(imageFrame, (x, y),
                                   (x + w, y + h),
                                   (0, 255, 255), 2)

        coordinate = [x + w / 2, y + h / 2]
        bounding_boxes["Box " + str(box_counter)] = {"sc": [x, y], "cc": coordinate,
                                                     "w": w, "h": h}

        cv2.putText(imageFrame, str(box_counter), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255))

print(bounding_boxes)

# Save results
if cv2.waitKey(10) & 0xFF == ord('q'):
    date = time.strftime("%Y%m%d-%H%M%S")

    cv2.imwrite("{}.jpg".format(date), imageFrame)
    file = open("{}.txt".format(date), "w")

    text = ""

    for box in bounding_boxes:

        text += box + "\n\nBox Start Coordinates: ({},{})\nBox Center Coordinates: ({},{})\nWidth: {}\nHeight: {" \
                      "}\n\n\n".format(bounding_boxes[box]["sc"][0], bounding_boxes[box]["sc"][1],
                                       bounding_boxes[box]["cc"][0], bounding_boxes[box]["cc"][1],
                                       bounding_boxes[box]["w"], bounding_boxes[box]["h"])

    file.write(text)

    file.close()
    webcam.release()
