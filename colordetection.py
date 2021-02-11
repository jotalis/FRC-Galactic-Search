import numpy as np
import cv2
# Capturing video through webcam
webcam = cv2.VideoCapture(0)

while True:

    # Reading the video from the
    # webcam in image frames
    ret, imageFrame = webcam.read()

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    blurredFrame = cv2.GaussianBlur(imageFrame, (11,11),0)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)
    
    
    # Set range for lower and upper yellow color
    yellow_lower = np.array([20, 50, 100],np.uint8)
    yellow_upper = np.array([50, 200, 255], np.uint8)

    
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    yellow_mask = cv2.erode(yellow_mask, None, iterations=2)
    yellow_mask = cv2.dilate(yellow_mask, None, iterations=2)
    res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=yellow_mask)

    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(yellow_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0,255,255), 2)

            cv2.putText(imageFrame, "Yellow Color", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0,255,255))

    # Program Termination
    cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

