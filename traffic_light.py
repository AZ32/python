import cv2
import numpy as np
# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

def detect_traffic_light(img_path):
    # Load Image
    img = cv2.imread(img_path)

    width = 500
    height = 300

    down_size = (width, height)
    img = cv2.resize(img, down_size, interpolation=cv2.INTER_LINEAR)

    # Convert image color to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color Ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2

    yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))

    combined_mask = red_mask | yellow_mask | green_mask

    combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in combined_contours:
    #     if 200 < cv2.contourArea(contour) < 3000:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for contour in red_contours:
        if 200 < cv2.contourArea(contour) < 3000:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) < 25:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    for contour in yellow_contours:
        if 200 < cv2.contourArea(contour) < 3000:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) < 25:
                cv2.rectangle(img, (x, y), (x + w, y + h), (30, 248, 252), 2)

    for contour in green_contours:
        if 200 < cv2.contourArea(contour) < 3000:
            x, y, w, h = cv2.boundingRect(contour)
            if abs(w - h) < 25:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Traffic Light Test", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_traffic_light("Red_yellow_green_traffic_lights.jpg")
# detect_traffic_light("test_4.jpg")

def get_traffic_light_color(cropped_image):
    # Convert image color to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Color Ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = red_mask1 | red_mask2

    yellow_mask = cv2.inRange(hsv, np.array([20, 50, 50]), np.array([30, 255, 255]))
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))

    combined_mask = red_mask | yellow_mask | green_mask

    # combined_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return red_contours, yellow_contours, green_contours