from ultralytics import YOLO
import cv2
import math 
from ultra_visualizer import visualize_results
# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 172)

# # model
# model = YOLO("ultra_models/model.pt")

# # object classes
# classNames = ["Apple", " Banana", "Blood", "Car", "Cat", "Cherry", "Clothes", " Grapes", "GreenTraffic Light", 
#               "Map", "Orange", "Peach", "Pear", "Person", "Pineapple", "RedTraffic Light", "Strawberry", 
#               "Sunburn", "Tomatoes", "Warning Sign", "Watermelon", "YellowTraffic Light"]

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confidence --->",confidence)

            # class name
            cls = int(box.cls[0])
            print("Class name -->", classNames[cls])

            img = visualize_results(img, box, classNames[cls], confidence)

    cv2.imshow('Colorblind Demo', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()