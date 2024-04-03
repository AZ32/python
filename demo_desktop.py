import cv2
import numpy as np
from PIL import Image
import torch
from demo_visualizer import optimized_visualize_results
# import threading
# from queue import Queue

PATH_TO_CUSTOMPT = "models/Custom/bestv9.pt"

class ColorVision:
    def __init__(self, camera_path=0):
        # Get Camera Feed
        print("Initializing camera feed...")
        self.cap = cv2.VideoCapture(camera_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # reduce the width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 172)  # reduce the height

        print("Loading model! Please wait...")
        self.model = torch.hub.load("WongKinYiu/yolov7", "custom", PATH_TO_CUSTOMPT, trust_repo=True)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        print("Model loaded successfully!")

    def analyze_feed(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Resize the frame to 320x172
                frame = cv2.resize(frame, (320, 172))

                # Convert the color space from BGR (OpenCV default) to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
                results = self.model(image)

                # Get list of box locations, classes, and confidence scores
                items = results.pandas().xyxy[0]
                
                boxes = items[["xmin", "ymin", "xmax", "ymax"]].values
                classes = items["name"].values
                confidence_scores = items["confidence"].values

                annotated_frame = optimized_visualize_results(frame, boxes, classes, confidence_scores, max_detections=3)
                # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('Color Vision', annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit loop if 'q' is pressed
                    break
            else:
                print("Failed to grab frame")
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    color_vision = ColorVision("videos/Fruits.mp4")
    color_vision.analyze_feed()
