import cv2
import numpy as np
import digitalio
import os
import time
import board
import adafruit_st7789
from adafruit_rgb_display import st7789  # pylint: disable=unused-import
import detection_visualizer
from PIL import Image, ImageOps
import torch
import numpy as np

PATH_TO_CUSTOMPT = "models/Custom/bestv9.pt"

# Button pins for EYESPI Pi Beret
BUTTON_NEXT = board.D5
BUTTON_PREVIOUS = board.D6

# Initialize TFT display
# CS and DC pins for EYEPSPI Pi Beret:
cs_pin = digitalio.DigitalInOut(board.CE0)
dc_pin = digitalio.DigitalInOut(board.D25)

# Reset pin for EYESPI Pi Beret
reset_pin = digitalio.DigitalInOut(board.D27)

# Backlight pin for Pi Beret
backlight = digitalio.DigitalInOut(board.D18)
backlight.switch_to_output()
backlight.value = True

# Config for display baudrate (default max is 64mhz):
BAUDRATE = 64000000

# Setup SPI bus using hardware SPI:
spi = board.SPI()

disp = st7789.ST7789(spi, rotation=90, width=172, height=320, x_offset=34, # 1.47" ST7789
                    cs=cs_pin,
                    dc=dc_pin,
                    rst=reset_pin,
                    baudrate=BAUDRATE,
)


class ColorVision:
    def __init__(self, camera_path=0):
        # Get Camera Feed
        self.cap = cv2.VideoCapture(camera_path)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # reduce the width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 172)  # reduce the height

        self.frame_skip = 2  # Skip every 2 frames
        self.frame_count = 0

        self.advance_button = self.init_button(BUTTON_NEXT)
        self.back_button = self.init_button(BUTTON_PREVIOUS)

        print("Loading model! Please wait...")
        self.model = torch.hub.load("WongKinYiu/yolov7", "custom", f"{PATH_TO_CUSTOMPT}", trust_repo=True)
        self.model.eval()
        print("Model loaded successfully!")



    def init_button(self, pin):
        button = digitalio.DigitalInOut(pin)
        button.switch_to_input()
        button.pull = digitalio.Pull.UP
        return button

    def analyze_feed(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            frame_count += 1
            if ret:
                if frame_count % self.frame_skip != 0:
                    continue  # Skip this frame

                # Convert the color space from BGR (OpenCV default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to fit the display resolution
                frame = cv2.resize(frame, (disp.height, disp.width))

                # Give the frame to the custom model
                frame_tensor = torch.from_numpy(frame).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)
                with torch.no_grad():
                    results = self.model(frame_tensor)

                # get list of box locations
                # boxes = results['detection_boxes'][0].numpy()
                boxes = results.xyxy[0].numpy()

                # get list of classes
                # classes = results['detection_classes'][0].numpy()
                classes = boxes[:,-2]

                # get list of confidence scores
                # confidence_scores = results['detection_scores'][0].numpy()
                confidence_scores = boxes[:,:4]

                # Annotate the frame with boxes and labels for object name and color
                for box, label, confidence in zip(boxes, classes, confidence_scores):
                    # Draw the bounding box
                    x1, y1, x2, y2 = box[:4]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)

                    color = detection_visualizer.fast_dominant_color(frame)
                    color_name = detection_visualizer.get_color_name(color)

                    # Draw the label
                    label = int(label)
                    confidence = float(confidence)
                    cv2.putText(frame, f"{label} {confidence} {color_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)


                # Convert the frame to a PIL Image
                frame = Image.fromarray(frame)

                # Draw the image on the display
                disp.image(frame)
            else:
                print("Failed to grab frame")
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    color_vision = ColorVision()
    color_vision.analyze_feed()