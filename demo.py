import cv2
import numpy as np
import digitalio
import os
import time
import board
import adafruit_st7789
from adafruit_rgb_display import st7789  # pylint: disable=unused-import
from PIL import Image, ImageOps
import torch
import numpy as np
from detection_visualizer import visualize_results, crop_image, attach_to_original

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

        self.button_5 = self.init_button(BUTTON_NEXT)
        self.button_6 = self.init_button(BUTTON_PREVIOUS)

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
        while cap.isOpened():
            _btn_5_val = self.button_5.value
            _btn_6_val = self.button_6.value

            if _btn_5_val:
                print("Button 5 was pressed!")
            elif _btn_6_val:
                print("Button 6 was pressed!")

            ret, frame = cap.read()
            frame_count += 1
            if ret:
                if frame_count % frame_skip != 0:
                    continue  # Skip this frame

                # Convert the color space from BGR (OpenCV default) to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Resize the frame to fit the display resolution
                frame = cv2.resize(frame, (disp.height, disp.width))

                # Give the frame to the AI
                frame_tensor = torch.from_numpy(frame).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)

                # AI Detection
                # results = detector(frame_tensor)
                with torch.no_grad():
                    results = model(frame_tensor)

                # get list of box locations
                boxes = results.xyxy[0].numpy()
                # get list of classes
                classes = boxes[:,-2]
                # get list of confidence scores
                confidence_scores = boxes[:,:4]

                # Annotate the frame
                annotated_frame = visualize_results(frame, boxes, classes, confidence_scores, max_detections=3)

                # Convert the frame to a PIL Image
                frame = Image.fromarray(annotated_frame)

                # Draw the image on the display
                disp.image(frame)
            else:
                print("Failed to grab frame")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    color_vision = ColorVision()
    color_vision.analyze_feed()