import cv2
import numpy as np
import digitalio
import os
import time
import board
import adafruit_st7789
from adafruit_rgb_display import st7789  # pylint: disable=unused-import
from PIL import Image, ImageOps

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

# Get Camera Feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Convert the color space from BGR (OpenCV default) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize the frame to fit the display resolution
        frame = cv2.resize(frame, (disp.height, disp.width))

        # Convert the frame to a PIL Image
        frame = Image.fromarray(frame)

        # Draw the image on the display
        disp.image(frame)
    else:
        print("Failed to grab frame")
        break

cap.release()
cv2.destroyAllWindows()
