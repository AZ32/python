import cv2
import numpy as np
import digitalio
import board
import adafruit_st7789

# Initialize TFT display
cs_pin = digitalio.DigitalInOut(board.CEO)
dc_pin = digitalio.DigitalInOut(board.D25)
reset_pin = digitalio.DigitalInOut(board.D24)
spi = board.SPI()
disp = adafruit_st7789.ST7789(spi, cs=cs_pin, dc=dc_pin, rst=reset_pin, width=320, height=172)

# Get Camera Feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (172, 320))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_bytes = frame_rgb.tobytes()

        disp.image(frame_bytes)
    else:
        print("Failed to grab frame")
        break

cap.release()
cv2.destroyAllWindows()