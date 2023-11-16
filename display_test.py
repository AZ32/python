import cv2
import numpy as np
# import digitalio
import displayio
import board
import adafruit_st7789

# Initialize TFT display
displayio.release_displays()
# cs_pin = digitalio.DigitalInOut(board.D8)
# dc_pin = digitalio.DigitalInOut(board.D24)
# reset_pin = digitalio.DigitalInOut(board.D25)
tft_cs = board.D5
tft_dc = board.D6
spi = board.SPI()
display_bus = displayio.FourWire(spi, command=tft_dc, chip_select=tft_cs, reset=board.D9)

disp = adafruit_st7789.ST7789(display_bus, width=320, height=172, rotation=180)

# Get Camera Feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (320, 172))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bitmap = displayio.Bitmap(320, 172, 3)

        palette = displayio.Palette(3)
        palette[0] = 0x000000
        palette[1] = 0xFFFFFF
        palette[2] = 0xFF0000

        for y in range(320):
            for x in range(172):
                bitmap[x, y] = frame_rgb[y][x]

        tile_grid = displayio.TileGrid(bitmap, pixel_shader=palette)
        group = displayio.Group()
        group.append(tile_grid)
        disp.show(group)

        # disp.image(frame_bytes)
    else:
        print("Failed to grab frame")
        break

cap.release()
cv2.destroyAllWindows()