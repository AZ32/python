import requests
import cv2
import numpy as np
import displayio
import board
import adafruit_st7789
from detection_visualizer import visualize_results, crop_image, attach_to_original

API_URL = "http://127.0.0.1:5000/detect"


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


# vid_capture = cv2.VideoCapture("videos/traffic_light_video.mp4")
# vid_capture = cv2.VideoCapture("videos/new_york.mp4")
# vid_capture = cv2.VideoCapture("videos/Fruits.mp4")
vid_capture = cv2.VideoCapture(0)

while vid_capture.isOpened():
    success, frame = vid_capture.read()
    if not success:
        break

    frame = cv2.resize(frame, (320, 172), interpolation=cv2.INTER_LINEAR)

    original_height, original_width, _ = frame.shape
    cropped_area = crop_image(frame)
    cropped_height, cropped_width, _ = cropped_area.shape

    # frame_resized = cv2.resize(cropped_area, (320, 320), interpolation=cv2.INTER_LINEAR)
    frame_resized = frame
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # palette = displayio.Palette(3)
    # palette[0] = 0x000000
    # palette[1] = 0xFFFFFF
    # palette[2] = 0xFF0000

    _, frame_encoded = cv2.imencode('.jpg', frame_rgb)
    response = requests.post(API_URL, files={"frame": frame_encoded.tobytes()})

    # print(f"TESTING: {response.content}")

    detection_results = response.json()

    boxes = np.array(detection_results["boxes"])
    classes = np.array(detection_results["classes"])
    confidence_scores = np.array(detection_results["confidence_scores"])

    # Annotate the frame
    annotated_frame = visualize_results(frame_resized, boxes, classes, confidence_scores, max_detections=3)

    annotated_frame = cv2.resize(annotated_frame, (cropped_width, cropped_height), interpolation=cv2.INTER_LINEAR)
    frame = attach_to_original(annotated_frame, frame)
    frame_rgb = (frame // 32).astype("uint16")

    frame_rgb = (frame[...,0] << 11) | (frame_rgb[...,1] << 5) | (frame_rgb[...,2])

    # frame = cv2.resize(frame, (original_width // 4, original_height // 4), interpolation=cv2.INTER_LINEAR)

    # frame_data = response.content
    # processed_frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), -1)

    # Set the frame size to the resolution of the display of our prototype, change as needed.
    # frame = cv2.resize(frame, (320, 172), interpolation=cv2.INTER_LINEAR)
    # frame = cv2.resize(frame, (720, 480), interpolation=cv2.INTER_LINEAR)

    bitmap = displayio.Bitmap(320, 172, 65536) # 16-bit palette
    for y in range(320):
        for x in range(172):
            bitmap[x, y] = frame_rgb[y][x]


    # cv2.imshow("Processed Video", frame)
    # key = cv2.waitKey(1)

    # if key == ord("q"):
    #     break

    tile_grid = displayio.TileGrid(bitmap, pixel_shader=displayio.ColorConverter())
    group = displayio.Group()
    group.append(tile_grid)
    disp.show(group)

vid_capture.release()
cv2.destroyAllWindows()