import requests
import cv2
import numpy as np
from detection_visualizer import visualize_results, crop_image, attach_to_original

API_URL = "http://127.0.0.1:5000/detect"

vid_capture = cv2.VideoCapture("videos/traffic_light_video.mp4")
# vid_capture = cv2.VideoCapture(0)

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

    _, frame_encoded = cv2.imencode('.jpg', frame_rgb)
    response = requests.post(API_URL, files={"frame": frame_encoded.tobytes()})

    # print(f"TESTING: {response.content}")

    detection_results = response.json()

    boxes = np.array(detection_results["boxes"])
    classes = np.array(detection_results["classes"])
    confidence_scores = np.array(detection_results["confidence_scores"])

    # Annotate the frame
    annotated_frame = visualize_results(frame_resized, boxes, classes, confidence_scores, max_detections=1)

    annotated_frame = cv2.resize(annotated_frame, (cropped_width, cropped_height), interpolation=cv2.INTER_LINEAR)
    frame = attach_to_original(annotated_frame, frame)

    # frame = cv2.resize(frame, (original_width // 4, original_height // 4), interpolation=cv2.INTER_LINEAR)

    # frame_data = response.content
    # processed_frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), -1)

    # Set the frame size to the resolution of the display of our prototype, change as needed.
    # frame = cv2.resize(frame, (320, 172), interpolation=cv2.INTER_LINEAR)

    cv2.imshow("Processed Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

vid_capture.release()
cv2.destroyAllWindows()