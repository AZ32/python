import requests
import cv2

API_URL = "http://127.0.0.1:5000/detect"

vid_capture = cv2.VideoCapture("videos/traffic_light_video.mp4")

while vid_capture.isOpened():
    success, frame = vid_capture.read()
    if not success:
        break

    _, frame_encoded = cv2.imencode('.jpg', frame)
    response = requests.post(API_URL, files={"frame": frame_encoded.tobytes()})

    frame_data = response.content
    processed_frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), -1)

    cv2.imshow("Processed Video", processed_frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

vid_capture.release()
cv2.destroyAllWindows()