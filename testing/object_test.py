import cv2
import mediapipe as mp

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=3,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.99,
                                   model_name="Cup")

# setup OpenCV videos
vid_capture = cv2.VideoCapture(0)

if (vid_capture.isOpened() == False):
    print("There was an error opening the file!")
else:
    fps = vid_capture.get(5)
    print(f"Video framerate: {fps} FPS")

    frame_count = vid_capture.get(7)
    print(f"Frame count: {frame_count}")

while(vid_capture.isOpened()):
    success, frame = vid_capture.read()

    if success:

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = objectron.process(frame_rgb)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                # mp_drawing.draw_landmarks(frame, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                # mp_drawing.draw_axis(frame, detected_object.rotation, detected_object.translation)

                landmarks = detected_object.landmarks_2d.landmark
                min_x = int(min([landmark.x for landmark in landmarks]) * frame.shape[1])
                max_x = int(max([landmark.x for landmark in landmarks]) * frame.shape[1])
                min_y = int(min([landmark.y for landmark in landmarks]) * frame.shape[0])
                max_y = int(max([landmark.y for landmark in landmarks]) * frame.shape[0])

                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

        cv2.imshow("Detection Window", frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
    else:
        break

vid_capture.release()
cv2.destroyAllWindows()