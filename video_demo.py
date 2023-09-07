import cv2
import tensorflow as tf
from detection_visualizer import visualize_results

PATH_TO_CKPT = "models\ssd_mobilenet_v2_320x320_coco17_tpu-8\ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model"
detector = tf.saved_model.load(PATH_TO_CKPT)

vid_capture = cv2.VideoCapture("videos/traffic_light_video.mp4")
# vid_capture = cv2.VideoCapture(0)

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
        frame_resized = cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.uint8)[tf.newaxis, ...]

        # Detection
        results = detector(frame_tensor)

        # get list of box locations
        boxes = results['detection_boxes'][0].numpy()

        # get list of classes
        classes = results['detection_classes'][0].numpy()

        # get list of confidence scores
        confidence_scores = results['detection_scores'][0].numpy()

        # Annotate the frame
        annotated_frame = visualize_results(frame_resized, boxes, classes, confidence_scores)

        cv2.imshow("Test Video", annotated_frame)

        key = cv2.waitKey(20)

        if key == ord('q'):
            break
    else:
        break

vid_capture.release()
cv2.destroyAllWindows()