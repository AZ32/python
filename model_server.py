from flask import Flask, request, jsonify
import tensorflow as tf
import cv2
import numpy as np

app = Flask(__name__)

PATH_TO_CKPT = "models\ssd_mobilenet_v2_320x320_coco17_tpu-8\ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model"

# TODO fix typeerror
# PATH_TO_CKPT = "models\centernet_mobilenetv2fpn_512x512_coco17_od\centernet_mobilenetv2_fpn_od\saved_model"

print("Loading model! Please wait...")
detector = tf.saved_model.load(PATH_TO_CKPT)
print("Model loaded successfully!")

@app.route("/detect", methods=["POST"])
def detect_objects():
    frame_data = request.files["frame"].read()
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), -1)

    # original_height, original_width, _ = frame.shape
    # cropped_area = crop_image(frame)
    # cropped_height, cropped_width, _ = cropped_area.shape

    # # if not flag:
    # #     flag = True
    # #     print(f"{cropped_height}, {cropped_width}")

    # frame_resized = cv2.resize(cropped_area, (320, 320), interpolation=cv2.INTER_LINEAR)
    # frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)[tf.newaxis, ...]

    # Detection
    results = detector(frame_tensor)

    # get list of box locations
    boxes = results['detection_boxes'][0].numpy()

    # get list of classes
    classes = results['detection_classes'][0].numpy()

    # get list of confidence scores
    confidence_scores = results['detection_scores'][0].numpy()

    return jsonify({
        "boxes": boxes.tolist(),
        "classes": classes.tolist(),
        "confidence_scores": confidence_scores.tolist()
    })

    # # Annotate the frame
    # annotated_frame = visualize_results(frame_resized, boxes, classes, confidence_scores)

    # annotated_frame = cv2.resize(annotated_frame, (cropped_width, cropped_height), interpolation=cv2.INTER_LINEAR)
    # frame = attach_to_original(annotated_frame, frame)

    # frame = cv2.resize(frame, (original_width // 4, original_height // 4), interpolation=cv2.INTER_LINEAR)

    # _, buf = cv2.imencode(".jpg", frame)
    # return buf.tobytes()

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)