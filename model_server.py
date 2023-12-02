from flask import Flask, request, jsonify
# import tensorflow as tf
import torch
import cv2
import numpy as np

app = Flask(__name__)

PATH_TO_CKPT = "models\ssd_mobilenet_v2_320x320_coco17_tpu-8\ssd_mobilenet_v2_320x320_coco17_tpu-8\saved_model"
PATH_TO_CUSTOMPT = "models/Custom/bestv9.pt"

# TODO fix typeerror
# PATH_TO_CKPT = "models\centernet_mobilenetv2fpn_512x512_coco17_od\centernet_mobilenetv2_fpn_od\saved_model"

print("Loading model! Please wait...")
# detector = tf.saved_model.load(PATH_TO_CKPT)
model = torch.hub.load("WongKinYiu/yolov7", "custom", f"{PATH_TO_CUSTOMPT}", trust_repo=True)
model.eval()
print("Model loaded successfully!")

@app.route("/detect", methods=["POST"])
def detect_objects():
    frame_data = request.files["frame"].read()
    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), -1)

    # PREPROCESS FOR TENSOR VERSIONS
    # frame_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)[tf.newaxis, ...]
    # PREPROCESS FOR TORCH VERSIONS
    frame_resized = cv2.resize(frame, (640,640), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).float().div(255.0).unsqueeze(0).permute(0, 3, 1, 2)

    # Tensor Flow Detection
    # results = detector(frame_tensor)
    with torch.no_grad():
        results = model(frame_tensor)

    # get list of box locations
    # boxes = results['detection_boxes'][0].numpy()
    boxes = results.xyxy[0].numpy()

    # get list of classes
    # classes = results['detection_classes'][0].numpy()
    classes = boxes[:,-2]

    # get list of confidence scores
    # confidence_scores = results['detection_scores'][0].numpy()
    confidence_scores = boxes[:,:4]

    return jsonify({
        "boxes": boxes.tolist(),
        "classes": classes.tolist(),
        "confidence_scores": confidence_scores.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)