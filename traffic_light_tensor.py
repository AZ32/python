import cv2
import tensorflow as tf

# on Andrew's machine, tensorflow uninstalled numpy 1.25.1 --> 1.24.3

label_map = {10: 'Traffic Light'}

def visualize_results(image, colors, boxes, classes, confidence_scores):
    height, width, _ = image.shape

    for i in range(boxes.shape[0]):
        class_id = int(classes[i])
        if class_id in label_map:
            pass
            #dar the rest

# Path to our downloaded detection model
PATH_TO_CKPT = "[name of model folder]/saved_model"

# Load model and image
detector = tf.saved_model.load(PATH_TO_CKPT)
image = cv2.imread("test_1.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = tf.convert_to_tensor([image_rgb])

# Detection
results = detector(image_tensor)

# get list of box locations
boxes = results['detection_boxes'][0].numpy()

# get list of classes
classes = results['detection_classes'][0].numpy()

# get list of confidence scores
confidence_scores = results['detection_scores'][0].numpy()

# TODO: incorporate color detection code here
color_data = None

# Add visuals
image = visualize_results(image, color_data, boxes, classes, confidence_scores)

# Display result
cv2.imshow("Traffic Light Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()