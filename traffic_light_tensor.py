import cv2
import tensorflow as tf
from traffic_light import get_traffic_light_color
from sklearn.cluster import KMeans

# on Andrew's machine, tensorflow uninstalled numpy 1.25.1 --> 1.24.3
# Models to Test:
# EfficientDet D3 896x896
# CenterNet ResnetV2 FPN 512x512
# SSD MobileNet v2 320x320


label_map = {1: 'Person', 3: 'Car', 10: 'Traffic Light', 13: 'Stop Sign', 52: "Banana", 53: "Apple", 55: "Orange"}

def dominant_color(image, k=1):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)

    dominant = kmeans.cluster_centers_[0]
    return dominant

def visualize_results(image, boxes, classes, confidence_scores, threshold=0.5):
    height, width, _ = image.shape

    for i in range(boxes.shape[0]):
        if confidence_scores[i] >= threshold:
            class_id = int(classes[i])
            if class_id in label_map:
                label = label_map[class_id]
                color_label = ""
                box = boxes[i]

                top, left, bottom, right = box
                left = int(left * width)
                right = int(right * width)
                top = int(top * height)
                bottom = int(bottom * height)

                cropped_image = image[top:bottom, left:right]
                # Draw any extras + figure out color
                if class_id == 10:
                    section, color_label = get_traffic_light_color(cropped_image)
                    image[top:bottom, left:right] = section
                else:
                    dom_color = dominant_color(cropped_image)
                    print(f"Dominant Color: {dom_color}")


                # Draw bounding box
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

                # Display label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_rect_left, label_rect_top = int(left), int(top) - label_size[1]
                label_rect_right, label_rect_bottom = left + label_size[0], int(top)
                color_label_rect_left, color_label_rect_top = int(left), int(top) - (label_size[1] * 2)
                color_label_rect_right, color_label_rect_bottom = left + label_size[0], int(top) - label_size[1]
                cv2.rectangle(image, (label_rect_left, label_rect_top - 10), (label_rect_right, label_rect_bottom), (0, 0, 0), -1)
                cv2.putText(image, label, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.rectangle(image, (color_label_rect_left, color_label_rect_top - 10), (color_label_rect_right, color_label_rect_bottom - 10), (0, 0, 0), -1)
                cv2.putText(image, color_label, (left + 5, top - label_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return image

# Path to our downloaded detection model
PATH_TO_CKPT = "models\centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8\saved_model"

# Load model and image
detector = tf.saved_model.load(PATH_TO_CKPT)
image = cv2.imread("test_3.jpg")

width = 500
height = 500

down_size = (width, height)
image = cv2.resize(image, down_size, interpolation=cv2.INTER_LINEAR)

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)[tf.newaxis, ...]

# Detection
results = detector(image_tensor)

# get list of box locations
boxes = results['detection_boxes'][0].numpy()

# get list of classes
classes = results['detection_classes'][0].numpy()

# get list of confidence scores
confidence_scores = results['detection_scores'][0].numpy()

# # TODO: incorporate color detection code here
# color_data = None

# Add visuals
image = visualize_results(image, boxes, classes, confidence_scores, 0.45)

# Display result
cv2.imshow("Traffic Light Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()