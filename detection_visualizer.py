import cv2
import tensorflow as tf
from traffic_light import get_traffic_light_color
from sklearn.cluster import KMeans
import pandas as pd

# on Andrew's machine, tensorflow uninstalled numpy 1.25.1 --> 1.24.3
# Models to Test:
# EfficientDet D3 896x896
# CenterNet ResnetV2 FPN 512x512
# SSD MobileNet v2 320x320


label_map = {1: 'Person', 3: 'Car', 10: 'Traffic Light', 13: 'Stop Sign', 52: "Banana", 53: "Apple", 55: "Orange"}
index = ["color", "color_name", "hex", "R", "G", "B"]
csv = pd.read_csv("Simplified_colors.csv", names=index, header=None)

# Calculate distance to get color name
def get_color_name(B, G, R):
    minimum = 10000
    # d = abs(Red — ithRedColor) + (Green — ithGreenColor) + (Blue — ithBlueColor)
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

def dominant_color(image, k=1):
    pixels = image.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    dominant = kmeans.cluster_centers_[0]
    return dominant

def fast_dominant_color(image):
    # Resize image to a very small size
    small = cv2.resize(image, (10,10), interpolation=cv2.INTER_LINEAR)
    return small.mean(axis=0).mean(axis=0)

def crop_image(image, crop_factor=0.6):
    result = image
    height, width, _ = image.shape 

    spacer = (1 - crop_factor) / 2

    # Transform result
    top, left, bottom, right = (spacer, spacer, spacer + crop_factor, spacer + crop_factor) # 0.2, 0.2, 0.8, 0.8
    left = int(left * width)
    right = int(right * width)
    top = int(top * height)
    bottom = int(bottom * height)

    result = image[top:bottom, left:right]

    return result

def attach_to_original(image, original_image, scale=(0.2, 0.2, 0.8, 0.8)):
    top, left, bottom, right = scale
    height, width, _ = original_image.shape 
    left = int(left * width)
    right = int(right * width)
    top = int(top * height)
    bottom = int(bottom * height)
    original_image[top:bottom, left:right] = image
    return image

def visualize_results(image, boxes, classes, confidence_scores, threshold=0.5, max_detections=10):
    height, width, _ = image.shape # (500, 450, color_info)

    sorted_indices = confidence_scores.argsort()[::-1][:max_detections]

    for i in sorted_indices:
        if confidence_scores[i] >= threshold:
            class_id = int(classes[i])
            if class_id in label_map:
                label = label_map[class_id]
                color_label = ""
                box = boxes[i]

                top, left, bottom, right = box # (0.6, 0.4, 0.7, 0.5)
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
                    # dom_color = dominant_color(cropped_image)
                    dom_color = fast_dominant_color(cropped_image)
                    color_label = get_color_name(dom_color[2], dom_color[1], dom_color[0])
                    # print(f"Dominant Color: {dom_color}")


                # Draw bounding box
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), 2)

                # Display label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                label_rect_left, label_rect_top = int(left), int(top) - label_size[1]
                label_rect_right, label_rect_bottom = left + label_size[0], int(top)
                color_label_rect_left, color_label_rect_top = int(left), int(top) - (label_size[1] * 2)
                color_label_rect_right, color_label_rect_bottom = left + label_size[0], int(top) - label_size[1]
                cv2.rectangle(image, (label_rect_left, label_rect_top - 10), (label_rect_right+10, label_rect_bottom), (0, 0, 0), -1)
                cv2.putText(image, label, (left + 5, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.rectangle(image, (color_label_rect_left, color_label_rect_top - 10), (color_label_rect_right+10, color_label_rect_bottom - 10), (0, 0, 0), -1)
                cv2.putText(image, color_label, (left + 5, top - label_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return image

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# # Path to our downloaded detection model
# PATH_TO_CKPT = "models\centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8\saved_model"

# # Load model and image
# detector = tf.saved_model.load(PATH_TO_CKPT)
# image = cv2.imread("Person.jpg")

# width = 450
# height = 450

# down_size = (width, height)
# image = cv2.resize(image, down_size, interpolation=cv2.INTER_LINEAR)

# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image_tensor = tf.convert_to_tensor(image_rgb, dtype=tf.uint8)[tf.newaxis, ...]

# # Detection
# results = detector(image_tensor)

# # get list of box locations
# boxes = results['detection_boxes'][0].numpy()

# # get list of classes
# classes = results['detection_classes'][0].numpy()

# # get list of confidence scores
# confidence_scores = results['detection_scores'][0].numpy()

# # # TODO: incorporate color detection code here
# # color_data = None

# # Add visuals
# image = visualize_results(image, boxes, classes, confidence_scores, 0.45)

# # Display result
# cv2.imshow("Traffic Light Detection", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()