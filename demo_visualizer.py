import cv2
import numpy as np
import pandas as pd

# Assuming csv is already loaded as shown in your provided snippet
csv = pd.read_csv("Simplified_colors.csv", names=["color", "color_name", "hex", "R", "G", "B"], header=None)

def get_color_name(B, G, R, csv):
    minimum = 10000
    cname = "N/A"
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
        if (d <= minimum):
            minimum = d
            cname = csv.loc[i, "color_name"]
    return cname

def get_optimized_color_name(B, G, R, colors_df):
    # Vectorized approach to find the closest color
    colors = colors_df[['R', 'G', 'B']].to_numpy()
    color = np.array([R, G, B])
    distances = np.sqrt(np.sum((colors - color)**2, axis=1))
    index_min = np.argmin(distances)
    return colors_df.loc[index_min, "color_name"]

def visualize_results(frame, boxes, classes, confidence_scores, max_detections=5):
    sorted_indices = np.argsort(confidence_scores)[::-1]
    count = 0

    for i in sorted_indices:
        if count >= max_detections:
            break
        box = boxes[i]
        class_name = classes[i]
        confidence = confidence_scores[i]

        x_min, y_min, x_max, y_max = box

        # Extract the color of the detected object
        bbox = frame[int(y_min):int(y_max), int(x_min):int(x_max)]
        avg_color_per_row = np.average(bbox, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        B, G, R = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
        color_name = get_color_name(B, G, R, csv)

        # Draw bounding box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        # Prepare text for class name and confidence
        label = f'{class_name}: {confidence*100:.2f}%'
        color_label = f'Color: {color_name}'

        # Determine text size to create a background for it
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (color_label_width, color_label_height), _ = cv2.getTextSize(color_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw background for text
        cv2.rectangle(frame, (int(x_min), int(y_min) - 35), (int(x_min) + max(label_width, color_label_width), int(y_min)), (255, 0, 0), cv2.FILLED)

        # Put text on the frame
        cv2.putText(frame, label, (int(x_min), int(y_min) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, color_label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        count += 1

    return frame

def optimized_visualize_results(frame, boxes, classes, confidence_scores, max_detections=5):
    # Sort detections by confidence score in descending order
    indices = np.argsort(confidence_scores)[::-1]
    count = 0
    
    for i in indices:
        if count >= max_detections:
            break
        box = boxes[i].astype(int)
        class_name = classes[i]
        confidence = confidence_scores[i]

        # if confidence < 0.5:
        #     continue
        
        # Extract bounding box coordinates
        x_min, y_min, x_max, y_max = box
        
        # Simplified color detection by analyzing the center pixel of the bounding box
        center_color = frame[y_min + (y_max - y_min) // 2, x_min + (x_max - x_min) // 2]
        B, G, R = center_color
        color_name = get_optimized_color_name(B, G, R, csv)
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), 2)

        # Prepare text for class name and confidence
        label = f'{class_name}: {confidence*100:.2f}%'
        color_label = f'Color: {color_name}'

        # Determine text size to create a background for it
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (color_label_width, color_label_height), _ = cv2.getTextSize(color_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw background for text
        cv2.rectangle(frame, (int(x_min), int(y_min) - 35), (int(x_min) + max(label_width, color_label_width), int(y_min)), (0, 0, 0), cv2.FILLED)

        # Put text on the frame
        cv2.putText(frame, label, (int(x_min), int(y_min) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, color_label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        count += 1
    
    return frame

def improved_visualize_results(frame, boxes, classes, confidence_scores, max_detections=5, crop_margin=0.2):
    sorted_indices = np.argsort(confidence_scores)[::-1]
    count = 0

    for i in sorted_indices:
        if count >= max_detections:
            break
        box = boxes[i]
        class_name = classes[i]
        confidence = confidence_scores[i]

        x_min, y_min, x_max, y_max = box

        center_x_min = x_min + (x_max - x_min) * crop_margin
        center_x_max = x_max - (x_max - x_min) * crop_margin
        center_y_min = y_min + (y_max - y_min) * crop_margin
        center_y_max = y_max - (y_max - y_min) * crop_margin

        # Extract the color of the detected object
        bbox = frame[int(center_y_min):int(center_y_max), int(center_x_min):int(center_x_max)]
        avg_color_per_row = np.average(bbox, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        B, G, R = int(avg_color[0]), int(avg_color[1]), int(avg_color[2])
        color_name = get_color_name(B, G, R, csv)

        # Draw bounding box
        cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 0), 2)

        # Prepare text for class name and confidence
        label = f'{class_name}: {confidence*100:.2f}%'
        color_label = f'Color: {color_name}'

        # Determine text size to create a background for it
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (color_label_width, color_label_height), _ = cv2.getTextSize(color_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Draw background for text
        cv2.rectangle(frame, (int(x_min), int(y_min) - 35), (int(x_min) + max(label_width, color_label_width), int(y_min)), (0, 0, 0), cv2.FILLED)

        # Put text on the frame
        cv2.putText(frame, label, (int(x_min), int(y_min) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, color_label, (int(x_min), int(y_min) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        count += 1

    return frame