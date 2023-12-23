import torch
import cv2
from PIL import Image

default_path = "models/Custom/bestv11.pt"

class Analyzer:
    def __init__(self, model_path=default_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("WongKinYiu/yolov7", "custom", f"{model_path}", trust_repo=True)
        self.model.eval()

    def analyze_frame(self, frame, confidence_threshold=0.8):
        # Pre-process frame
        # - make the image smaller
        output_width, output_height = 320, 172

        # Crop to match display aspect
        height, width, _ = frame.shape
        new_height = int(width * (output_height / output_width))
        top = max(0, (height - new_height) // 2)
        bottom = top + new_height
        cropped_frame = frame[top:bottom, :]
        resized_frame = cv2.resize(cropped_frame, (output_width, output_height))

        # Compatibility Conversions
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)

        results = self.model(image)
        items = results.pandas().xyxy[0]
        print(items)

        scaling_factor = width / output_width
        for idx, row in items.iterrows():
            xmin = int(row["xmin"] / scaling_factor)
            ymin = int(row["ymin"] / scaling_factor)
            xmax = int(row["xmax"] / scaling_factor)
            ymax = int(row["ymax"] / scaling_factor)
            label = row["name"]
            confidence = row["confidence"]
            print(f"attempting to draw {label} {confidence}")
            confidence = 1.0
            if confidence >= confidence_threshold:
                print("success")
                cv2.rectangle(resized_frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
                cv2.putText(resized_frame, f"{label}", (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        return items, resized_frame

    def start_video_capture(self, capture_path=0, skip_frames=5):
        cap = cv2.VideoCapture(capture_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                results, modified_frame = self.analyze_frame(frame)

                cv2.imshow("Frame", modified_frame)

            # cv2.imshow("Frame", frame)

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()