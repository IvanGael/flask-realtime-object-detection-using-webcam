import cv2
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
        self.classes = self.model.names
        self.detected_classes = {cls: True for cls in self.classes.values()}

    def detect(self, frame):
        results = self.model(frame)[0]  # Get the first (and only) result
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            class_index = int(cls)
            if class_index < 0 or class_index >= len(self.classes):
                print(f"Invalid class index detected: {class_index}")
                continue

            class_name = self.classes[class_index]

            if class_name not in self.detected_classes:
                print(f"New class detected: {class_name}")
                self.detected_classes[class_name] = True

            if self.detected_classes[class_name]:
                label = f'{class_name} {conf:.2f}'
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return frame