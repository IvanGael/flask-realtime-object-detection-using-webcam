import torch
import cv2

class YOLO:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.classes = self.model.names
        self.detected_classes = {cls: True for cls in self.classes}

    def detect(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0].numpy()

        for *xyxy, conf, cls in detections:
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
                frame = cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                frame = cv2.putText(frame, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def toggle_detection(self, cls_name, enable):
        if cls_name in self.detected_classes:
            self.detected_classes[cls_name] = enable
        else:
            print(f"Class {cls_name} not found in detected_classes")
