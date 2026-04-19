from ultralytics import YOLO
import cv2

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        # This will download the model mapping to the local dir if not present
        self.model = YOLO(model_name)
        # We only care about person (0) and cell phone (67) in COCO dataset
        self.target_classes = [0, 67] 

    def detect(self, frame):
        """Detects people and cell phones in the frame."""
        results = self.model(frame, verbose=False)[0]
        detections = []
        
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in self.target_classes:
                conf = float(box.conf[0])
                if conf > 0.3: # Threshold
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    label = self.model.names[cls]
                    detections.append({
                        "label": label,
                        "conf": conf,
                        "bbox": [x1, y1, x2, y2]
                    })
        return detections
