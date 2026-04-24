import cv2
import PIL.Image
import numpy as np

class VideoStream:
    def __init__(self, device_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device_id)
        self.width = width
        self.height = height

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        
    def get_frame(self):
        """Captures a frame and resizes it."""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Resize for performance as requested
        frame = cv2.resize(frame, (self.width, self.height))
        return frame

    def to_st_format(self, frame):
        """Converts OpenCV BGR frame to RGB for Streamlit/PIL."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()

def annotate_frame(frame, detections, mp_results=None):
    """Draws bounding boxes and landmarks on the frame."""
    annotated = frame.copy()
    
    # YOLO Detections
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        label = f"{det['label']} {det['conf']:.2f}"
        color = (0, 255, 0) if det['label'] == 'person' else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return annotated
