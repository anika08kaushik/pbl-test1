from detection.yolo import YOLODetector
from detection.mediapipe_utils import FaceMonitor
from logic.rules import BehaviorEngine
from utils.video import VideoStream

def process_video(video_path):
    yolo = YOLODetector()
    face_monitor = FaceMonitor()
    rules = BehaviorEngine()

    stream = VideoStream(video_path)

    all_stats = []
    frame_count = 0

    while True:
        frame = stream.get_frame()
        if frame is None:
            break

        yolo_dets = yolo.detect(frame)
        face_results = face_monitor.process(frame)

        stats = rules.analyze(yolo_dets, face_results)

        all_stats.append(stats)
        frame_count += 1

    stream.release()

    # 👉 simple summary (important for API response)
    if not all_stats:
        return {"error": "No frames processed"}

    avg_suspicion = sum(s["suspicion_score"] for s in all_stats) / len(all_stats)

    return {
        "frames_processed": frame_count,
        "avg_suspicion_score": round(avg_suspicion, 2),
        "final_status": "Good" if avg_suspicion < 30 else "Suspicious"
    }