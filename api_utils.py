import base64
import cv2
import numpy as np
import time
from uuid import uuid4

# In-memory session store
# session_id -> { "screener": data, "integrity": [logs], "suspicion_score": int }
sessions = {}

def create_session():
    session_id = str(uuid4())
    sessions[session_id] = {
        "screener": None,
        "integrity": [],
        "suspicion_score": 0,
        "start_time": time.time()
    }
    return session_id

def get_session(session_id):
    return sessions.get(session_id)

def decode_base64_frame(base64_str):
    """Decodes a base64 string into an OpenCV frame."""
    try:
        # Remove header if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None

def log_integrity_event(session_id, status, message, score_increment):
    session = get_session(session_id)
    if session:
        event = {
            "timestamp": time.strftime("%H:%M:%S"),
            "status": status,
            "message": message,
            "score_increment": score_increment
        }
        session["integrity"].append(event)
        session["suspicion_score"] = min(100, session["suspicion_score"] + score_increment)
        return event
    return None
