import time

class BehaviorEngine:
    def __init__(self):
        self.reset()

    def analyze(self, yolo_detections, face_data):
        """Analyzes frame data to update session metrics and detect suspicious behavior."""
        current_time = time.time()
        results = {
            "eye_contact_pct": 0,
            "face_present": False,
            "phone_detected": False,
            "multiple_people": False,
            "status": "Normal",
            "alerts": []
        }
        
        # 1. Phone Detection
        phone_count = sum(1 for d in yolo_detections if d['label'] == 'cell phone')
        results["phone_detected"] = phone_count > 0
        if results["phone_detected"]:
            self._add_log("Phone usage detected", 40)

        # 2. Face Analytics
        num_faces = len(face_data)
        results["face_present"] = num_faces > 0
        results["multiple_people"] = num_faces > 1
        
        if num_faces == 0:
            if self.absence_start is None:
                self.absence_start = current_time
            elif current_time - self.absence_start > 3.0: # 3s absence threshold
                self._add_log("User not present", 30)
        else:
            self.absence_start = None
            # Extract analytics from primary face
            face = face_data[0]
            results["eye_contact_pct"] = face["eye_contact_pct"]
            
            # 3. Looking Down / Distracted Logic
            pose = face["pose"]["direction"]
            if pose == "down":
                if self.looking_down_start is None:
                    self.looking_down_start = current_time
                elif current_time - self.looking_down_start > 5.0:
                    alert_msg = "Looking Away / Distracted"
                    # Combine with phone for "High Confidence Cheating"
                    if results["phone_detected"]:
                        alert_msg = "High Confidence Cheating (Phone + Down)"
                        self._add_log(alert_msg, 50)
                    else:
                        self._add_log(alert_msg, 20)
            else:
                self.looking_down_start = None
                
            # 4. General Looking Away (Left/Right)
            if pose in ["left", "right"]:
                if self.gaze_away_start is None:
                    self.gaze_away_start = current_time
                elif current_time - self.gaze_away_start > 5.0:
                    self._add_log(f"Looking {pose} away", 15)
            else:
                self.gaze_away_start = None

        # 5. Multiple People Rule
        if num_faces > 1:
            self._add_log("Multiple people detected", 50)

        # Determine overall status
        if self.suspicion_score > 60:
            results["status"] = "Highly Suspicious"
        elif self.suspicion_score > 30:
            results["status"] = "Suspicious"
        
        results["suspicion_score"] = self.suspicion_score
        return results

    def _add_log(self, message, score_inc):
        """Adds a log and updates score with a 5s cooldown per message type."""
        now = time.time()
        if message not in self.last_log_time or now - self.last_log_time[message] > 5.0:
            self.logs.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "message": message,
                "score_increment": score_inc
            })
            self.suspicion_score = min(100, self.suspicion_score + score_inc)
            self.last_log_time[message] = now

    def get_logs(self):
        return self.logs

    def reset(self):
        self.suspicion_score = 0
        self.logs = []
        self.last_log_time = {}
        self.absence_start = None
        self.looking_down_start = None
        self.gaze_away_start = None
