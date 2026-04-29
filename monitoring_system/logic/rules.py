import time

class BehaviorEngine:
    def __init__(self):
        self.reset()
        self.behavior_observations = []
        self.confidence_scores = []
        self.posture_scores = []
        self.fidgeting_detected_frames = 0
        self.total_frames_analyzed = 0

    def analyze(self, yolo_detections, face_data, pose_data=None):
        """Analyzes frame data to update session metrics, detect suspicious behavior, and track behavioral cues."""
        current_time = time.time()
        results = {
            "eye_contact_pct": 0,
            "face_present": False,
            "phone_detected": False,
            "multiple_people": False,
            "status": "Normal",
            "alerts": [],
            "behavior": {
                "confidence_level": "unknown",
                "posture_score": 0,
                "fidgeting_rate": 0,
                "body_language_cues": [],
                "leaning": "unknown"
            }
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

        # 5. Pose Analysis (Body Language & Confidence)
        if pose_data:
            self.total_frames_analyzed += 1
            results["behavior"]["confidence_level"] = pose_data.get("confidence_level", "unknown")
            results["behavior"]["posture_score"] = pose_data.get("posture_score", 0)
            results["behavior"]["fidgeting_rate"] = pose_data.get("fidgeting_detected", False)
            results["behavior"]["body_language_cues"] = pose_data.get("body_language_cues", [])
            results["behavior"]["leaning"] = pose_data.get("leaning", "unknown")
            
            if pose_data.get("confidence_level"):
                self.confidence_scores.append(pose_data["confidence_level"])
            if pose_data.get("posture_score"):
                self.posture_scores.append(pose_data["posture_score"])
            if pose_data.get("fidgeting_detected"):
                self.fidgeting_detected_frames += 1
            
            # Log behavior observations periodically
            if self.total_frames_analyzed % 10 == 0:
                observation = self._create_behavior_observation(pose_data)
                if observation:
                    self.behavior_observations.append({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "observation": observation
                    })

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

    def _create_behavior_observation(self, pose_data):
        """Create a concise behavior observation from pose data."""
        observations = []
        
        confidence = pose_data.get("confidence_level", "unknown")
        if confidence == "confident":
            observations.append("appears confident and composed")
        elif confidence == "nervous":
            observations.append("showing signs of nervousness")
        
        posture = pose_data.get("posture_score", 0)
        if posture > 80:
            observations.append("excellent upright posture")
        elif posture < 60:
            observations.append("poor posture detected")
        
        if pose_data.get("fidgeting_detected"):
            observations.append("fidgeting/notable movement")
        
        leaning = pose_data.get("leaning", "center")
        if leaning == "forward":
            observations.append("leaning forward - engaged")
        elif leaning == "backward":
            observations.append("leaning back - possibly disengaged")
        
        if observations:
            return "; ".join(observations)
        return None

    def get_behavior_summary(self):
        """Generate a summary of behavioral observations for the report."""
        if not self.confidence_scores:
            return {
                "overall_confidence": "unknown",
                "avg_posture_score": 0,
                "fidgeting_rate": 0,
                "behavior_observations": []
            }
        
        confident_count = self.confidence_scores.count("confident")
        nervous_count = self.confidence_scores.count("nervous")
        total = len(self.confidence_scores)
        
        if confident_count / total > 0.5:
            overall = "confident"
        elif nervous_count / total > 0.4:
            overall = "nervous"
        else:
            overall = "neutral"
        
        avg_posture = sum(self.posture_scores) / len(self.posture_scores) if self.posture_scores else 0
        fidget_rate = (self.fidgeting_detected_frames / self.total_frames_analyzed * 100) if self.total_frames_analyzed > 0 else 0
        
        return {
            "overall_confidence": overall,
            "avg_posture_score": round(avg_posture, 1),
            "fidgeting_rate": round(fidget_rate, 1),
            "behavior_observations": self.behavior_observations[-10:] if self.behavior_observations else []
        }

    def reset(self):
        self.suspicion_score = 0
        self.logs = []
        self.last_log_time = {}
        self.absence_start = None
        self.looking_down_start = None
        self.gaze_away_start = None
        self.behavior_observations = []
        self.confidence_scores = []
        self.posture_scores = []
        self.fidgeting_detected_frames = 0
        self.total_frames_analyzed = 0
