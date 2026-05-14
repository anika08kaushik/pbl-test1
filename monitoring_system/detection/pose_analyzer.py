import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import os
from .model_downloader import download_model_if_needed

class PoseAnalyzer:
    def __init__(self):
        """Initialize MediaPipe Pose detector for body language analysis."""
        try:
            model_path = download_model_if_needed(model_type="pose")
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.mp_pose = mp.solutions.pose
            self.initialized = True
        except Exception as e:
            print(f"Pose detector initialization failed: {e}")
            self.initialized = False
        
        self.prev_landmarks = None
        self.fidget_frames = 0
        self.total_frames = 0
        self.confidence_history = []
        
    def analyze_pose(self, frame):
        """Analyze body posture and movement for confidence assessment."""
        if not self.initialized:
            return self._default_result()
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)
        
        if not results.pose_landmarks:
            return self._default_result()
        
        landmarks = results.pose_landmarks[0]
        self.total_frames += 1
        
        result = {
            "posture_score": self._calculate_posture_score(landmarks),
            "confidence_level": "neutral",
            "fidgeting_detected": False,
            "body_language_cues": [],
            "shoulder_alignment": 0,
            "leaning": "center"
        }
        
        # Posture analysis
        posture = self._analyze_posture(landmarks)
        result.update(posture)
        
        # Fidgeting detection
        if self.prev_landmarks:
            movement = self._calculate_movement(landmarks, self.prev_landmarks)
            if movement > 0.02:
                self.fidget_frames += 1
                result["fidgeting_detected"] = True
                result["body_language_cues"].append("excessive_movement")
        
        self.prev_landmarks = landmarks
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(result)
        result["confidence_level"] = confidence
        result["confidence_score"] = confidence
        
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > 30:
            self.confidence_history.pop(0)
        
        return result
    
    def _calculate_posture_score(self, landmarks):
        """Calculate posture score based on shoulder and spine alignment."""
        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
            
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
            shoulder_alignment = max(0, 1 - shoulder_diff * 5)
            
            hip_mid_y = (left_hip.y + right_hip.y) / 2
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
            spine_vertical = abs(nose.y - hip_mid_y)
            
            return int((shoulder_alignment * 0.5 + min(1.0, spine_vertical) * 0.5) * 100)
        except:
            return 70
    
    def _analyze_posture(self, landmarks):
        """Detailed posture analysis."""
        try:
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
            
            shoulder_mid_x = (left_shoulder.x + right_shoulder.x) / 2
            leaning = "center"
            if nose.x < shoulder_mid_x - 0.05:
                leaning = "forward"
            elif nose.x > shoulder_mid_x + 0.05:
                leaning = "backward"
            
            shoulder_alignment = 1 - abs(left_shoulder.y - right_shoulder.y) * 3
            shoulder_alignment = max(0, min(1, shoulder_alignment))
            
            cues = []
            if leaning == "forward":
                cues.append("leaning_forward_engaged")
            elif leaning == "backward":
                cues.append("leaning_back_relaxed_or_disengaged")
            
            if shoulder_alignment < 0.7:
                cues.append("poor_posture_slouching")
            
            return {
                "leaning": leaning,
                "shoulder_alignment": int(shoulder_alignment * 100),
                "body_language_cues": cues
            }
        except:
            return {"leaning": "center", "shoulder_alignment": 70, "body_language_cues": []}
    
    def _calculate_movement(self, curr, prev):
        """Calculate movement between frames."""
        try:
            key_points = [0, 11, 12, 23, 24]
            total_movement = 0
            for idx in key_points:
                curr_lm = curr[idx]
                prev_lm = prev[idx]
                dist = ((curr_lm.x - prev_lm.x)**2 + (curr_lm.y - prev_lm.y)**2)**0.5
                total_movement += dist
            return total_movement / len(key_points)
        except:
            return 0
    
    def _calculate_confidence(self, pose_data):
        """Determine confidence level based on posture and behavior."""
        score = 0
        
        if pose_data.get("posture_score", 0) > 75:
            score += 2
        elif pose_data.get("posture_score", 0) < 60:
            score -= 1
        
        if pose_data.get("leaning") == "forward":
            score += 1
        elif pose_data.get("leaning") == "backward":
            score -= 1
        
        if pose_data.get("fidgeting_detected"):
            score -= 2
        
        if pose_data.get("shoulder_alignment", 0) > 80:
            score += 1
        elif pose_data.get("shoulder_alignment", 0) < 60:
            score -= 1
        
        if score >= 2:
            return "confident"
        elif score <= -2:
            return "nervous"
        else:
            return "neutral"
    
    def get_average_confidence(self):
        """Get average confidence over recent frames."""
        if not self.confidence_history:
            return "neutral"
        confident_count = self.confidence_history.count("confident")
        nervous_count = self.confidence_history.count("nervous")
        total = len(self.confidence_history)
        
        if confident_count / total > 0.5:
            return "confident"
        elif nervous_count / total > 0.4:
            return "nervous"
        return "neutral"
    
    def get_fidgeting_rate(self):
        """Calculate fidgeting rate as percentage."""
        if self.total_frames == 0:
            return 0
        return int((self.fidget_frames / self.total_frames) * 100)
    
    def _default_result(self):
        return {
            "posture_score": 0,
            "confidence_level": "unknown",
            "fidgeting_detected": False,
            "body_language_cues": [],
            "shoulder_alignment": 0,
            "leaning": "unknown",
            "confidence_score": "unknown"
        }
    
    def draw_pose_annotations(self, frame, pose_data):
        """Draw pose analysis info on frame."""
        if not pose_data:
            return frame
        
        annotated = frame.copy()
        y_offset = 90
        
        color = (0, 255, 0) if pose_data.get("confidence_level") == "confident" else \
                (0, 165, 255) if pose_data.get("confidence_level") == "neutral" else \
                (0, 0, 255)
        
        cv2.putText(annotated, f"Confidence: {pose_data.get('confidence_level', 'N/A').upper()}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_offset += 25
        
        cv2.putText(annotated, f"Posture: {pose_data.get('posture_score', 0)}%", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 25
        
        if pose_data.get("fidgeting_detected"):
            cv2.putText(annotated, "Fidgeting Detected", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return annotated
