import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os
from .model_downloader import download_model_if_needed

class FaceMonitor:
    def __init__(self):
        # 1. Ensure model exists
        model_path = download_model_if_needed()
        
        # 2. Configure Tasks API
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=5,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def process(self, frame):
        """Processes frame to detect faces, head pose, and eye contact."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        results = self.detector.detect(mp_image)
        
        face_data = []
        if results.face_landmarks:
            for landmarks in results.face_landmarks:
                # 1. Eye Contact Percentage
                eye_contact_pct = self.calculate_eye_contact(landmarks)
                
                # 2. Head Pose (Pitch, Yaw)
                pose = self.get_detailed_pose(landmarks)
                
                face_data.append({
                    "eye_contact_pct": eye_contact_pct,
                    "pose": pose, # {'pitch': ..., 'yaw': ..., 'direction': ...}
                    "landmarks": landmarks
                })
        
        return face_data

    def calculate_eye_contact(self, landmarks):
        """
        Estimates eye contact % based on iris position relative to eye corners.
        Returns 0-100.
        """
        try:
            # Indices for iris centers (approximate in 478 model)
            # Left: 468, Right: 473
            # If the model has fewer landmarks, we fallback to a simpler head pose check
            if len(landmarks) < 470:
                return 100 if self.get_simple_direction(landmarks) == "center" else 0
            
            # Left Eye Iris and Corners
            l_iris = landmarks[468]
            l_inner = landmarks[133]
            l_outer = landmarks[33]
            
            # Right Eye Iris and Corners
            r_iris = landmarks[473]
            r_inner = landmarks[362]
            r_outer = landmarks[263]
            
            # Distance from iris to corners
            # We want the iris to be roughly in the middle
            def eye_score(iris, inner, outer):
                center_x = (inner.x + outer.x) / 2
                dist = abs(iris.x - center_x)
                max_dist = abs(inner.x - outer.x) / 2
                if max_dist == 0: return 1.0
                score = max(0, 1 - (dist / max_dist))
                return score

            l_score = eye_score(l_iris, l_inner, l_outer)
            r_score = eye_score(r_iris, r_inner, r_outer)
            
            avg_score = (l_score + r_score) / 2
            return int(avg_score * 100)
        except Exception:
            return 50 # Fallback

    def get_detailed_pose(self, landmarks):
        """Analyzes landmarks for Pitch (Looking Down) and Yaw (Looking Aside)."""
        # Pitch: Nose tip (1) relative to mid-eye line and chin
        nose = landmarks[1]
        forehead = landmarks[10]
        chin = landmarks[152]
        
        # Yaw: Nose relative to face edges
        left_edge = landmarks[234]
        right_edge = landmarks[454]
        
        # Calculate Pitch (Looking Down)
        # Ratio of nose height between forehead and chin
        face_height = chin.y - forehead.y
        if face_height == 0: pitch_ratio = 0.5
        else: pitch_ratio = (nose.y - forehead.y) / face_height
        
        # Calculate Yaw (Looking Aside)
        face_width = right_edge.x - left_edge.x
        if face_width == 0: yaw_ratio = 0.5
        else: yaw_ratio = (nose.x - left_edge.x) / face_width
        
        direction = "center"
        if pitch_ratio > 0.75: direction = "down"
        elif yaw_ratio < 0.35: direction = "right"
        elif yaw_ratio > 0.65: direction = "left"
        
        return {
            "pitch_ratio": pitch_ratio,
            "yaw_ratio": yaw_ratio,
            "direction": direction
        }

    def get_simple_direction(self, landmarks):
        """Fallback for orientation."""
        nose = landmarks[1].x
        left = landmarks[234].x
        right = landmarks[454].x
        width = right - left
        if width == 0: return "center"
        ratio = (nose - left) / width
        if ratio < 0.35: return "right"
        elif ratio > 0.65: return "left"
        return "center"

    def draw_landmarks(self, frame, face_data):
        """Draws visual indicators on the frame."""
        annotated = frame.copy()
        if face_data:
            face = face_data[0]
            pct = face['eye_contact_pct']
            direction = face['pose']['direction']
            
            # Color based on eye contact
            color = (0, 255, 0) if pct > 70 else (0, 255, 255) if pct > 40 else (0, 0, 255)
            
            # Display metrics on frame
            cv2.putText(annotated, f"Eye Contact: {pct}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(annotated, f"Pose: {direction.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
        return annotated
