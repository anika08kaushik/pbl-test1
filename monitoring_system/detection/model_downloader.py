import urllib.request
import os

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

def download_model_if_needed(model_type="face"):
    """Downloads MediaPipe models if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    if model_type == "face":
        path = os.path.join(MODELS_DIR, "face_landmarker.task")
        url = FACE_MODEL_URL
    else:
        path = os.path.join(MODELS_DIR, "pose_landmarker.task")
        url = POSE_MODEL_URL

    if not os.path.exists(path):
        print(f"Downloading {model_type} model to {path}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")
    return path

if __name__ == "__main__":
    download_model_if_needed()
