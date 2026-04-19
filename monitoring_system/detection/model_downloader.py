import urllib.request
import os

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_landmarker.task")

def download_model_if_needed():
    """Downloads the MediaPipe Face Landmarker model if it doesn't exist."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading model to {MODEL_PATH}...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # We'll use a slightly safer download than raw urllib if possible, but this is simple.
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    return MODEL_PATH

if __name__ == "__main__":
    download_model_if_needed()
