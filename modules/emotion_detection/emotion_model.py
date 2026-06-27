import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf

EMOTIONS   = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
IMG_SIZE   = 48
MODEL_PATH = Path("modules/emotion_detection/models/emotion_model.h5")

# Load once at import time
_model      = None
_face_cascade = None


def _load_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Emotion model not found at {MODEL_PATH}.\n"
                "Run train_emotion_model.py first after downloading FER-2013."
            )
        _model = tf.keras.models.load_model(str(MODEL_PATH))
    return _model


def _load_face_detector():
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade


def detect_emotion_from_frame(frame: np.ndarray) -> dict:
    """
    Takes a raw BGR webcam frame, detects the face,
    and returns the predicted emotion with confidence scores.

    Returns:
        {
            "emotion": "happy",
            "confidence": 0.94,
            "all_scores": {"happy": 0.94, "neutral": 0.04, ...},
            "face_found": True
        }
    """
    model    = _load_model()
    detector = _load_face_detector()

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return {"emotion": None, "confidence": 0.0, "all_scores": {}, "face_found": False}

    # Use the largest face detected
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_roi    = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_roi, (IMG_SIZE, IMG_SIZE))
    face_input   = face_resized.astype("float32") / 255.0
    face_input   = np.expand_dims(face_input, axis=(0, -1))  # (1, 48, 48, 1)

    predictions = model.predict(face_input, verbose=0)[0]
    label_id    = int(np.argmax(predictions))

    return {
        "emotion":    EMOTIONS[label_id],
        "confidence": round(float(predictions[label_id]), 3),
        "all_scores": {e: round(float(p), 3) for e, p in zip(EMOTIONS, predictions)},
        "face_found": True,
        "face_box":   (x, y, w, h),
    }