import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

MODELS_DIR = "models"
FACE_PROTO = os.path.join(MODELS_DIR, "deploy.prototxt")
FACE_MODEL = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
AGE_PROTO = os.path.join(MODELS_DIR, "age_deploy.prototxt")
AGE_MODEL = os.path.join(MODELS_DIR, "age_net.caffemodel")

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_INTERVALS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

# Load models globally to avoid reloading on every request
face_net = None
age_net = None

def load_models():
    global face_net, age_net
    if face_net is None or age_net is None:
        logger.info("Initializing OpenCV DNN Models...")
        if not os.path.exists(FACE_MODEL):
            raise FileNotFoundError(f"Model file not found: {FACE_MODEL}. Please run download_models.py first.")
        if not os.path.exists(AGE_MODEL):
            raise FileNotFoundError(f"Model file not found: {AGE_MODEL}. Please run download_models.py first.")
        
        face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
        logger.info("Models loaded successfully.")

def predict_age(image_bytes: bytes):
    load_models()

    # Decode image bytes
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Could not decode image")

    h, w = frame.shape[:2]

    # Detect face
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()

    # Find the most confident face
    max_confidence = 0
    box = None
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5 and confidence > max_confidence:
            max_confidence = confidence
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Add padding since the age net expects some padding around the face
            padding = 20
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            box = (x1, y1, x2, y2)

    if not box:
        return {"error": "No face detected in the image."}

    x1, y1, x2, y2 = box
    face_img = frame[y1:y2, x1:x2]

    # Predict age
    face_blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(face_blob)
    age_preds = age_net.forward()
    age_idx = age_preds[0].argmax()
    age = AGE_INTERVALS[age_idx]
    
    # We map numpy float32 to python float for JSON serialization
    confidence_val = float(age_preds[0][age_idx])

    return {
        "age_prediction": age,
        "confidence": confidence_val,
        "face_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    }
