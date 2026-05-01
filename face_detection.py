"""
================================================================================
  Face Detection Module
================================================================================
  MTCNN-based face detection for images and video frames.
  Handles cropping, resizing to 160x160, and normalization.
================================================================================
"""

import numpy as np
import cv2
from PIL import Image
from mtcnn import MTCNN

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
FACENET_INPUT_SIZE = (160, 160)
MTCNN_CONFIDENCE_THRESHOLD = 0.90


def get_detector():
    """Create and return an MTCNN detector instance."""
    return MTCNN()


def detect_faces_in_frame(rgb_frame, detector):
    """
    Detect all faces in an RGB frame using MTCNN.

    Args:
        rgb_frame : numpy array (H, W, 3) in RGB format.
        detector  : MTCNN detector instance.

    Returns:
        List of dicts, each containing:
            - "box": (x, y, w, h)
            - "confidence": float
            - "face_array": numpy array (160, 160, 3) normalized to [0, 1]
    """
    results = []

    try:
        detections = detector.detect_faces(rgb_frame)
    except Exception:
        return results

    if not detections:
        return results

    h_img, w_img = rgb_frame.shape[:2]

    for det in detections:
        conf = det["confidence"]
        if conf < MTCNN_CONFIDENCE_THRESHOLD:
            continue

        x, y, w, h = det["box"]

        # Clamp bounding box to image bounds
        x, y = max(0, x), max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)

        # Skip tiny faces
        if (x2 - x) < 10 or (y2 - y) < 10:
            continue

        # Crop face region
        face_crop = rgb_frame[y:y2, x:x2]

        # Resize to 160x160
        try:
            face_resized = cv2.resize(face_crop, FACENET_INPUT_SIZE)
        except Exception:
            continue

        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0

        results.append({
            "box": (x, y, x2 - x, y2 - y),
            "confidence": conf,
            "face_array": face_normalized,
        })

    return results


def detect_face_from_path(image_path, detector):
    """
    Detect the best face from an image file path.

    Args:
        image_path : Path to image file.
        detector   : MTCNN detector instance.

    Returns:
        face_array : numpy array (160, 160, 3) normalized [0, 1], or None.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
    except Exception:
        return None

    if img_array.ndim != 3 or img_array.shape[2] != 3:
        return None

    faces = detect_faces_in_frame(img_array, detector)

    if not faces:
        return None

    # Return face with highest confidence
    best = max(faces, key=lambda f: f["confidence"])
    return best["face_array"]
