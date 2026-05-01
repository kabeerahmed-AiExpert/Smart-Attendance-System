"""
================================================================================
  Smart Attendance System - Embedding Generator
================================================================================
  Description : Processes the face dataset, detects faces using MTCNN,
                generates 128-d FaceNet embeddings for each student,
                computes centroid embeddings, and saves to .pkl file.

  Pipeline    : Image -> MTCNN Face Detection -> Crop 160x160 -> Normalize
                -> FaceNet Embedding -> Mean Centroid -> Save Database

  Usage       : python build_embeddings.py
================================================================================
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS (with error handling)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from mtcnn import MTCNN
except ImportError:
    print("[ERROR] MTCNN not installed. Run: pip install mtcnn")
    sys.exit(1)

try:
    from keras_facenet import FaceNet
except ImportError:
    print("[ERROR] keras-facenet not installed. Run: pip install keras-facenet")
    sys.exit(1)

from config import (
    DATASET_DIR,
    OUTPUT_DIR,
    EMBEDDINGS_PATH,
    FACENET_INPUT_SIZE,
    EMBEDDING_DIM,
    VALID_IMAGE_EXTENSIONS,
    MTCNN_CONFIDENCE_THRESHOLD,
    SEED,
)

np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FACE DETECTION MODULE
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_crop_face(image_path, detector, target_size=FACENET_INPUT_SIZE):
    """
    Detect face(s) in an image using MTCNN and crop the best face.

    Args:
        image_path  : Path to the input image file.
        detector    : MTCNN detector instance.
        target_size : Tuple (H, W) for output face size (default: 160x160).

    Returns:
        face_array  : Numpy array of shape (160, 160, 3), normalized to [0,1].
                      Returns None if no face detected or image is corrupted.
    """
    try:
        # Load image using PIL (handles various formats safely)
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        # Validate image dimensions
        if img_array.ndim != 3 or img_array.shape[2] != 3:
            print(f"    [SKIP] Invalid image dimensions: {image_path}")
            return None

    except Exception as e:
        print(f"    [SKIP] Corrupted image: {os.path.basename(image_path)} -> {e}")
        return None

    # Detect faces
    try:
        detections = detector.detect_faces(img_array)
    except Exception as e:
        print(f"    [SKIP] MTCNN failed on {os.path.basename(image_path)}: {e}")
        return None

    if len(detections) == 0:
        print(f"    [WARN] No face detected in {os.path.basename(image_path)}")
        return None

    # Handle multiple faces: select the one with highest confidence
    if len(detections) > 1:
        print(f"    [INFO] Multiple faces ({len(detections)}) in {os.path.basename(image_path)}, using best")

    best = max(detections, key=lambda d: d["confidence"])

    # Skip low-confidence detections
    if best["confidence"] < MTCNN_CONFIDENCE_THRESHOLD:
        print(f"    [SKIP] Low confidence ({best['confidence']:.2f}) in {os.path.basename(image_path)}")
        return None

    # Extract bounding box
    x, y, w, h = best["box"]

    # Clamp to image bounds (MTCNN can return negative values)
    x, y = max(0, x), max(0, y)
    x2 = min(img_array.shape[1], x + w)
    y2 = min(img_array.shape[0], y + h)

    # Validate crop dimensions
    if (x2 - x) < 10 or (y2 - y) < 10:
        print(f"    [SKIP] Face too small in {os.path.basename(image_path)}")
        return None

    # Crop face region
    face_crop = img_array[y:y2, x:x2]

    # Resize to FaceNet input size (160x160)
    face_img = Image.fromarray(face_crop).resize(target_size, Image.BILINEAR)
    face_array = np.array(face_img, dtype=np.float32)

    # Normalize pixel values to [0, 1] — consistent preprocessing
    face_array = face_array / 255.0

    return face_array


# ─────────────────────────────────────────────────────────────────────────────
# 2. EMBEDDING GENERATION MODULE
# ─────────────────────────────────────────────────────────────────────────────

def generate_embedding(face_array, facenet_model):
    """
    Generate a 128-dimensional embedding for a preprocessed face.

    Args:
        face_array    : Numpy array of shape (160, 160, 3), normalized [0,1].
        facenet_model : Loaded FaceNet model instance.

    Returns:
        embedding : 1D numpy array of shape (128,), L2-normalized.
    """
    # FaceNet expects pixel values in [0, 255] range, uint8-like
    # Convert back from [0,1] to [0,255] for the model's internal preprocessing
    face_pixels = (face_array * 255).astype(np.float32)

    # Add batch dimension: (1, 160, 160, 3)
    face_batch = np.expand_dims(face_pixels, axis=0)

    # Generate embedding using FaceNet
    embedding = facenet_model.embeddings(face_batch)

    # Return flattened, L2-normalized embedding
    embedding = embedding[0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


# ─────────────────────────────────────────────────────────────────────────────
# 3. DATABASE BUILDER MODULE
# ─────────────────────────────────────────────────────────────────────────────

def build_face_database(dataset_dir, detector, facenet_model):
    """
    Process the entire dataset and build an embedding database.

    For each student:
      1. Load all images from their folder
      2. Detect and crop faces
      3. Generate FaceNet embeddings
      4. Compute the mean embedding (centroid)

    Args:
        dataset_dir   : Root directory with one folder per student.
        detector      : MTCNN detector instance.
        facenet_model : Loaded FaceNet model instance.

    Returns:
        database : Dictionary with structure:
            {
                "student_name": {
                    "centroid": np.array (128,),
                    "all_embeddings": [np.array, ...],
                    "num_images": int,
                    "num_faces_found": int
                },
                ...
            }
    """
    database = {}

    # Get sorted list of student folders
    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    if len(class_names) == 0:
        print("[ERROR] No student folders found in dataset directory!")
        return database

    print(f"\n  Found {len(class_names)} students in dataset:\n")

    total_images = 0
    total_faces = 0

    for idx, student_name in enumerate(class_names):
        student_path = os.path.join(dataset_dir, student_name)

        # Get all valid image files
        image_files = [
            f for f in os.listdir(student_path)
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]

        if len(image_files) == 0:
            print(f"  [{idx+1:02d}] {student_name}: [SKIP] No images found")
            continue

        print(f"  [{idx+1:02d}] Processing: {student_name} ({len(image_files)} images)")

        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(student_path, img_file)

            # Detect and crop face
            face = detect_and_crop_face(img_path, detector)
            if face is None:
                continue

            # Generate embedding
            embedding = generate_embedding(face, facenet_model)
            embeddings.append(embedding)

        if len(embeddings) == 0:
            print(f"       -> [WARNING] No valid face embeddings for {student_name}")
            continue

        # Compute centroid (mean embedding)
        all_embeddings = np.array(embeddings)
        centroid = np.mean(all_embeddings, axis=0)

        # L2-normalize the centroid
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        # Store in database
        database[student_name] = {
            "centroid": centroid,
            "all_embeddings": all_embeddings,
            "num_images": len(image_files),
            "num_faces_found": len(embeddings),
        }

        total_images += len(image_files)
        total_faces += len(embeddings)

        print(f"       -> {len(embeddings)}/{len(image_files)} faces embedded | "
              f"Embedding shape: {centroid.shape}")

    print(f"\n  {'='*50}")
    print(f"  Database Summary:")
    print(f"    Students enrolled  : {len(database)}")
    print(f"    Total images       : {total_images}")
    print(f"    Total faces found  : {total_faces}")
    print(f"    Embedding dimension: {EMBEDDING_DIM}")
    print(f"  {'='*50}")

    return database


def save_database(database, filepath):
    """
    Save the embedding database to a pickle file.

    Args:
        database : Dictionary of student embeddings.
        filepath : Path to save the .pkl file.
    """
    with open(filepath, "wb") as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size = os.path.getsize(filepath) / 1024  # KB
    print(f"\n  [OK] Database saved to: {filepath}")
    print(f"       File size: {file_size:.1f} KB")


def load_database(filepath):
    """
    Load the embedding database from a pickle file.

    Args:
        filepath : Path to the .pkl file.

    Returns:
        database : Dictionary of student embeddings.
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] Embeddings file not found: {filepath}")
        return None

    with open(filepath, "rb") as f:
        database = pickle.load(f)

    print(f"  [OK] Database loaded: {len(database)} students")
    return database


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    Execute the complete embedding generation pipeline.
    """
    print("\n")
    print("+" + "=" * 62 + "+")
    print("|   SMART ATTENDANCE - FaceNet EMBEDDING GENERATOR             |")
    print("|   Pipeline: MTCNN Detection -> FaceNet 128-d Embeddings      |")
    print("+" + "=" * 62 + "+")
    print()

    # ── Step 1: Validate dataset directory ──
    print("=" * 60)
    print("  STEP 1: Validating Dataset Directory")
    print("=" * 60)

    if not os.path.exists(DATASET_DIR):
        print(f"  [ERROR] Dataset directory not found: {DATASET_DIR}")
        sys.exit(1)

    class_count = len([
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ])
    print(f"  [OK] Dataset directory found: {DATASET_DIR}")
    print(f"  [OK] Student folders: {class_count}")

    # ── Step 2: Initialize MTCNN detector ──
    print("\n" + "=" * 60)
    print("  STEP 2: Initializing MTCNN Face Detector")
    print("=" * 60)

    detector = MTCNN()
    print("  [OK] MTCNN initialized successfully")

    # ── Step 3: Load FaceNet model ──
    print("\n" + "=" * 60)
    print("  STEP 3: Loading Pretrained FaceNet Model")
    print("=" * 60)

    print("  [INFO] Downloading/loading FaceNet weights (first run may take time)...")
    facenet_model = FaceNet()
    print(f"  [OK] FaceNet loaded successfully")
    print(f"       Output embedding dimension: {EMBEDDING_DIM}")

    # ── Step 4: Build face database ──
    print("\n" + "=" * 60)
    print("  STEP 4: Building Face Embedding Database")
    print("=" * 60)

    database = build_face_database(DATASET_DIR, detector, facenet_model)

    if len(database) == 0:
        print("\n  [ERROR] No embeddings generated. Check your dataset!")
        sys.exit(1)

    # ── Step 5: Save database ──
    print("\n" + "=" * 60)
    print("  STEP 5: Saving Embedding Database")
    print("=" * 60)

    save_database(database, EMBEDDINGS_PATH)

    # ── Step 6: Verification ──
    print("\n" + "=" * 60)
    print("  STEP 6: Verification")
    print("=" * 60)

    print("\n  Enrolled Students:")
    print("  " + "-" * 50)
    for i, (name, data) in enumerate(database.items()):
        print(f"    [{i+1:02d}] {name}")
        print(f"         Images: {data['num_images']} | Faces: {data['num_faces_found']} | "
              f"Centroid norm: {np.linalg.norm(data['centroid']):.4f}")

    # ── Final Summary ──
    print("\n")
    print("+" + "=" * 62 + "+")
    print("|                  EMBEDDING GENERATION COMPLETE                |")
    print("+" + "=" * 62 + "+")
    print(f"|  Students enrolled : {len(database):<41}|")
    print(f"|  Embeddings file   : {os.path.basename(EMBEDDINGS_PATH):<41}|")
    print(f"|  Location          : {OUTPUT_DIR:<41}|")
    print("|                                                              |")
    print("|  Next step: Run realtime_recognition.py                      |")
    print("+" + "=" * 62 + "+")


if __name__ == "__main__":
    main()
