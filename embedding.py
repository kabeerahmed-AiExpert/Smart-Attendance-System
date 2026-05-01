"""
================================================================================
  Embedding Module
================================================================================
  FaceNet embedding generation and database loading.
  Uses pretrained FaceNet as a pure feature extractor (NO softmax).
================================================================================
"""

import os
import pickle
import numpy as np
from keras_facenet import FaceNet

# ─────────────────────────────────────────────────────────────────────────────
# PATHS (relative to this file)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDINGS_PATH = os.path.join(BASE_DIR, "output", "face_embeddings.pkl")


def get_facenet_model():
    """Load and return the pretrained FaceNet model."""
    return FaceNet()


def generate_embedding(face_array, model):
    """
    Generate a 512-d L2-normalized embedding for a preprocessed face.

    Args:
        face_array : numpy array (160, 160, 3), normalized to [0, 1].
        model      : FaceNet model instance.

    Returns:
        embedding : 1D numpy array of shape (512,), L2-normalized.
    """
    # FaceNet expects pixel values in [0, 255]
    face_pixels = (face_array * 255).astype(np.float32)
    face_batch = np.expand_dims(face_pixels, axis=0)

    embedding = model.embeddings(face_batch)[0]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def load_embedding_database(path=None):
    """
    Load the prebuilt face embedding database from .pkl file.

    Args:
        path : Optional path to the .pkl file. Uses default if None.

    Returns:
        database : dict mapping student_name -> {"centroid": np.array, ...}
                   Returns empty dict if file not found.
    """
    if path is None:
        path = EMBEDDINGS_PATH

    if not os.path.exists(path):
        return {}

    with open(path, "rb") as f:
        database = pickle.load(f)

    return database
