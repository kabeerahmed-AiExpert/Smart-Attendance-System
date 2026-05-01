"""
================================================================================
  Recognition Module
================================================================================
  Cosine similarity matching against the embedding database.
  Uses threshold-based decision — NO softmax classification.
  Unknown faces are properly handled via thresholding.
================================================================================
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────
COSINE_SIMILARITY_THRESHOLD = 0.70


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.

    Returns:
        float in range [-1, 1]. Higher = more similar.
    """
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def recognize_face(embedding, database):
    """
    Compare an embedding against all centroids in the database.

    Decision Rule:
        If best cosine similarity >= THRESHOLD -> return identity
        Else -> return "Unknown"

    Args:
        embedding : 1D numpy array (512,).
        database  : dict {name: {"centroid": np.array, ...}}.

    Returns:
        name       : str — recognized student name or "Unknown".
        confidence : float — cosine similarity score of best match.
    """
    if not database:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_score = -1.0

    for student_name, data in database.items():
        centroid = data["centroid"]
        score = cosine_similarity(embedding, centroid)

        if score > best_score:
            best_score = score
            best_name = student_name

    # Apply threshold — do NOT force prediction if confidence is low
    if best_score < COSINE_SIMILARITY_THRESHOLD:
        return "Unknown", best_score

    return best_name, best_score


def parse_student_display_name(full_name):
    """
    Parse folder name like 'F24ARI129_Kabeer' into display-friendly parts.

    Returns:
        student_id   : str (e.g., "F24ARI129")
        display_name : str (e.g., "Kabeer")
    """
    parts = full_name.split("_", 1)
    if len(parts) > 1:
        return parts[0], parts[1]
    return full_name, full_name
