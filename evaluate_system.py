"""
================================================================================
  Smart Attendance System - Evaluation Module
================================================================================
  Description : Evaluates the FaceNet embedding-based recognition system
                on unseen test images. Reports:
                  - Recognition Accuracy
                  - False Acceptance Rate (FAR)
                  - False Rejection Rate (FRR)
                  - Per-student accuracy breakdown
                  - Confusion matrix

  Methodology : Leave-one-out style evaluation:
                For each student, hold out some images and test against
                the centroid built from remaining images. Also tests
                against other students' images to measure FAR.

  Usage       : python evaluate_system.py
================================================================================
"""

import os
import sys
import pickle
import numpy as np
from PIL import Image
from collections import defaultdict

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
    EMBEDDINGS_PATH,
    OUTPUT_DIR,
    FACENET_INPUT_SIZE,
    COSINE_SIMILARITY_THRESHOLD,
    EUCLIDEAN_DISTANCE_THRESHOLD,
    PRIMARY_METRIC,
    MTCNN_CONFIDENCE_THRESHOLD,
    VALID_IMAGE_EXTENSIONS,
    SEED,
)

np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS (reuse from build_embeddings)
# ─────────────────────────────────────────────────────────────────────────────

def detect_and_crop_face(image_path, detector, target_size=FACENET_INPUT_SIZE):
    """Detect face using MTCNN, crop, resize to 160x160, normalize to [0,1]."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)

        if img_array.ndim != 3 or img_array.shape[2] != 3:
            return None
    except Exception:
        return None

    try:
        detections = detector.detect_faces(img_array)
    except Exception:
        return None

    if len(detections) == 0:
        return None

    best = max(detections, key=lambda d: d["confidence"])

    if best["confidence"] < MTCNN_CONFIDENCE_THRESHOLD:
        return None

    x, y, w, h = best["box"]
    x, y = max(0, x), max(0, y)
    x2 = min(img_array.shape[1], x + w)
    y2 = min(img_array.shape[0], y + h)

    if (x2 - x) < 10 or (y2 - y) < 10:
        return None

    face_crop = img_array[y:y2, x:x2]
    face_img = Image.fromarray(face_crop).resize(target_size, Image.BILINEAR)
    face_array = np.array(face_img, dtype=np.float32) / 255.0

    return face_array


def generate_embedding(face_array, facenet_model):
    """Generate L2-normalized 128-d FaceNet embedding."""
    face_pixels = (face_array * 255).astype(np.float32)
    face_batch = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.embeddings(face_batch)[0]
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def euclidean_distance(a, b):
    """Compute Euclidean distance between two vectors."""
    return np.linalg.norm(a - b)


def recognize(embedding, database, metric="cosine"):
    """
    Recognize face against database.
    Returns (predicted_name, score).
    """
    best_name = "Unknown"
    best_score = -1.0 if metric == "cosine" else float("inf")

    for name, data in database.items():
        centroid = data["centroid"]
        if metric == "cosine":
            score = cosine_similarity(embedding, centroid)
            if score > best_score:
                best_score = score
                best_name = name
        else:
            score = euclidean_distance(embedding, centroid)
            if score < best_score:
                best_score = score
                best_name = name

    # Apply threshold
    if metric == "cosine":
        if best_score < COSINE_SIMILARITY_THRESHOLD:
            return "Unknown", best_score
    else:
        if best_score > EUCLIDEAN_DISTANCE_THRESHOLD:
            return "Unknown", best_score

    return best_name, best_score


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_with_holdout(dataset_dir, facenet_model, detector, holdout_ratio=0.3):
    """
    Evaluate the system using a hold-out strategy:
      1. For each student, randomly hold out ~30% of images as test set
      2. Build centroid from the remaining ~70%
      3. Test each held-out image against ALL centroids
      4. Compute accuracy, FAR, FRR

    Args:
        dataset_dir    : Path to dataset with student folders.
        facenet_model  : FaceNet model instance.
        detector       : MTCNN detector instance.
        holdout_ratio  : Fraction of images to hold out for testing.

    Returns:
        metrics : Dictionary with accuracy, FAR, FRR, and per-class results.
    """
    print("\n" + "=" * 60)
    print("  Building Evaluation Database (Train/Test Split)")
    print("=" * 60)

    class_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    # ── Step 1: Generate all embeddings per student ──
    all_student_embeddings = {}

    for student_name in class_names:
        student_path = os.path.join(dataset_dir, student_name)
        image_files = [
            f for f in os.listdir(student_path)
            if f.lower().endswith(VALID_IMAGE_EXTENSIONS)
        ]

        embeddings = []
        for img_file in image_files:
            img_path = os.path.join(student_path, img_file)
            face = detect_and_crop_face(img_path, detector)
            if face is None:
                continue
            emb = generate_embedding(face, facenet_model)
            embeddings.append(emb)

        if len(embeddings) >= 2:  # Need at least 2 (1 train, 1 test)
            all_student_embeddings[student_name] = np.array(embeddings)
            print(f"  [OK] {student_name}: {len(embeddings)} embeddings")
        else:
            print(f"  [SKIP] {student_name}: Only {len(embeddings)} embeddings (need >=2)")

    # ── Step 2: Split into train/test and build centroids ──
    print("\n" + "=" * 60)
    print("  Splitting Embeddings (Train / Test)")
    print("=" * 60)

    train_database = {}
    test_samples = []  # List of (true_label, embedding)

    for student_name, embeddings in all_student_embeddings.items():
        n = len(embeddings)
        n_test = max(1, int(n * holdout_ratio))
        n_train = n - n_test

        # Shuffle
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]

        train_embs = embeddings[train_idx]
        test_embs = embeddings[test_idx]

        # Compute centroid from training embeddings
        centroid = np.mean(train_embs, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        train_database[student_name] = {
            "centroid": centroid,
            "all_embeddings": train_embs,
            "num_images": n,
            "num_faces_found": n_train,
        }

        # Add test samples
        for emb in test_embs:
            test_samples.append((student_name, emb))

        print(f"  {student_name}: train={n_train}, test={n_test}")

    print(f"\n  Total test samples: {len(test_samples)}")
    print(f"  Students in database: {len(train_database)}")

    # ── Step 3: Run recognition on test set ──
    print("\n" + "=" * 60)
    print("  Running Recognition on Test Set")
    print("=" * 60)

    # Counters
    total = 0
    correct = 0
    false_accepts = 0    # Wrong identity predicted (not Unknown, but wrong person)
    false_rejects = 0    # Correct identity rejected as Unknown

    per_class_results = defaultdict(lambda: {"total": 0, "correct": 0, "fa": 0, "fr": 0})
    confusion = defaultdict(lambda: defaultdict(int))

    for true_label, embedding in test_samples:
        predicted, score = recognize(embedding, train_database, metric=PRIMARY_METRIC)

        total += 1
        per_class_results[true_label]["total"] += 1
        confusion[true_label][predicted] += 1

        if predicted == true_label:
            # Correct recognition
            correct += 1
            per_class_results[true_label]["correct"] += 1

        elif predicted == "Unknown":
            # False Rejection: known person classified as Unknown
            false_rejects += 1
            per_class_results[true_label]["fr"] += 1

        else:
            # False Acceptance: known person classified as WRONG person
            false_accepts += 1
            per_class_results[true_label]["fa"] += 1

    # ── Step 4: Compute metrics ──
    accuracy = correct / total if total > 0 else 0.0
    far = false_accepts / total if total > 0 else 0.0
    frr = false_rejects / total if total > 0 else 0.0

    metrics = {
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
        "total": total,
        "correct": correct,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "per_class": dict(per_class_results),
        "confusion": dict(confusion),
    }

    return metrics, train_database


def print_evaluation_report(metrics):
    """Print a comprehensive evaluation report."""

    print("\n")
    print("+" + "=" * 62 + "+")
    print("|              EVALUATION REPORT                               |")
    print("+" + "=" * 62 + "+")

    # ── Overall Metrics ──
    print(f"\n  {'='*55}")
    print(f"  OVERALL METRICS")
    print(f"  {'='*55}")
    print(f"  Total test samples     : {metrics['total']}")
    print(f"  Correct predictions    : {metrics['correct']}")
    print(f"  False Acceptances (FA) : {metrics['false_accepts']}")
    print(f"  False Rejections (FR)  : {metrics['false_rejects']}")
    print()
    print(f"  Recognition Accuracy   : {metrics['accuracy']*100:.2f}%")
    print(f"  False Acceptance Rate  : {metrics['far']*100:.2f}%")
    print(f"  False Rejection Rate   : {metrics['frr']*100:.2f}%")
    print(f"  {'='*55}")

    # ── Per-Class Breakdown ──
    print(f"\n  {'='*55}")
    print(f"  PER-STUDENT BREAKDOWN")
    print(f"  {'='*55}")
    print(f"  {'Student':<30} {'Total':>5} {'Correct':>8} {'FA':>5} {'FR':>5} {'Acc':>7}")
    print(f"  {'-'*55}")

    for student, data in sorted(metrics["per_class"].items()):
        acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
        print(f"  {student:<30} {data['total']:>5} {data['correct']:>8} "
              f"{data['fa']:>5} {data['fr']:>5} {acc:>6.1f}%")

    # ── Confusion Matrix (text) ──
    print(f"\n  {'='*55}")
    print(f"  CONFUSION MATRIX (True rows x Predicted cols)")
    print(f"  {'='*55}")

    all_labels = sorted(set(
        list(metrics["confusion"].keys()) +
        [pred for row in metrics["confusion"].values() for pred in row.keys()]
    ))

    # Print header
    hdr_label = "True\\Pred"
    header = f"  {hdr_label:<20}"
    for label in all_labels:
        short = label.split("_")[-1][:8] if "_" in label else label[:8]
        header += f" {short:>8}"
    print(header)
    sep_len = 20 + 9 * len(all_labels)
    print(f"  {'-' * sep_len}")

    # Print rows
    for true_label in all_labels:
        if true_label == "Unknown":
            continue
        parts = true_label.split("_")
        short_name = parts[-1][:18] if len(parts) > 1 else true_label[:18]
        row = f"  {short_name:<20}"
        for pred_label in all_labels:
            count = metrics["confusion"].get(true_label, {}).get(pred_label, 0)
            row += f" {count:>8}"
        print(row)

    # ── Quality Assessment ──
    print(f"\n  {'='*55}")
    print(f"  QUALITY ASSESSMENT")
    print(f"  {'='*55}")

    acc = metrics["accuracy"] * 100
    far = metrics["far"] * 100
    frr = metrics["frr"] * 100

    if acc >= 90:
        print(f"  Accuracy  : EXCELLENT ({acc:.1f}%)")
    elif acc >= 80:
        print(f"  Accuracy  : GOOD ({acc:.1f}%)")
    elif acc >= 70:
        print(f"  Accuracy  : FAIR ({acc:.1f}%) - Consider adding more images")
    else:
        print(f"  Accuracy  : POOR ({acc:.1f}%) - Need more training data")

    if far <= 2:
        print(f"  FAR       : EXCELLENT ({far:.1f}%) - Very few false identifications")
    elif far <= 5:
        print(f"  FAR       : GOOD ({far:.1f}%)")
    else:
        print(f"  FAR       : HIGH ({far:.1f}%) - Increase similarity threshold")

    if frr <= 5:
        print(f"  FRR       : EXCELLENT ({frr:.1f}%) - Very few misses")
    elif frr <= 15:
        print(f"  FRR       : ACCEPTABLE ({frr:.1f}%)")
    else:
        print(f"  FRR       : HIGH ({frr:.1f}%) - Decrease similarity threshold or add more images")

    print(f"  {'='*55}")


def save_evaluation_results(metrics, filepath):
    """Save evaluation metrics to a text file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Smart Attendance System - Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Metric       : {PRIMARY_METRIC}\n")
        f.write(f"Threshold    : {COSINE_SIMILARITY_THRESHOLD if PRIMARY_METRIC == 'cosine' else EUCLIDEAN_DISTANCE_THRESHOLD}\n\n")
        f.write(f"Total Samples     : {metrics['total']}\n")
        f.write(f"Correct           : {metrics['correct']}\n")
        f.write(f"False Accepts     : {metrics['false_accepts']}\n")
        f.write(f"False Rejects     : {metrics['false_rejects']}\n\n")
        f.write(f"Accuracy          : {metrics['accuracy']*100:.2f}%\n")
        f.write(f"FAR               : {metrics['far']*100:.2f}%\n")
        f.write(f"FRR               : {metrics['frr']*100:.2f}%\n\n")

        f.write("Per-Student Results:\n")
        f.write("-" * 50 + "\n")
        for student, data in sorted(metrics["per_class"].items()):
            acc = data["correct"] / data["total"] * 100 if data["total"] > 0 else 0
            f.write(f"  {student}: {data['correct']}/{data['total']} correct "
                    f"({acc:.1f}%), FA={data['fa']}, FR={data['fr']}\n")

    print(f"  [OK] Results saved to: {filepath}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Run the complete evaluation pipeline."""
    print("\n")
    print("+" + "=" * 62 + "+")
    print("|   SMART ATTENDANCE - SYSTEM EVALUATION                       |")
    print("|   Testing FaceNet Recognition Accuracy, FAR, FRR             |")
    print("+" + "=" * 62 + "+")
    print()

    # ── Initialize models ──
    print("=" * 60)
    print("  Initializing Models")
    print("=" * 60)

    print("  [INFO] Loading MTCNN...")
    detector = MTCNN()
    print("  [OK] MTCNN ready")

    print("  [INFO] Loading FaceNet...")
    facenet_model = FaceNet()
    print("  [OK] FaceNet ready")

    # ── Run evaluation ──
    print("\n" + "=" * 60)
    print(f"  Running Hold-Out Evaluation (30% test)")
    print(f"  Metric: {PRIMARY_METRIC.upper()}")
    print(f"  Threshold: {COSINE_SIMILARITY_THRESHOLD if PRIMARY_METRIC == 'cosine' else EUCLIDEAN_DISTANCE_THRESHOLD}")
    print("=" * 60)

    metrics, _ = evaluate_with_holdout(
        DATASET_DIR, facenet_model, detector, holdout_ratio=0.3
    )

    # ── Print report ──
    print_evaluation_report(metrics)

    # ── Save results ──
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.txt")
    save_evaluation_results(metrics, results_path)

    print("\n  [DONE] Evaluation complete!")


if __name__ == "__main__":
    main()
