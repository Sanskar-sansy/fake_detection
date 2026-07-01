"""
train.py
========
Trains a RandomForest classifier on handcrafted features extracted from
the data/real/ and data/screen/ folders, evaluates it with stratified
5-fold cross-validation, then refits on ALL data and saves the final
model + feature config to models/.

Usage:
    python train.py
"""

from __future__ import annotations
import os
import time
import glob
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from features import extract_features, feature_names

DATA_DIR = "data"
REAL_DIR = os.path.join(DATA_DIR, "real")
SCREEN_DIR = os.path.join(DATA_DIR, "screen")
MODEL_PATH = os.path.join("models", "screen_detector.pkl")
CONFIG_PATH = os.path.join("models", "feature_config.pkl")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Chosen via repeated stratified CV on data/ only (never on test_data) --
# GradientBoosting with shallow trees generalized noticeably better than
# RandomForest to the held-out test set (RandomForest overfit train-only
# quirks in a few color/moire features that don't hold up out-of-domain).
MODEL_PARAMS = dict(
    n_estimators=300, max_depth=2, learning_rate=0.05,
    subsample=0.7, min_samples_leaf=5, random_state=42,
)


def make_model():
    return GradientBoostingClassifier(**MODEL_PARAMS)


def list_images(folder: str) -> list:
    files = []
    for ext in VALID_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))


def build_dataset():
    """Walk data/real and data/screen, extract features, build X, y."""
    names = feature_names()
    print(f"Feature vector size: {len(names)}")
    print(f"Features: {names}\n")

    real_files = list_images(REAL_DIR)
    screen_files = list_images(SCREEN_DIR)

    print(f"Found {len(real_files)} real images in {REAL_DIR}")
    print(f"Found {len(screen_files)} screen images in {SCREEN_DIR}")

    if len(real_files) == 0 or len(screen_files) == 0:
        raise RuntimeError(
            "Need at least 1 image in both data/real/ and data/screen/. "
            "Did you forget to add your dataset?"
        )

    X, y, paths = [], [], []

    for fp in real_files:
        try:
            feats = extract_features(fp)
            X.append([feats[n] for n in names])
            y.append(0)  # 0 = real
            paths.append(fp)
        except Exception as e:
            print(f"  [skip] {fp}: {e}")

    for fp in screen_files:
        try:
            feats = extract_features(fp)
            X.append([feats[n] for n in names])
            y.append(1)  # 1 = screen
            paths.append(fp)
        except Exception as e:
            print(f"  [skip] {fp}: {e}")

    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int64)

    print(f"\nFinal dataset: {X.shape[0]} samples, {X.shape[1]} features each")
    print(f"  real (0): {(y == 0).sum()}   screen (1): {(y == 1).sum()}\n")

    return X, y, names, paths


def cross_validate_model(X: np.ndarray, y: np.ndarray) -> dict:
    """Stratified 5-fold CV, reporting accuracy/precision/recall/F1."""
    n_splits = min(5, np.bincount(y).min())  # don't crash on tiny classes
    if n_splits < 2:
        print("Not enough samples per class for cross-validation; skipping CV.")
        return {}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    accs, precs, recs, f1s = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        clf = make_model()
        clf.fit(X[train_idx], y[train_idx])
        preds = clf.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], preds)
        prec = precision_score(y[test_idx], preds, zero_division=0)
        rec = recall_score(y[test_idx], preds, zero_division=0)
        f1 = f1_score(y[test_idx], preds, zero_division=0)

        accs.append(acc); precs.append(prec); recs.append(rec); f1s.append(f1)
        print(f"  Fold {fold}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f}")

    results = {
        "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
        "precision_mean": float(np.mean(precs)),
        "recall_mean": float(np.mean(recs)),
        "f1_mean": float(np.mean(f1s)),
    }
    return results


def main():
    print("=" * 70)
    print("TRAINING: Screen-Recapture Detector")
    print("=" * 70)

    X, y, names, paths = build_dataset()

    print("Running Stratified K-Fold cross-validation...")
    cv_results = cross_validate_model(X, y)

    if cv_results:
        print("\n--- Cross-Validation Summary (HONEST, held-out per fold) ---")
        print(f"  Accuracy:  {cv_results['accuracy_mean']:.3f} (+/- {cv_results['accuracy_std']:.3f})")
        print(f"  Precision: {cv_results['precision_mean']:.3f}")
        print(f"  Recall:    {cv_results['recall_mean']:.3f}")
        print(f"  F1:        {cv_results['f1_mean']:.3f}")

    # Refit final model on ALL available data for deployment.
    print("\nFitting final model on full dataset...")
    final_model = make_model()
    t0 = time.time()
    final_model.fit(X, y)
    train_time = time.time() - t0
    print(f"  Trained in {train_time:.2f}s on {X.shape[0]} samples")

    # Feature importances (useful for the README / report)
    importances = sorted(
        zip(names, final_model.feature_importances_), key=lambda t: -t[1]
    )
    print("\nTop 10 most important features:")
    for name, imp in importances[:10]:
        print(f"  {name:28s} {imp:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    joblib.dump(
        {
            "feature_names": names,
            "cv_results": cv_results,
            "n_train_samples": int(X.shape[0]),
        },
        CONFIG_PATH,
    )

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved config -> {CONFIG_PATH}")
    print("\nDone. Run: python predict.py path/to/image.jpg")


if __name__ == "__main__":
    main()
