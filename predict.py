"""
predict.py
==========
Loads the trained model and prints a single float probability that the
given image is a "screen recapture" (1.0) vs a "real photo" (0.0).

Usage:
    python predict.py path/to/image.jpg

Output (stdout, ONLY this line):
    0.93
"""

from __future__ import annotations
import sys
import time
import joblib
import numpy as np

from features import extract_features

MODEL_PATH = "models/screen_detector.pkl"
CONFIG_PATH = "models/feature_config.pkl"


def load_model():
    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)
    return model, config["feature_names"]


def predict_image(image_path: str, model=None, names=None) -> float:
    """Return probability in [0,1] that image_path is a screen recapture."""
    if model is None or names is None:
        model, names = load_model()

    feats = extract_features(image_path)
    vec = np.array([[feats[n] for n in names]], dtype=np.float64)

    proba = model.predict_proba(vec)[0]
    # class index 1 corresponds to "screen" (see train.py: y=1 for screen)
    class_idx = list(model.classes_).index(1)
    return float(proba[class_idx])


def benchmark_latency(image_path: str, n_runs: int = 20) -> dict:
    """Helper to measure end-to-end per-image latency.

    Includes feature extraction + inference, NOT model loading (which
    happens once per process and is amortized across many images in a
    real deployment).
    """
    model, names = load_model()

    # warm-up run (avoids counting first-call overhead/caching effects)
    predict_image(image_path, model, names)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_image(image_path, model, names)
        times.append((time.perf_counter() - t0) * 1000.0)  # ms

    return {
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "min_ms": float(np.min(times)),
        "max_ms": float(np.max(times)),
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    if "--benchmark" in sys.argv:
        stats = benchmark_latency(image_path)
        print(stats, file=sys.stderr)
        return

    try:
        prob = predict_image(image_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # ONLY print the float, exactly as required by the spec.
    print(f"{prob:.4f}")


if __name__ == "__main__":
    main()
