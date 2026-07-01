"""
evaluate.py
===========
Evaluates the trained model on a separate, fully held-out test folder
(images the model never saw during training/CV). This is what produces
the HONEST accuracy number for the report.

Expected layout (adjust paths via command-line args if yours differ):

    test_data/
        real/      <- held-out real photos
        screen/    <- held-out screen recaptures

Usage:
    python evaluate.py --real_dir test_data/real --screen_dir test_data/screen
"""

from __future__ import annotations
import argparse
import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from features import extract_features
from predict import load_model, predict_image

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def list_images(folder: str) -> list:
    files = []
    for ext in VALID_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(files))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", default="test_data/real")
    parser.add_argument("--screen_dir", default="test_data/screen")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    model, names = load_model()

    real_files = list_images(args.real_dir)
    screen_files = list_images(args.screen_dir)

    if not real_files and not screen_files:
        print(f"No images found in {args.real_dir} or {args.screen_dir}.")
        print("Create a held-out test folder with real/ and screen/ subfolders first.")
        return

    y_true, y_pred, y_prob, paths = [], [], [], []

    for fp in real_files:
        p = predict_image(fp, model, names)
        y_true.append(0); y_prob.append(p); y_pred.append(1 if p >= args.threshold else 0)
        paths.append(fp)

    for fp in screen_files:
        p = predict_image(fp, model, names)
        y_true.append(1); y_prob.append(p); y_pred.append(1 if p >= args.threshold else 0)
        paths.append(fp)

    y_true = np.array(y_true); y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("=" * 60)
    print(f"HELD-OUT TEST SET EVALUATION  (n={len(y_true)}, threshold={args.threshold})")
    print("=" * 60)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"\nConfusion matrix [rows=true, cols=pred] (0=real, 1=screen):")
    print(cm)

    print("\nMisclassified images:")
    any_wrong = False
    for fp, t, p, prob in zip(paths, y_true, y_pred, y_prob):
        if t != p:
            any_wrong = True
            true_label = "real" if t == 0 else "screen"
            pred_label = "real" if p == 0 else "screen"
            print(f"  {fp}: true={true_label} pred={pred_label} (prob={prob:.3f})")
    if not any_wrong:
        print("  None! 100% on this held-out set.")


if __name__ == "__main__":
    main()
