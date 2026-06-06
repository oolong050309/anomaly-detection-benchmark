"""Demonstration script for Robust Anomaly Defense.

This script loads a dataset, injects 20% label noise (label flips),
and trains both an undefended LightGBM model and a defended RobustDefenseWrapper model.
It showcases how the unsupervised denoising pre-filter successfully recovers performance!
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapters import load_dataset
from data.contaminate import contaminate_supervised
from models import LightGBMDetector, IForestDetector, RobustDefenseWrapper
from sklearn.metrics import roc_auc_score


def main():
    print("=" * 60)
    print("   ROBUST ANOMALY DEFENSE WORKBENCH (DEMO)   ")
    print("=" * 60)

    # 1. Load data
    dataset_name = "cardio"
    print(f"[*] Loading dataset: {dataset_name}...")
    bundle = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = bundle.as_tuple()

    print(f"    - Train size: {X_train.shape[0]} samples (anomaly rate: {np.mean(y_train)*100:.1f}%)")
    print(f"    - Test size: {X_test.shape[0]} samples (anomaly rate: {np.mean(y_test)*100:.1f}%)")

    # 2. Inject label noise (20% flip rate)
    flip_rate = 0.20
    print(f"\n[*] Injecting {flip_rate*100:.0f}% symmetric label flips into training set...")
    X_train_noisy, y_train_noisy, meta = contaminate_supervised(X_train, y_train, flip_rate, seed=42)

    # Calculate actual label overlap/correctness
    labels_changed = np.sum(y_train != y_train_noisy)
    print(f"    - Flipped {labels_changed} labels out of {len(y_train)}")

    # 3. Train Undefended Classifier
    print("\n[*] Training standard (undefended) LightGBM...")
    clf_standard = LightGBMDetector(n_estimators=100, random_state=42)
    clf_standard.fit(X_train_noisy, y_train_noisy)
    scores_standard = clf_standard.decision_function(X_test)
    auc_standard = roc_auc_score(y_test, scores_standard)
    print(f"    -> [Undefended LightGBM] AUC-ROC = {auc_standard:.4f}")

    # 4. Train Defended Classifier (Trim Strategy)
    print("\n[*] Training RobustDefenseWrapper (Strategy: 'trim')...")
    # Base supervised model
    clf_base_trim = LightGBMDetector(n_estimators=100, random_state=42)
    # Unsupervised density guide
    cleaner_trim = IForestDetector(random_state=42)
    # Wrap them
    clf_defended_trim = RobustDefenseWrapper(
        base_detector=clf_base_trim,
        unsupervised_cleaner=cleaner_trim,
        trim_rate=flip_rate,
        strategy="trim"
    )
    clf_defended_trim.fit(X_train_noisy, y_train_noisy)
    scores_trim = clf_defended_trim.decision_function(X_test)
    auc_trim = roc_auc_score(y_test, scores_trim)
    print(f"    -> [Defended LightGBM - TRIM] AUC-ROC = {auc_trim:.4f} (Change: {auc_trim - auc_standard:+.4f})")

    # 5. Train Defended Classifier (Flip Correction Strategy)
    print("\n[*] Training RobustDefenseWrapper (Strategy: 'flip')...")
    clf_base_flip = LightGBMDetector(n_estimators=100, random_state=42)
    cleaner_flip = IForestDetector(random_state=42)
    clf_defended_flip = RobustDefenseWrapper(
        base_detector=clf_base_flip,
        unsupervised_cleaner=cleaner_flip,
        trim_rate=flip_rate,
        strategy="flip"
    )
    clf_defended_flip.fit(X_train_noisy, y_train_noisy)
    scores_flip = clf_defended_flip.decision_function(X_test)
    auc_flip = roc_auc_score(y_test, scores_flip)
    print(f"    -> [Defended LightGBM - FLIP] AUC-ROC = {auc_flip:.4f} (Change: {auc_flip - auc_standard:+.4f})")

    print("\n" + "=" * 60)
    print("   DEMO COMPLETED SUCCESSFULLY!   ")
    print("=" * 60)


if __name__ == "__main__":
    main()
