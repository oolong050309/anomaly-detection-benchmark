"""Robust Anomaly Defense Wrapper for Supervised Detectors under Label Contamination.

Implements two-stage unsupervised pre-filtering (denoising) to clean/correct
polluted labels before fitting highly-overfitting supervised classifiers (e.g. LightGBM, XGBoost).
Supported strategies:
1. "trim": Drop highly suspicious contradictory label samples.
2. "flip": Symmetrically correct/flip the labels back based on density anomaly scores.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from models.base import SupervisedDetector


class RobustDefenseWrapper(SupervisedDetector):
    """Robust Defense Wrapper for Supervised Detectors.

    Parameters
    ----------
    base_detector : SupervisedDetector
        The supervised detector instance (e.g., LightGBMDetector, XGBoostDetector).
    unsupervised_cleaner : BaseDetector
        The unsupervised detector instance used for density/outlier scoring (e.g., IForestDetector, ECODDetector).
    trim_rate : float, default=0.20
        Proportion of training samples targeted for denoising/correction.
    strategy : str, default="trim"
        The defense strategy to use: "trim" (remove suspicious) or "flip" (correct labels).
    random_state : int | None, default=42
    """

    def __init__(
        self,
        base_detector: SupervisedDetector,
        unsupervised_cleaner: Any,
        trim_rate: float = 0.20,
        strategy: str = "trim",
        random_state: int | None = 42,
    ) -> None:
        super().__init__(contamination=base_detector.contamination, random_state=random_state)
        self.base_detector = base_detector
        self.unsupervised_cleaner = unsupervised_cleaner
        if strategy not in ("trim", "flip"):
            raise ValueError(f"strategy must be 'trim' or 'flip', got {strategy}")
        self.trim_rate = float(trim_rate)
        self.strategy = strategy

    def _fit(self, X: np.ndarray, y: np.ndarray | None = None, **kwargs: Any) -> None:
        if y is None:
            raise ValueError("RobustDefenseWrapper requires y labels to perform denoising.")

        y_arr = np.asarray(y).ravel()
        n_samples = X.shape[0]

        # Step 1: Fit the unsupervised density estimator to score all instances
        # This estimator completely ignores the noisy y labels, fitting purely on X features
        self.unsupervised_cleaner.fit(X)
        scores = self.unsupervised_cleaner.decision_function(X)  # Higher scores = more anomalous

        # Step 2: Adaptive Sample Selection / Label Denoising
        y_clean = np.copy(y_arr)
        keep_mask = np.ones(n_samples, dtype=bool)

        n_to_clean = int(round(n_samples * self.trim_rate))
        
        if n_to_clean > 0:
            # We split denoising equally between potential False Positives and False Negatives
            y1_idx = np.flatnonzero(y_arr == 1)
            y0_idx = np.flatnonzero(y_arr == 0)

            n_clean_y1 = min(n_to_clean // 2, len(y1_idx))
            n_clean_y0 = min(n_to_clean // 2, len(y0_idx))

            # Case A: Labeled as Anomaly (1) but has extremely LOW anomaly score (looks very normal)
            if len(y1_idx) > 0 and n_clean_y1 > 0:
                y1_scores = scores[y1_idx]
                # Sort indices of y1 by anomaly score in ascending order (lowest anomaly scores are most suspicious)
                suspicious_y1_idx = y1_idx[np.argsort(y1_scores)[:n_clean_y1]]
                if self.strategy == "trim":
                    keep_mask[suspicious_y1_idx] = False
                elif self.strategy == "flip":
                    y_clean[suspicious_y1_idx] = 0

            # Case B: Labeled as Normal (0) but has extremely HIGH anomaly score (looks very anomalous)
            if len(y0_idx) > 0 and n_clean_y0 > 0:
                y0_scores = scores[y0_idx]
                # Sort indices of y0 by anomaly score in descending order (highest anomaly scores are most suspicious)
                suspicious_y0_idx = y0_idx[np.argsort(y0_scores)[-n_clean_y0:]]
                if self.strategy == "trim":
                    keep_mask[suspicious_y0_idx] = False
                elif self.strategy == "flip":
                    y_clean[suspicious_y0_idx] = 1

        # Step 3: Train the base supervised classifier on the cleaned data
        if self.strategy == "trim":
            X_clean = X[keep_mask]
            y_clean = y_clean[keep_mask]
        else:
            X_clean = X

        # Safety Fallback: Ensure the cleaned subset still contains both 0 and 1 classes
        unique_classes = np.unique(y_clean)
        if len(unique_classes) < 2:
            # If trimming removed an entire class, fall back to fitting on original noisy data
            self.base_detector.fit(X, y_arr, **kwargs)
        else:
            self.base_detector.fit(X_clean, y_clean, **kwargs)

    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        # Step 4: Outsource inference directly to the robustly trained base supervised detector
        return self.base_detector.decision_function(X)
