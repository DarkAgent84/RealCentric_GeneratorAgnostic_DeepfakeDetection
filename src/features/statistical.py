"""
statistical.py — Statistical Envelope Features
================================================
Implements Component 1 from the Mathematical Formulation paper exactly.

Feature vector: f_stat = [μ, σ², H, d] ∈ ℝ⁴

where:
  μ  = mean intensity        = (1/HW) Σ I(i,j)
  σ² = intensity variance    = (1/HW) Σ (I(i,j) - μ)²
  H  = intensity entropy     = -Σ pk log pk
  d  = Mahalanobis distance  = √((f - μr)ᵀ Σr⁻¹ (f - μr))

The Mahalanobis distance measures how far an image's statistics
deviate from the real-image distribution envelope, incorporating
correlations among features — more sensitive to subtle anomalies
than a simple Euclidean metric.

Reference:
  Mathematical Formulation.pdf — Statistical Envelope Modeling
"""

import numpy as np
from pathlib import Path


class StatisticalFeatureExtractor:
    """
    Extracts the 4-dimensional statistical feature vector defined
    in the Mathematical Formulation paper.

    Usage:
        extractor = StatisticalFeatureExtractor()

        # During training — fit envelope on real images
        real_features = [extractor.extract_raw(img) for img in real_images]
        extractor.fit_envelope(real_features)

        # During inference — full 4-dim vector including Mahalanobis
        f = extractor.extract(img)   # shape (4,)
    """

    def __init__(self):
        # Real-image envelope parameters (μr, Σr)
        # Estimated from training real images, used for Mahalanobis distance
        self._envelope_mean = None   # μr ∈ ℝ³  (mean of [μ, σ², H] over real images)
        self._envelope_cov  = None   # Σr ∈ ℝ³ˣ³
        self._envelope_cov_inv = None
        self._is_fitted     = False

    # ── Raw 3-dim features (no Mahalanobis) ──────────────────────────

    def extract_raw(self, img: np.ndarray) -> np.ndarray:
        """
        Extract [μ, σ², H] from a single image.
        Does NOT require envelope to be fitted.

        Args:
            img: uint8 image array, shape (H, W) or (H, W, C)

        Returns:
            f_stat_raw: shape (3,)  = [mean, variance, entropy]
        """
        # Convert to grayscale intensity I(i,j)
        I = self._to_grayscale(img).astype(np.float64)

        mean     = self._compute_mean(I)
        variance = self._compute_variance(I, mean)
        entropy  = self._compute_entropy(I)

        return np.array([mean, variance, entropy], dtype=np.float32)

    # ── Full 4-dim vector (with Mahalanobis) ─────────────────────────

    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Extract full [μ, σ², H, d] feature vector.
        Requires envelope to be fitted first via fit_envelope().

        Args:
            img: uint8 image array, shape (H, W) or (H, W, C)

        Returns:
            f_stat: shape (4,)  = [mean, variance, entropy, mahalanobis_dist]
        """
        f_raw = self.extract_raw(img)

        if self._is_fitted:
            d = self._mahalanobis_distance(f_raw)
        else:
            # Before fitting — use 0.0 as placeholder
            d = 0.0

        return np.append(f_raw, d).astype(np.float32)

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract features from a batch of images.

        Args:
            images: list of uint8 numpy arrays

        Returns:
            features: shape (N, 4)
        """
        return np.array([self.extract(img) for img in images],
                        dtype=np.float32)

    # ── Envelope fitting ─────────────────────────────────────────────

    def fit_envelope(self, real_features: np.ndarray):
        """
        Estimate the real-image envelope (μr, Σr) from real image features.

        From the paper:
            μr = E[f_stat]
            Σr = Cov[f_stat]

        Under a Gaussian assumption, this defines the ellipsoidal
        confidence region that encloses real-image statistics.

        Args:
            real_features: shape (N, 3) — raw features from real images
                           Each row is [μ, σ², H] for one image
        """
        real_features = np.array(real_features, dtype=np.float64)
        if real_features.ndim == 1:
            real_features = real_features.reshape(1, -1)

        # μr — mean of real-image statistics
        self._envelope_mean = real_features.mean(axis=0)

        # Σr — covariance of real-image statistics
        # ddof=1 for unbiased estimate
        if len(real_features) > 1:
            cov = np.cov(real_features.T)
        else:
            cov = np.eye(real_features.shape[1])

        # Regularize: add small diagonal to prevent singular matrix
        reg = 1e-6 * np.eye(cov.shape[0])
        self._envelope_cov = cov + reg

        # Pre-compute inverse for efficient Mahalanobis distance
        try:
            self._envelope_cov_inv = np.linalg.inv(self._envelope_cov)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse if matrix is singular
            self._envelope_cov_inv = np.linalg.pinv(self._envelope_cov)

        self._is_fitted = True

    def recompute_mahalanobis(self, features_raw: np.ndarray) -> np.ndarray:
        """
        Given raw (N, 3) features, compute full (N, 4) features
        with Mahalanobis distance appended.
        Useful after fitting envelope on training set.

        Args:
            features_raw: shape (N, 3)

        Returns:
            features_full: shape (N, 4)
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Envelope not fitted. Call fit_envelope() first."
            )
        distances = np.array(
            [self._mahalanobis_distance(f) for f in features_raw],
            dtype=np.float32
        )
        return np.column_stack([features_raw, distances])

    # ── Core mathematical formulas ────────────────────────────────────

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale intensity I(i,j).
        Handles both grayscale and BGR/RGB inputs.
        """
        if img is None:
            raise ValueError("Image is None")

        img = np.asarray(img)

        if img.ndim == 2:
            return img.astype(np.float64)

        if img.ndim == 3:
            if img.shape[2] == 1:
                return img[:, :, 0].astype(np.float64)
            # BGR weighted luminance (matches OpenCV's BGR2GRAY formula)
            # 0.114*B + 0.587*G + 0.299*R
            return (0.114 * img[:, :, 0].astype(np.float64) +
                    0.587 * img[:, :, 1].astype(np.float64) +
                    0.299 * img[:, :, 2].astype(np.float64))

        raise ValueError(f"Unexpected image shape: {img.shape}")

    @staticmethod
    def _compute_mean(I: np.ndarray) -> float:
        """
        Mean intensity:
            μ = (1/HW) Σᵢⱼ I(i,j)
        """
        return float(I.mean())

    @staticmethod
    def _compute_variance(I: np.ndarray, mean: float) -> float:
        """
        Intensity variance:
            σ² = (1/HW) Σᵢⱼ (I(i,j) - μ)²
        """
        return float(((I - mean) ** 2).mean())

    @staticmethod
    def _compute_entropy(I: np.ndarray,
                         n_bins: int = 256) -> float:
        """
        Intensity entropy:
            H = -Σₖ pₖ log pₖ

        where pₖ = fraction of pixels at intensity level k.
        Uses natural logarithm (log base e).
        Zero-probability bins are excluded (0 * log(0) = 0 by convention).
        """
        # Compute histogram over [0, 255]
        I_int = np.clip(I, 0, 255).astype(np.int32)
        hist  = np.bincount(I_int.ravel(), minlength=n_bins).astype(np.float64)

        # Normalize to probabilities
        total = hist.sum()
        if total == 0:
            return 0.0
        pk = hist / total

        # Entropy — only sum over non-zero bins
        nonzero = pk > 0
        return float(-np.sum(pk[nonzero] * np.log(pk[nonzero])))

    def _mahalanobis_distance(self, f: np.ndarray) -> float:
        """
        Mahalanobis distance from real-image envelope:
            d(X) = √((f_stat - μr)ᵀ Σr⁻¹ (f_stat - μr))

        Incorporates correlations among features — more sensitive
        to subtle anomalies than Euclidean distance.
        """
        delta = f.astype(np.float64) - self._envelope_mean
        dist_sq = float(delta @ self._envelope_cov_inv @ delta)
        # Guard against numerical issues (should always be ≥ 0)
        return float(np.sqrt(max(dist_sq, 0.0)))

    # ── Persistence ───────────────────────────────────────────────────

    def get_envelope_params(self) -> dict:
        """Return envelope parameters for saving."""
        if not self._is_fitted:
            return {}
        return {
            "envelope_mean":    self._envelope_mean,
            "envelope_cov":     self._envelope_cov,
            "envelope_cov_inv": self._envelope_cov_inv,
        }

    def set_envelope_params(self, params: dict):
        """Restore envelope parameters from saved state."""
        if not params:
            return
        self._envelope_mean    = params["envelope_mean"]
        self._envelope_cov     = params["envelope_cov"]
        self._envelope_cov_inv = params["envelope_cov_inv"]
        self._is_fitted        = True

    # ── Diagnostics ───────────────────────────────────────────────────

    def describe_envelope(self) -> str:
        """Return a human-readable description of the fitted envelope."""
        if not self._is_fitted:
            return "Envelope not fitted yet."

        labels = ["mean", "variance", "entropy"]
        lines  = ["Real-Image Statistical Envelope:", "-" * 35]
        for i, label in enumerate(labels):
            lines.append(
                f"  {label:<12}  μr={self._envelope_mean[i]:.4f}"
            )
        lines.append(f"\n  Covariance matrix Σr:")
        for row in self._envelope_cov:
            lines.append("    " + "  ".join(f"{v:10.4f}" for v in row))
        return "\n".join(lines)

    @property
    def output_dim(self) -> int:
        return 4

    @property
    def feature_names(self) -> list:
        return ["stat_mean", "stat_variance", "stat_entropy",
                "stat_mahalanobis"]
