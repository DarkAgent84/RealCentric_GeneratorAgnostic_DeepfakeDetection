"""
extractor.py — Unified Feature Fusion Pipeline
================================================
Combines all four feature components into one fused vector z.

From the Mathematical Formulation paper:
    z = [g_stat(X); f_freq(X); f_wavelet(X); f_CNN(X)] ∈ R^D

where:
    g_stat   = [f_stat; d]  ∈ R^4    statistical + Mahalanobis
    f_freq               ∈ R^2    spectral slope + energy ratio
    f_wavelet            ∈ R^200  wavelet sub-band features
    f_CNN                ∈ R^512  ResNet-18  (or R^1280 EfficientNet)
    ─────────────────────────────────────────
    D = 4 + 2 + 200 + 512  = 718  (ResNet-18)
    D = 4 + 2 + 200 + 1280 = 1486 (EfficientNet-B0)

Two operating modes:
    EXTRACT_ONLY  — extract and return raw features (no normalisation)
                    Used when building the one-class envelope
    FULL          — extract, normalise (z-score), return
                    Used during inference and MLP training

Envelope fitting (unsupervised mode):
    The statistical extractor's Mahalanobis distance requires fitting
    on real images first. This class manages that fit automatically
    when fit_on_real() is called.
"""

import numpy as np
from pathlib import Path

from src.features.statistical  import StatisticalFeatureExtractor
from src.features.frequency    import FrequencyFeatureExtractor
from src.features.wavelet      import WaveletFeatureExtractor
from src.features.cnn_backbone import create_cnn_extractor


class FeatureFusionPipeline:
    """
    Unified pipeline that extracts and fuses all feature components.

    Usage:
        pipeline = FeatureFusionPipeline(cfg, backbone='resnet18')

        # Step 1 — fit on real images (required before full extraction)
        pipeline.fit_on_real(real_images)

        # Step 2 — extract fused vector for any image
        z = pipeline.extract(img)          # shape (718,) or (1486,)
        Z = pipeline.extract_batch(imgs)   # shape (N, 718)
    """

    def __init__(self, cfg: dict = None,
                 backbone: str = "resnet18",
                 freeze_cnn: bool = False,
                 device: str = None):
        """
        Args:
            cfg        : config dict from config.yaml
            backbone   : 'resnet18' or 'efficientnet_b0'
            freeze_cnn : freeze CNN weights (True for unsupervised mode)
            device     : 'cuda', 'cpu', or None (auto-detect)
        """
        self._cfg     = cfg
        self._backbone_name = backbone.lower()

        # ── Initialise all four extractors ────────────────────────────
        self._stat = StatisticalFeatureExtractor()
        self._freq = FrequencyFeatureExtractor(cfg)
        self._wav  = WaveletFeatureExtractor(cfg)
        self._cnn  = create_cnn_extractor(
            backbone=backbone,
            cfg=cfg,
            freeze=freeze_cnn,
            device=device
        )

        # ── Normalisation parameters (fitted on training set) ─────────
        self._norm_mean   = None   # shape (D,)
        self._norm_std    = None   # shape (D,)
        self._is_norm_fitted = False

        # ── Envelope fitting state ────────────────────────────────────
        self._is_env_fitted  = False

        # Compute total output dim
        self._dim_stat  = self._stat.output_dim     # 4
        self._dim_freq  = self._freq.output_dim     # 2
        self._dim_wav   = self._wav.output_dim      # 200
        self._dim_cnn   = self._cnn.output_dim      # 512 or 1280
        self._total_dim = (self._dim_stat + self._dim_freq
                           + self._dim_wav  + self._dim_cnn)

    # ─────────────────────────────────────────────────────────────────
    #  Fitting
    # ─────────────────────────────────────────────────────────────────

    def fit_on_real(self, real_images: list,
                    fit_normalisation: bool = True):
        """
        Fit the statistical envelope and feature normalisation
        on a set of real training images.

        Must be called before extract() produces meaningful
        Mahalanobis distance values.

        Args:
            real_images       : list of uint8 numpy arrays (real images)
            fit_normalisation : also fit z-score normalisation params
        """
        print(f"  Fitting pipeline on {len(real_images)} real images...")

        # Step 1 — extract raw 3-dim statistical features (no Mahalanobis yet)
        print("    [1/3] Statistical envelope...", end=" ", flush=True)
        raw_stat = np.array([
            self._stat.extract_raw(img) for img in real_images
        ], dtype=np.float32)
        self._stat.fit_envelope(raw_stat)
        self._is_env_fitted = True
        print("done")

        # Step 2 — optionally fit z-score normalisation
        if fit_normalisation:
            print("    [2/3] Feature normalisation...", end=" ", flush=True)
            raw_features = self._extract_raw_batch(real_images)
            self._norm_mean = raw_features.mean(axis=0)
            self._norm_std  = raw_features.std(axis=0) + 1e-8
            self._is_norm_fitted = True
            print("done")
        else:
            print("    [2/3] Skipping normalisation")

        print("    [3/3] Pipeline ready")
        print(f"  Fused vector dim: {self._total_dim}")

    # ─────────────────────────────────────────────────────────────────
    #  Extraction
    # ─────────────────────────────────────────────────────────────────

    def extract(self, img: np.ndarray,
                normalise: bool = True) -> np.ndarray:
        """
        Extract full fused feature vector z for one image.

        Args:
            img       : uint8 numpy array (H, W, C) or (H, W)
            normalise : apply z-score normalisation if fitted

        Returns:
            z: shape (D,)  where D = 718 (ResNet) or 1486 (EfficientNet)
        """
        # Extract each component
        f_stat = self._stat.extract(img)     # (4,)
        f_freq = self._freq.extract(img)     # (2,)
        f_wav  = self._wav.extract(img)      # (200,)
        f_cnn  = self._cnn.extract(img)      # (512,) or (1280,)

        # Fuse — concatenate in paper order
        z = np.concatenate([f_stat, f_freq, f_wav, f_cnn]).astype(np.float32)

        # NaN/Inf protection
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score normalisation
        if normalise and self._is_norm_fitted:
            z = (z - self._norm_mean) / self._norm_std

        return z

    def extract_batch(self, images: list,
                      normalise: bool = True,
                      cnn_batch_size: int = 32,
                      progress: bool = True) -> np.ndarray:
        """
        Extract fused features from a list of images.
        Processes CNN features in batches for GPU efficiency.
        Handcrafted features are processed per-image.

        Args:
            images        : list of uint8 numpy arrays
            normalise     : apply z-score normalisation if fitted
            cnn_batch_size: GPU batch size for CNN forward pass
            progress      : print progress every 500 images

        Returns:
            Z: shape (N, D)
        """
        N = len(images)

        # ── Handcrafted features (per-image) ─────────────────────────
        stat_feats = np.array([self._stat.extract(img)  for img in images])
        freq_feats = np.array([self._freq.extract(img)  for img in images])
        wav_feats  = np.array([self._wav.extract(img)   for img in images])

        # ── CNN features (batched for GPU efficiency) ─────────────────
        cnn_feats  = self._cnn.extract_batch(images, batch_size=cnn_batch_size)

        # ── Fuse ──────────────────────────────────────────────────────
        Z = np.concatenate(
            [stat_feats, freq_feats, wav_feats, cnn_feats], axis=1
        ).astype(np.float32)

        # NaN/Inf protection
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        # Z-score normalisation
        if normalise and self._is_norm_fitted:
            Z = (Z - self._norm_mean) / self._norm_std

        return Z

    def _extract_raw_batch(self, images: list) -> np.ndarray:
        """Extract un-normalised features for fitting normalisation."""
        return self.extract_batch(images, normalise=False)

    # ─────────────────────────────────────────────────────────────────
    #  Individual component access
    # ─────────────────────────────────────────────────────────────────

    def extract_components(self, img: np.ndarray) -> dict:
        """
        Extract each component separately — useful for ablation studies
        and debugging which feature group drives the decision.

        Returns:
            dict with keys: 'statistical', 'frequency', 'wavelet', 'cnn'
        """
        return {
            "statistical": self._stat.extract(img),
            "frequency":   self._freq.extract(img),
            "wavelet":     self._wav.extract(img),
            "cnn":         self._cnn.extract(img),
        }

    # ─────────────────────────────────────────────────────────────────
    #  Persistence
    # ─────────────────────────────────────────────────────────────────

    def get_state(self) -> dict:
        """
        Return full pipeline state for saving to checkpoint.
        Includes envelope params and normalisation stats.
        """
        return {
            "backbone_name":    self._backbone_name,
            "total_dim":        self._total_dim,
            "is_env_fitted":    self._is_env_fitted,
            "is_norm_fitted":   self._is_norm_fitted,
            "envelope_params":  self._stat.get_envelope_params(),
            "norm_mean":        self._norm_mean,
            "norm_std":         self._norm_std,
        }

    def set_state(self, state: dict):
        """Restore pipeline state from checkpoint."""
        self._is_env_fitted  = state.get("is_env_fitted", False)
        self._is_norm_fitted = state.get("is_norm_fitted", False)

        if state.get("envelope_params"):
            self._stat.set_envelope_params(state["envelope_params"])

        self._norm_mean = state.get("norm_mean")
        self._norm_std  = state.get("norm_std")

    # ─────────────────────────────────────────────────────────────────
    #  Properties & diagnostics
    # ─────────────────────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        return self._total_dim

    @property
    def is_fitted(self) -> bool:
        return self._is_env_fitted

    @property
    def feature_names(self) -> list:
        return (self._stat.feature_names
                + self._freq.feature_names
                + self._wav.feature_names
                + self._cnn.feature_names)

    @property
    def component_dims(self) -> dict:
        return {
            "statistical": self._dim_stat,
            "frequency":   self._dim_freq,
            "wavelet":     self._dim_wav,
            "cnn":         self._dim_cnn,
            "total":       self._total_dim,
        }

    def describe(self) -> str:
        lines = [
            "Feature Fusion Pipeline",
            "=" * 50,
            f"  Backbone      : {self._backbone_name}",
            f"  Envelope fit  : {self._is_env_fitted}",
            f"  Norm fit      : {self._is_norm_fitted}",
            "",
            "  Component dimensions:",
            f"    Statistical   :   {self._dim_stat:>4}-dim"
            f"  [μ, σ², H, Mahalanobis]",
            f"    Frequency     :   {self._dim_freq:>4}-dim"
            f"  [spectral slope s, energy ratio R]",
            f"    Wavelet       :   {self._dim_wav:>4}-dim"
            f"  [energies + moments + ratios + histograms]",
            f"    CNN           :   {self._dim_cnn:>4}-dim"
            f"  [{self._backbone_name} penultimate layer]",
            "    " + "-" * 38,
            f"    Fused z       :   {self._total_dim:>4}-dim",
        ]
        return "\n".join(lines)
