"""
frequency.py — Frequency Consistency Features
===============================================
Implements Component 2 from the Mathematical Formulation paper exactly.

Feature vector: f_freq = [s, R] ∈ ℝ²

where:
  s = radial spectral slope (log-log linear regression)
  R = spectral energy ratio (E_low / E_high)

Key insight from the paper:
  Natural images follow a 1/f^α power law — their power spectrum
  decays as a straight line in log-log space with slope s ≈ -2.
  GAN-generated and diffusion-generated fakes violate this regularity,
  showing deviations in slope and energy distribution.

Formulas (exact from paper):
  ─────────────────────────────────────────────────────────────
  Azimuthally averaged power spectrum:
    P(ρ) = (1/|{ω:|ω|=ρ}|) Σ_{|ω|=ρ} |F(ω)|²

  Radial slope s (log-log linear regression):
    s = Σ_k (log ρ_k - log ρ̄)(log P(ρ_k) - log P̄)
        ─────────────────────────────────────────────
        Σ_k (log ρ_k - log ρ̄)²

  Energy ratio R:
    E_low  = Σ_{|ω| ≤ ρ₀} |F(ω)|²
    E_high = Σ_{|ω| >  ρ₀} |F(ω)|²
    R      = E_low / E_high

Reference:
  Mathematical Formulation.pdf — Frequency Consistency Analysis
"""

import numpy as np


class FrequencyFeatureExtractor:
    """
    Extracts the 2-dimensional frequency feature vector defined
    in the Mathematical Formulation paper.

    Features:
        s — spectral slope: deviation from natural 1/f decay
        R — energy ratio:   low-frequency vs high-frequency energy

    Usage:
        extractor = FrequencyFeatureExtractor(cfg)
        f = extractor.extract(img)        # shape (2,)
        F = extractor.extract_batch(imgs) # shape (N, 2)
    """

    def __init__(self, cfg: dict = None):
        """
        Args:
            cfg: config dict from config.yaml
                 Uses cfg['features']['frequency'] section.
                 If None, uses default values.
        """
        if cfg is not None:
            freq_cfg = cfg["features"]["frequency"]
            self._cutoff_fraction = freq_cfg["cutoff_fraction"]
            self._radial_bins     = freq_cfg["radial_bins"]
        else:
            # Defaults matching config.yaml
            self._cutoff_fraction = 0.3   # ρ₀ = 0.3 × ρ_max
            self._radial_bins     = 64

    # ── Public interface ──────────────────────────────────────────────

    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Extract [s, R] from a single image.

        Args:
            img: uint8 image, shape (H, W) or (H, W, C)

        Returns:
            f_freq: shape (2,)  = [spectral_slope, energy_ratio]
        """
        I = self._to_grayscale(img)
        F = self._compute_fft(I)

        s = self._spectral_slope(F, I.shape)
        R = self._energy_ratio(F, I.shape)

        # Guard against NaN/Inf from degenerate images
        s = float(np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0))
        R = float(np.nan_to_num(R, nan=1.0, posinf=10.0, neginf=0.0))

        return np.array([s, R], dtype=np.float32)

    def extract_batch(self, images: list) -> np.ndarray:
        """
        Extract features from a list of images.

        Returns:
            features: shape (N, 2)
        """
        return np.array([self.extract(img) for img in images],
                        dtype=np.float32)

    # ── Core formulas ─────────────────────────────────────────────────

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        """Convert to float64 grayscale intensity."""
        img = np.asarray(img, dtype=np.float64)
        if img.ndim == 2:
            return img
        if img.ndim == 3:
            if img.shape[2] == 1:
                return img[:, :, 0]
            # BGR → luminance
            return (0.114 * img[:, :, 0] +
                    0.587 * img[:, :, 1] +
                    0.299 * img[:, :, 2])
        raise ValueError(f"Unexpected image shape: {img.shape}")

    @staticmethod
    def _compute_fft(I: np.ndarray) -> np.ndarray:
        """
        Compute 2D DFT and shift zero-frequency to centre.
        Returns complex spectrum F(u,v).
        """
        F = np.fft.fft2(I)
        F = np.fft.fftshift(F)   # shift DC to centre for azimuthal averaging
        return F

    def _azimuthal_average(self, F: np.ndarray,
                            shape: tuple) -> tuple:
        """
        Compute azimuthally averaged power spectrum P(ρ).

        From the paper:
            P(ρ) = (1/|{ω:|ω|=ρ}|) Σ_{|ω|=ρ} |F(ω)|²

        Returns:
            rho_vals: radial distances (bin centres)
            P_vals:   average power at each radial distance
        """
        H, W     = shape
        power    = np.abs(F) ** 2

        # Build radial coordinate grid (centred at DC)
        cy, cx   = H // 2, W // 2
        y_idx    = np.arange(H) - cy
        x_idx    = np.arange(W) - cx
        xx, yy   = np.meshgrid(x_idx, y_idx)
        rho_grid = np.sqrt(xx**2 + yy**2)

        rho_max  = min(cx, cy)   # max usable radius (stays within image)

        # Bin radial distances
        n_bins   = self._radial_bins
        bin_edges = np.linspace(0, rho_max, n_bins + 1)

        rho_vals = []
        P_vals   = []

        for i in range(n_bins):
            mask = (rho_grid >= bin_edges[i]) & (rho_grid < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            rho_vals.append((bin_edges[i] + bin_edges[i + 1]) / 2.0)
            P_vals.append(power[mask].mean())

        return np.array(rho_vals), np.array(P_vals)

    def _spectral_slope(self, F: np.ndarray, shape: tuple) -> float:
        """
        Radial spectral slope s via log-log linear regression.

        From the paper:
            s = Σ_k (log ρ_k - log ρ̄)(log P(ρ_k) - log P̄)
                ─────────────────────────────────────────────
                Σ_k (log ρ_k - log ρ̄)²

        Natural images have s ≈ -2 (1/f² power law).
        Deepfakes deviate from this.
        """
        rho_vals, P_vals = self._azimuthal_average(F, shape)

        # Keep only positive values for log
        valid = (rho_vals > 0) & (P_vals > 0)
        if valid.sum() < 2:
            return 0.0

        log_rho = np.log(rho_vals[valid])
        log_P   = np.log(P_vals[valid])

        # Linear regression in log-log space
        log_rho_bar = log_rho.mean()
        log_P_bar   = log_P.mean()

        numerator   = np.sum((log_rho - log_rho_bar) * (log_P - log_P_bar))
        denominator = np.sum((log_rho - log_rho_bar) ** 2)

        if abs(denominator) < 1e-12:
            return 0.0

        return float(numerator / denominator)

    def _energy_ratio(self, F: np.ndarray, shape: tuple) -> float:
        """
        Spectral energy ratio R = E_low / E_high.

        From the paper:
            E_low  = Σ_{|ω| ≤ ρ₀} |F(ω)|²
            E_high = Σ_{|ω| >  ρ₀} |F(ω)|²
            R      = E_low / E_high

        where ρ₀ = cutoff_fraction × ρ_max

        Real images tend to have predictable low/high energy ratios.
        Synthetic images often shift this balance (e.g. too much
        high-frequency noise from upsampling in GANs).
        """
        H, W   = shape
        power  = np.abs(F) ** 2

        cy, cx = H // 2, W // 2
        y_idx  = np.arange(H) - cy
        x_idx  = np.arange(W) - cx
        xx, yy = np.meshgrid(x_idx, y_idx)
        rho    = np.sqrt(xx**2 + yy**2)

        rho_max = min(cx, cy)
        rho_0   = self._cutoff_fraction * rho_max

        E_low  = power[rho <= rho_0].sum()
        E_high = power[rho >  rho_0].sum()

        if E_high < 1e-12:
            return 10.0   # degenerate: all energy in low freqs

        return float(E_low / E_high)

    # ── Diagnostics ───────────────────────────────────────────────────

    def get_power_spectrum(self, img: np.ndarray) -> tuple:
        """
        Return the full azimuthal power spectrum for plotting.

        Returns:
            rho_vals: radial distance array
            P_vals:   power at each distance
        """
        I = self._to_grayscale(img)
        F = self._compute_fft(I)
        return self._azimuthal_average(F, I.shape)

    @property
    def output_dim(self) -> int:
        return 2

    @property
    def feature_names(self) -> list:
        return ["freq_spectral_slope", "freq_energy_ratio"]
