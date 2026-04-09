import cv2
import numpy as np
from scipy.stats import skew, kurtosis

class SDFeatureExtractor:
    """
    Extracts Stable Diffusion-specific features:
    1. PRNU (Photo Response Non-Uniformity) proxy via noise residual statistics.
    2. Spectral Peak Detection for AI artifacts in high frequency domain.
    """
    def __init__(self, cfg=None):
        sd_cfg = (cfg or {}).get("features", {}).get("sd", {})
        self._enabled = sd_cfg.get("enabled", True)
        self._prnu_dim = 3
        self._spectral_dim = 4
        
    @property
    def output_dim(self):
        return self._prnu_dim + self._spectral_dim if self._enabled else 0

    @property
    def feature_names(self):
        if not self._enabled:
            return []
        names = ["prnu_var", "prnu_skew", "prnu_kurt"]
        names += [f"spectral_band_{i}" for i in range(self._spectral_dim)]
        return names

    def extract(self, img):
        if not self._enabled:
            return np.array([], dtype=np.float32)
            
        # Ensure grayscale for frequency/noise analysis
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = img.astype(np.float32)
            
        prnu_feats = self._extract_prnu(gray)
        spec_feats = self._extract_spectral(gray)
        
        return np.concatenate([prnu_feats, spec_feats]).astype(np.float32)

    def _extract_prnu(self, gray):
        # Very simple PRNU proxy: subtract a median filtered version to get noise residual
        denoised = cv2.medianBlur(gray.astype(np.uint8), 5).astype(np.float32)
        residual = gray - denoised
        
        # Compute statistics of the residual
        flat_residual = residual.flatten()
        var = np.var(flat_residual)
        skewness = skew(flat_residual)
        kurt = kurtosis(flat_residual)
        
        return np.array([var, skewness, kurt], dtype=np.float32)

    def _extract_spectral(self, gray):
        # 2D FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        # Divide into concentric rings to find high frequency anomalies
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        max_r = min(cx, cy)
        bands = []
        # Create 4 bands in the high frequency outer rings (0.5 max_r to max_r)
        num_bands = self._spectral_dim
        step = (0.5 * max_r) / num_bands
        
        start_r = 0.5 * max_r
        for i in range(num_bands):
            mask = (r >= start_r + i * step) & (r < start_r + (i + 1) * step)
            if np.any(mask):
                bands.append(np.mean(magnitude_spectrum[mask]))
            else:
                bands.append(0.0)
                
        return np.array(bands, dtype=np.float32)
