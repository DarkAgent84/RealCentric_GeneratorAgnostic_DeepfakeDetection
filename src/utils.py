"""
utils.py — Shared Utilities
=============================
Helper functions used across all modules.
Import from here to avoid duplication.
"""

import os
import random
import logging
import numpy as np
from pathlib import Path
from datetime import datetime


# ── Reproducibility ───────────────────────────────────────────────────
def set_seed(seed: int = 42):
    """
    Set all random seeds for full reproducibility.
    Call this at the start of every training script.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass


# ── Logging ───────────────────────────────────────────────────────────
def get_logger(name: str, log_file: str = None,
               level: str = "INFO") -> logging.Logger:
    """
    Create a logger that writes to both console and file.

    Args:
        name     : logger name (usually __name__)
        log_file : path to save log file (None = console only)
        level    : logging level string

    Returns:
        Configured logger instance
    """
    logger    = logging.getLogger(name)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)

    if logger.handlers:
        return logger  # already configured

    fmt     = logging.Formatter(
        "[%(asctime)s] %(levelname)s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ── Paths ─────────────────────────────────────────────────────────────
def ensure_dirs(*dirs):
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    """Return current timestamp string for naming outputs."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_image_paths(folder: Path, extensions=None) -> list:
    """
    Recursively collect all image paths under folder.
    Handles FF++ subdirectory structure automatically.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png"}
    paths = []
    for ext in extensions:
        paths.extend(folder.rglob(f"*{ext}"))
        paths.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


# ── Metrics helpers ───────────────────────────────────────────────────
def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Division that returns default instead of ZeroDivisionError."""
    return a / b if b != 0 else default


# ── Progress display ──────────────────────────────────────────────────
def progress_bar(current: int, total: int, prefix: str = "",
                 width: int = 30) -> str:
    """
    Simple ASCII progress bar for environments without tqdm.
    Falls back gracefully if tqdm not installed.
    """
    filled = int(width * current / max(total, 1))
    bar    = "█" * filled + "░" * (width - filled)
    pct    = 100 * current / max(total, 1)
    return f"{prefix} [{bar}] {current}/{total}  {pct:.1f}%"


def print_progress(current: int, total: int, prefix: str = "",
                   every: int = 100):
    """Print progress every `every` steps."""
    if current % every == 0 or current == total:
        print(f"\r  {progress_bar(current, total, prefix)}", end="", flush=True)
        if current == total:
            print()  # newline at end


# ── Device detection ──────────────────────────────────────────────────
def get_device():
    """
    Auto-detect best available device.
    Returns torch.device if torch is available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            mem_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  Device: GPU — {gpu_name}  ({mem_gb:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("  Device: CPU")
        return device
    except ImportError:
        print("  Device: CPU (torch not available)")
        return "cpu"


# ── Checkpoint helpers ────────────────────────────────────────────────
def save_checkpoint(obj, path: str, logger=None):
    """Save any object (model, dict, etc.) using torch or pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.save(obj, path)
    except Exception:
        import pickle
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    if logger:
        logger.info(f"Checkpoint saved → {path}")
    else:
        print(f"  Checkpoint saved → {path}")


def load_checkpoint(path: str):
    """Load checkpoint saved by save_checkpoint."""
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except Exception:
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Feature vector helpers ────────────────────────────────────────────
def normalize_features(features: np.ndarray,
                        mean: np.ndarray = None,
                        std: np.ndarray  = None):
    """
    Z-score normalize feature vectors.
    If mean/std not provided, compute from features.
    Returns (normalized_features, mean, std)
    """
    features = np.nan_to_num(features.astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
    if mean is None:
        mean = features.mean(axis=0)
    if std is None:
        std  = features.std(axis=0) + 1e-8
    return (features - mean) / std, mean, std
