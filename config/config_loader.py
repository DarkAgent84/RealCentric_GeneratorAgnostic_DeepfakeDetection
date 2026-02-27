"""
config_loader.py — Central Configuration Loader
=================================================
Single source of truth for all paths and hyperparameters.
Every module imports from here — never hardcode paths elsewhere.

Usage:
    from config.config_loader import load_config, get_paths, BASE

    cfg      = load_config()
    paths    = get_paths()
    FEAT_DIR = BASE / 'data' / 'features'
"""

import yaml
from pathlib import Path

# ── Single canonical base path ────────────────────────────────────────
BASE = Path("/data/mpstme-naman/deepfake_detection")

_CONFIG_DIR = Path(__file__).parent


def _load_yaml(filename: str) -> dict:
    with open(_CONFIG_DIR / filename) as f:
        return yaml.safe_load(f)


def load_config() -> dict:
    """Return master config dict (config.yaml)."""
    return _load_yaml("config.yaml")


# Alias for backward compatibility
get_config = load_config


def get_cluster_paths() -> dict:
    """Return full cluster_paths.yaml as dict."""
    return _load_yaml("cluster_paths.yaml")


def get_paths() -> dict:
    """Return paths section of cluster_paths.yaml."""
    return get_cluster_paths()["paths"]


def get_dataset_paths(dataset: str) -> dict:
    """
    Return all relevant directory Paths for one dataset.

    Args:
        dataset: "celebdf" | "faceforensics" | "stable_diffusion"

    Returns dict with Path values:
        raw_real, raw_fake, processed_real, processed_fake,
        features, checkpoints, results
    """
    p = get_paths()
    return {
        "raw_real":       Path(p["datasets"][dataset])  / "real",
        "raw_fake":       Path(p["datasets"][dataset])  / "fake",
        "processed_real": Path(p["processed"][dataset]) / "real",
        "processed_fake": Path(p["processed"][dataset]) / "fake",
        "features":       Path(p["features"]),
        "checkpoints":    Path(p["checkpoints"]),
        "results":        Path(p["results"]["root"]) / dataset,
    }


def get_pbs_defaults() -> dict:
    return get_cluster_paths()["pbs"]


def print_config_summary():
    cfg = load_config()
    p   = get_paths()
    print("\n" + "=" * 62)
    print("  Configuration Summary  [CLUSTER mode]")
    print("=" * 62)
    print(f"  Project root    : {p['project_root']}")
    print(f"  Image size      : {cfg['data']['image_size']}×{cfg['data']['image_size']}")
    print(f"  Split           : train={cfg['data']['split']['train']} "
          f"val={cfg['data']['split']['val']} test={cfg['data']['split']['test']}")
    print(f"\n  Feature groups:")
    for feat, fcfg in cfg["features"].items():
        if isinstance(fcfg, dict) and "enabled" in fcfg:
            sym = "✓" if fcfg["enabled"] else "✗"
            dim = fcfg.get("output_dim", "—")
            print(f"    {sym}  {feat:<22} dim={dim}")
    print(f"\n  Supervised backbone : {cfg['supervised']['backbone']}")
    print(f"  Unsupervised PCA    : {cfg['unsupervised']['pca']['n_components']} components")
    print(f"\n  Processed data  : {p['processed']['celebdf']}")
    print(f"  Features        : {p['features']}")
    print(f"  Checkpoints     : {p['checkpoints']}")
    print(f"  Results         : {p['results']['root']}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    print_config_summary()
