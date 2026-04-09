"""
preprocess.py — Dataset Preprocessing Pipeline
================================================
Handles all three datasets:
  - CelebDF V2          : real/ + fake/ folders
  - FaceForensics++     : real/ + fake/ with method subfolders
  - Stable Diffusion    : 512/ + 768/ + 1024/ subfolders (fake only)

What it does for each image:
  1. Load image
  2. Quality filter (brightness + blur check)
  3. Resize to 256×256
  4. Save with sequential filename

Usage on cluster:
  python src/preprocessing/preprocess.py --dataset celebdf
  python src/preprocessing/preprocess.py --dataset faceforensics --max 5000
  python src/preprocessing/preprocess.py --dataset stable_diffusion
  python src/preprocessing/preprocess.py --all
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Add project root to path ──────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import get_config, get_paths, get_dataset_paths
from src.utils import get_logger, ensure_dirs, get_image_paths, print_progress


# ═════════════════════════════════════════════════════════════════════
#  Quality Filter
# ═════════════════════════════════════════════════════════════════════

class QualityFilter:
    """
    Filters out bad frames before they enter the pipeline.
    Catches: black frames, overexposed frames, blurry frames, corrupt files.
    """

    def __init__(self, cfg: dict):
        qf = cfg["preprocessing"]["quality_filter"]
        self.min_brightness = qf["min_brightness"]
        self.max_brightness = qf["max_brightness"]
        self.min_blur_score = qf["min_blur_score"]

    def check(self, img: np.ndarray) -> tuple:
        """
        Returns (passed: bool, reason: str)
        """
        if img is None:
            return False, "corrupt"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
               if len(img.shape) == 3 else img

        # Brightness check
        mean_brightness = gray.mean()
        if mean_brightness < self.min_brightness:
            return False, "too_dark"
        if mean_brightness > self.max_brightness:
            return False, "too_bright"

        # Blur check — Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.min_blur_score:
            return False, "too_blurry"

        return True, "ok"


# ═════════════════════════════════════════════════════════════════════
#  Image Processor
# ═════════════════════════════════════════════════════════════════════

class ImageProcessor:
    """
    Loads, filters, resizes and saves a single image.
    Does NOT apply ImageNet normalization here — that happens
    inside the feature extraction pipeline at runtime.
    """

    def __init__(self, cfg: dict):
        self.target_size = cfg["preprocessing"]["target_size"]
        self.quality_filter = QualityFilter(cfg)

    def process(self, src_path: Path, dst_path: Path) -> tuple:
        """
        Process one image from src_path → dst_path.
        Returns (success: bool, reason: str)
        """
        # Load
        img = cv2.imread(str(src_path))
        if img is None:
            return False, "corrupt"

        # Quality filter
        passed, reason = self.quality_filter.check(img)
        if not passed:
            return False, reason

        # Resize — use INTER_AREA for downscaling (sharper result)
        h, w = img.shape[:2]
        interp = cv2.INTER_AREA \
                 if h > self.target_size or w > self.target_size \
                 else cv2.INTER_LINEAR

        img_resized = cv2.resize(
            img,
            (self.target_size, self.target_size),
            interpolation=interp
        )

        # Save
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(dst_path), img_resized)
        if not success:
            return False, "write_failed"

        return True, "ok"


# ═════════════════════════════════════════════════════════════════════
#  Dataset Preprocessors
# ═════════════════════════════════════════════════════════════════════

class DatasetPreprocessor:
    """Base class for dataset-specific preprocessors."""

    def __init__(self, cfg: dict, logger):
        self.cfg       = cfg
        self.logger    = logger
        self.processor = ImageProcessor(cfg)

    def _process_folder(self, src_folder: Path, dst_folder: Path,
                        prefix: str, max_images: int = None) -> dict:
        """
        Process all images from src_folder → dst_folder.
        Discovers recursively — handles any subfolder structure.

        Args:
            src_folder  : source directory (searched recursively)
            dst_folder  : destination directory
            prefix      : filename prefix e.g. "real" or "fake"
            max_images  : cap on number of images to process

        Returns:
            stats dict with counts
        """
        ensure_dirs(dst_folder)

        # Discover all images recursively
        all_paths = get_image_paths(src_folder)
        if not all_paths:
            self.logger.warning(f"No images found in {src_folder}")
            return {"found": 0, "saved": 0, "skipped": 0, "skip_reasons": {}}

        if max_images:
            all_paths = all_paths[:max_images]

        total        = len(all_paths)
        saved        = 0
        skip_reasons = {}

        self.logger.info(f"  Processing {total} images from {src_folder.name}/")
        self.logger.info(f"  Output → {dst_folder}")

        for idx, src_path in enumerate(all_paths, start=1):
            dst_filename = f"{prefix}_{idx:07d}.png"
            dst_path     = dst_folder / dst_filename

            success, reason = self.processor.process(src_path, dst_path)

            if success:
                saved += 1
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            # Progress every 500 images
            if idx % 500 == 0 or idx == total:
                skipped = idx - saved
                print(f"\r    [{idx:>6}/{total}]  saved={saved}  "
                      f"skipped={skipped}", end="", flush=True)

        print()  # newline after progress

        stats = {
            "found":        total,
            "saved":        saved,
            "skipped":      total - saved,
            "skip_reasons": skip_reasons
        }
        self._log_stats(stats, prefix)
        return stats

    def _log_stats(self, stats: dict, label: str):
        pct_saved = 100 * stats["saved"] / max(stats["found"], 1)
        self.logger.info(
            f"    {label}: {stats['saved']}/{stats['found']} saved "
            f"({pct_saved:.1f}%)  skipped={stats['skipped']} "
            f"{stats['skip_reasons']}"
        )


class CelebDFPreprocessor(DatasetPreprocessor):
    """
    CelebDF V2 structure:
        celebdf_v2/
            real/   ← 50,740 real face images
            fake/   ← 50,751 fake face images
    """

    def run(self, raw_dir: Path, out_dir: Path,
            max_per_class: int = None) -> dict:

        self.logger.info("=" * 55)
        self.logger.info("  CelebDF V2 Preprocessing")
        self.logger.info("=" * 55)

        results = {}

        for label in ["real", "fake"]:
            src = raw_dir / label
            dst = out_dir / label

            if not src.exists():
                self.logger.error(f"  Missing: {src}")
                continue

            self.logger.info(f"\n  [{label.upper()}]")
            results[label] = self._process_folder(
                src, dst, prefix=label, max_images=max_per_class
            )

        return results


class FaceForensicsPreprocessor(DatasetPreprocessor):
    """
    FaceForensics++ structure:
        faceforensics/
            real/       ← pristine YouTube frames
            fake/
                Deepfakes/
                Face2Face/
                FaceSwap/
                FaceShifter/
                NeuralTextures/

    All fake subfolders are merged into one fake/ output folder.
    """

    def run(self, raw_dir: Path, out_dir: Path,
            max_per_class: int = None) -> dict:

        self.logger.info("=" * 55)
        self.logger.info("  FaceForensics++ Preprocessing")
        self.logger.info("=" * 55)

        results = {}

        # Real images
        real_src = raw_dir / "real"
        real_dst = out_dir / "real"
        if real_src.exists():
            self.logger.info(f"\n  [REAL]")
            results["real"] = self._process_folder(
                real_src, real_dst, prefix="real",
                max_images=max_per_class
            )
        else:
            self.logger.error(f"  Missing: {real_src}")

        # Fake images — discover recursively across all method subfolders
        fake_src = raw_dir / "fake"
        fake_dst = out_dir / "fake"
        if fake_src.exists():
            self.logger.info(f"\n  [FAKE] (all manipulation methods merged)")
            # List which subfolders were found
            subfolders = [d.name for d in fake_src.iterdir() if d.is_dir()]
            if subfolders:
                self.logger.info(f"  Methods found: {', '.join(sorted(subfolders))}")

            results["fake"] = self._process_folder(
                fake_src, fake_dst, prefix="fake",
                max_images=max_per_class
            )
        else:
            self.logger.error(f"  Missing: {fake_src}")

        return results


class StableDiffusionPreprocessor(DatasetPreprocessor):
    """
    Stable Diffusion structure:
        stable_diffusion/
            512/    ← SD 1.5 generated faces  (512×512)
            768/    ← SD 2.1 generated faces  (768×768)
            1024/   ← SDXL generated faces    (1024×1024)

    All subfolders merged into one fake/ output.
    NO real/ folder — this is a test-only dataset.
    """

    def run(self, raw_dir: Path, out_dir: Path,
            max_per_class: int = None) -> dict:

        self.logger.info("=" * 55)
        self.logger.info("  Stable Diffusion Preprocessing  [FAKE ONLY]")
        self.logger.info("  (Test-only dataset — no real images)")
        self.logger.info("=" * 55)

        # Find resolution subfolders
        subfolders = sorted([
            d for d in raw_dir.iterdir()
            if d.is_dir() and d.name in ["512", "768", "1024"]
        ])

        if not subfolders:
            self.logger.error(
                f"  No 512/768/1024 subfolders found in {raw_dir}"
            )
            return {}

        self.logger.info(
            f"\n  Subfolders found: {[d.name for d in subfolders]}"
        )

        # Process all subfolders, output to single fake/ folder
        # We collect all paths first so sequential naming is global
        fake_dst = out_dir / "fake"
        ensure_dirs(fake_dst)

        all_paths = []
        for subfolder in subfolders:
            paths = get_image_paths(subfolder)
            self.logger.info(f"  {subfolder.name}/  → {len(paths)} images")
            all_paths.extend(paths)

        if max_per_class:
            all_paths = all_paths[:max_per_class]

        total        = len(all_paths)
        saved        = 0
        skip_reasons = {}

        self.logger.info(f"\n  Total: {total} images → {fake_dst}")

        for idx, src_path in enumerate(all_paths, start=1):
            dst_path = fake_dst / f"fake_{idx:07d}.png"

            success, reason = self.processor.process(src_path, dst_path)
            if success:
                saved += 1
            else:
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

            if idx % 500 == 0 or idx == total:
                print(f"\r    [{idx:>6}/{total}]  saved={saved}  "
                      f"skipped={idx-saved}", end="", flush=True)

        print()

        stats = {
            "found":        total,
            "saved":        saved,
            "skipped":      total - saved,
            "skip_reasons": skip_reasons
        }
        self._log_stats(stats, "fake")
        return {"fake": stats}


# ═════════════════════════════════════════════════════════════════════
#  Report Writer
# ═════════════════════════════════════════════════════════════════════

def write_report(results: dict, out_dir: Path, dataset: str):
    """Save a preprocessing summary report."""
    ensure_dirs(out_dir)
    report_path = out_dir / f"preprocessing_report_{dataset}.txt"

    lines = [
        "=" * 55,
        f"  Preprocessing Report — {dataset}",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 55,
    ]

    total_saved   = 0
    total_skipped = 0

    for label, stats in results.items():
        if not isinstance(stats, dict):
            continue
        lines.append(f"\n  [{label.upper()}]")
        lines.append(f"    Found   : {stats.get('found', 0)}")
        lines.append(f"    Saved   : {stats.get('saved', 0)}")
        lines.append(f"    Skipped : {stats.get('skipped', 0)}")
        if stats.get("skip_reasons"):
            for reason, count in stats["skip_reasons"].items():
                lines.append(f"      {reason:<15}: {count}")
        total_saved   += stats.get("saved", 0)
        total_skipped += stats.get("skipped", 0)

    lines.append(f"\n  TOTAL SAVED   : {total_saved}")
    lines.append(f"  TOTAL SKIPPED : {total_skipped}")
    lines.append("=" * 55)

    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print(report_text)
    print(f"\n  Report saved → {report_path}")


# ═════════════════════════════════════════════════════════════════════
#  Main Entry Point
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for deepfake detection"
    )
    parser.add_argument(
        "--dataset",
        choices=["celebdf", "faceforensics", "stable_diffusion"],
        help="Which dataset to preprocess"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Preprocess all three datasets"
    )
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max images per class (overrides config)"
    )
    parser.add_argument(
        "--mode",
        choices=["cluster", "local"],
        default=None,
        help="Path mode (default: auto-detect)"
    )
    return parser.parse_args()


def run_dataset(dataset: str, cfg: dict, paths: dict,
                max_override: int, logger):
    """Run preprocessing for one dataset."""

    # Determine max images
    max_images = max_override or cfg["data"]["max_images_per_class"].get(dataset)

    # Get paths
    raw_root = Path(paths["datasets"][dataset])
    out_root = Path(paths["processed"][dataset])

    if not raw_root.exists():
        logger.error(f"Dataset not found: {raw_root}")
        logger.error("Please upload the dataset to the cluster first.")
        return {}

    # Run appropriate preprocessor
    if dataset == "celebdf":
        preprocessor = CelebDFPreprocessor(cfg, logger)
        results      = preprocessor.run(raw_root, out_root, max_images)

    elif dataset == "faceforensics":
        preprocessor = FaceForensicsPreprocessor(cfg, logger)
        results      = preprocessor.run(raw_root, out_root, max_images)

    elif dataset == "stable_diffusion":
        preprocessor = StableDiffusionPreprocessor(cfg, logger)
        results      = preprocessor.run(raw_root, out_root, max_images)

    # Write report
    log_dir = Path(paths.get("logs", "./logs"))
    write_report(results, log_dir, dataset)

    return results


def main():
    args   = parse_args()
    cfg    = get_config()
    paths  = get_paths()  # always uses cluster paths (/data/mpstme-naman/...)

    # Setup logger
    log_dir = Path(paths.get("logs", "./logs"))
    ensure_dirs(log_dir)
    logger  = get_logger(
        "preprocess",
        log_file=str(log_dir / "preprocessing.log"),
        level=cfg["logging"]["level"]
    )

    logger.info("RealCentric Deepfake Detection — Preprocessing Pipeline")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    datasets = (
        ["celebdf", "faceforensics", "stable_diffusion"]
        if args.all
        else [args.dataset]
    )

    if not args.all and not args.dataset:
        print("ERROR: Specify --dataset <name> or --all")
        print("  python preprocess.py --dataset celebdf")
        print("  python preprocess.py --all")
        sys.exit(1)

    for dataset in datasets:
        logger.info(f"\n{'='*55}")
        logger.info(f"  Starting: {dataset}")
        logger.info(f"{'='*55}")
        run_dataset(dataset, cfg, paths, args.max, logger)

    logger.info(f"\nAll done: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
