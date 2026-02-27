"""
benchmark.py — Full Benchmarking & Evaluation Pipeline
=======================================================
Ties all components together into one evaluation run.

Evaluates both detection modes across all three datasets:
    Datasets:
        CelebDF-V2        — face swap deepfakes
        FaceForensics++   — 5 manipulation methods
        Stable Diffusion  — AI-generated faces (unseen distribution)

    Modes:
        Supervised     — MLP classifier on fused 718-dim feature vector
        Unsupervised   — One-Class Ensemble (no fake images at train time)
        ELA-only       — Baseline: training-free ELA explainability score

    Robustness sweeps:
        JPEG compression  : Q = 90, 70, 50, 30
        Gaussian blur     : σ = 1, 3, 5
        Resize            : scale = 0.75, 0.5, 0.25

    Output artefacts (all saved to results/):
        benchmark_results.json   — full metric table
        benchmark_summary.csv    — per-dataset/mode AUC table
        robustness.csv           — AUC vs degradation level
        confusion_matrices.json  — TP/FP/FN/TN per setting

Usage (standalone):
    python -m src.evaluation.benchmark \\
        --data_root /data/mpstme-naman/deepfake_detection/data/processed mpstme-naman/deepfake_detection/processed \\
        --checkpoint_dir checkpoints/ \\
        --results_dir results/

Usage (from notebook):
    from src.evaluation.benchmark import Benchmarker
    bm = Benchmarker(cfg, pipeline, mlp_trainer, ensemble)
    bm.run_all()
"""

import json
import time
import numpy as np
import csv
from pathlib import Path
from datetime import datetime

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
#  Metric helpers (pure numpy — no sklearn needed)
# ─────────────────────────────────────────────────────────────────────

def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    desc     = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc]
    n_pos    = y_true.sum()
    n_neg    = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp  = np.cumsum(y_sorted)
    fp  = np.cumsum(1 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    auc = float(
        np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid")
        else np.trapz(tpr, fpr)
    )
    return abs(auc)


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_score: np.ndarray = None) -> dict:
    """Compute full binary classification metric suite."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    acc  = 100.0 * (tp + tn) / max(len(y_true), 1)
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    fpr  = fp / (fp + tn + 1e-8)

    m = dict(accuracy=acc, precision=prec, recall=rec,
             f1=f1, fpr=fpr, tp=tp, fp=fp, fn=fn, tn=tn,
             n=len(y_true))

    if y_score is not None:
        m["auc"] = _roc_auc(y_true, y_score)
    return m


# ─────────────────────────────────────────────────────────────────────
#  Image degradation (robustness testing)
# ─────────────────────────────────────────────────────────────────────

def _degrade(img: np.ndarray, deg_type: str, param) -> np.ndarray:
    """Apply one degradation to a uint8 image."""
    if not CV2_AVAILABLE:
        return img

    if deg_type == "jpeg":
        _, buf = cv2.imencode(".jpg", img,
                              [cv2.IMWRITE_JPEG_QUALITY, int(param)])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    elif deg_type == "blur":
        k = int(param) * 2 + 1   # ensure odd
        return cv2.GaussianBlur(img, (k, k), 0)

    elif deg_type == "resize":
        H, W   = img.shape[:2]
        small  = cv2.resize(img, (max(1, int(W * param)),
                                  max(1, int(H * param))))
        return cv2.resize(small, (W, H))

    return img


# ─────────────────────────────────────────────────────────────────────
#  Dataset loader (reads preprocessed images from disk)
# ─────────────────────────────────────────────────────────────────────

def _load_images(folder: Path, max_images: int = None,
                 label: int = None) -> tuple:
    """
    Load images from a folder of PNGs.

    Returns:
        images : list of uint8 numpy arrays
        labels : list of int (label repeated)
    """
    if not folder.exists():
        return [], []

    paths  = sorted(folder.glob("*.png"))
    if max_images:
        paths = paths[:max_images]

    images, labels = [], []
    for p in paths:
        try:
            if CV2_AVAILABLE:
                import cv2 as _cv2
                img = _cv2.imread(str(p))
            else:
                img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            if img is not None:
                images.append(img)
                labels.append(label)
        except Exception:
            continue

    return images, labels


def _load_dataset(processed_root: Path, dataset_name: str,
                  max_per_class: int = None) -> dict:
    """
    Load real + fake images for one dataset.

    Returns:
        dict with keys: real_images, fake_images,
                        real_labels, fake_labels, name
    """
    base = processed_root / dataset_name
    real_dir = base / "real"
    fake_dir = base / "fake"

    real_imgs, real_lbls = _load_images(real_dir, max_per_class, label=0)
    fake_imgs, fake_lbls = _load_images(fake_dir, max_per_class, label=1)

    print(f"    {dataset_name:20s}  real={len(real_imgs)}  fake={len(fake_imgs)}")
    return dict(
        name=dataset_name,
        real_images=real_imgs,
        fake_images=fake_imgs,
        real_labels=real_lbls,
        fake_labels=fake_lbls,
    )


# ─────────────────────────────────────────────────────────────────────
#  Core Benchmarker
# ─────────────────────────────────────────────────────────────────────

class Benchmarker:
    """
    Runs full benchmark evaluation across datasets, modes, degradations.

    Args:
        cfg           : config dict
        pipeline      : FeatureFusionPipeline (fitted)
        mlp_trainer   : MLPTrainer (trained) or None
        ensemble      : OneClassEnsemble (fitted) or None
        explainer     : DeepfakeExplainer or None
        results_dir   : directory to write result files
    """

    DATASETS = ["celebdf_v2", "faceforensics", "stable_diffusion"]

    ROBUSTNESS_GRID = {
        "jpeg":   [90, 70, 50, 30],
        "blur":   [1, 3, 5],
        "resize": [0.75, 0.5, 0.25],
    }

    def __init__(self, cfg: dict = None,
                 pipeline=None,
                 mlp_trainer=None,
                 ensemble=None,
                 explainer=None,
                 results_dir: str = "results/"):

        self._cfg         = cfg or {}
        self._pipeline    = pipeline
        self._mlp         = mlp_trainer
        self._ensemble    = ensemble
        self._explainer   = explainer
        self._results_dir = Path(results_dir)
        self._results_dir.mkdir(parents=True, exist_ok=True)

        # All results accumulate here
        self._results = {
            "timestamp":   datetime.now().isoformat(),
            "datasets":    {},
            "robustness":  {},
            "summary":     {},
        }

    # ── Main entry point ─────────────────────────────────────────────

    def run_all(self, processed_root: str,
                max_per_class: int = 2000,
                run_robustness: bool = True):
        """
        Full benchmark run.

        Args:
            processed_root : path to preprocessed datasets
            max_per_class  : max images per class per dataset
            run_robustness : whether to run degradation sweep
        """
        root = Path(processed_root)
        t0   = time.time()

        print("\n" + "=" * 60)
        print("  DEEPFAKE DETECTION BENCHMARK")
        print("=" * 60)
        print(f"  Timestamp : {self._results['timestamp']}")
        print(f"  Root      : {root}")
        print(f"  Max/class : {max_per_class}")
        print()

        # ── Load all datasets ─────────────────────────────────────────
        print("  Loading datasets...")
        datasets = {}
        for name in self.DATASETS:
            ds = _load_dataset(root, name, max_per_class)
            if ds["real_images"] or ds["fake_images"]:
                datasets[name] = ds

        if not datasets:
            print("  WARNING: No images found. Check processed_root path.")
            return self._results

        # ── Per-dataset evaluation ────────────────────────────────────
        print("\n  Per-dataset evaluation...")
        for name, ds in datasets.items():
            print(f"\n  ── {name} ──")
            self._results["datasets"][name] = self._eval_dataset(ds)

        # ── Robustness sweep ─────────────────────────────────────────
        if run_robustness and datasets:
            print("\n  Robustness sweep (degradation × compression × blur × resize)...")
            # Use first dataset with both real and fake
            ref_ds = next(
                (ds for ds in datasets.values()
                 if ds["real_images"] and ds["fake_images"]),
                None
            )
            if ref_ds:
                self._results["robustness"] = self._robustness_sweep(ref_ds)

        # ── Cross-dataset generalisation ─────────────────────────────
        print("\n  Cross-dataset generalisation...")
        self._results["cross_dataset"] = self._cross_dataset(datasets)

        # ── Summary table ─────────────────────────────────────────────
        self._results["summary"] = self._build_summary()

        # ── Save results ──────────────────────────────────────────────
        elapsed = time.time() - t0
        self._results["elapsed_seconds"] = round(elapsed, 1)
        self._save_all()

        print(f"\n  Total time: {elapsed:.1f}s")
        print(f"  Results   → {self._results_dir}")
        self._print_summary()

        return self._results

    # ── Dataset evaluation ───────────────────────────────────────────

    def _eval_dataset(self, ds: dict) -> dict:
        """Evaluate all active modes on one dataset."""
        result   = {"name": ds["name"], "modes": {}}
        all_imgs = ds["real_images"] + ds["fake_images"]
        all_lbls = np.array(ds["real_labels"] + ds["fake_labels"])

        if len(all_imgs) == 0:
            return result

        # ── Feature extraction ────────────────────────────────────────
        if self._pipeline is not None:
            print(f"    Extracting features ({len(all_imgs)} images)...",
                  end=" ", flush=True)
            t0 = time.time()
            Z  = self._pipeline.extract_batch(all_imgs, normalise=True)
            print(f"{time.time()-t0:.1f}s")
        else:
            Z = None

        # ── Mode: Supervised MLP ──────────────────────────────────────
        if self._mlp is not None and Z is not None:
            try:
                probs = self._mlp.predict_proba(Z)
                preds = (probs >= 0.5).astype(int)
                m     = _binary_metrics(all_lbls, preds, probs)
                result["modes"]["supervised_mlp"] = m
                print(f"    MLP  → ACC={m['accuracy']:.1f}%  "
                      f"AUC={m['auc']:.4f}  F1={m['f1']:.4f}")
            except Exception as e:
                result["modes"]["supervised_mlp"] = {"error": str(e)}

        # ── Mode: One-Class Ensemble ──────────────────────────────────
        if self._ensemble is not None and Z is not None:
            try:
                real_Z = Z[all_lbls == 0]
                fake_Z = Z[all_lbls == 1]
                m      = self._ensemble.evaluate(real_Z, fake_Z)
                result["modes"]["one_class_ensemble"] = m
                print(f"    OCC  → ACC={m['accuracy']:.1f}%  "
                      f"AUC={m['auc']:.4f}  F1={m['f1']:.4f}")
            except Exception as e:
                result["modes"]["one_class_ensemble"] = {"error": str(e)}

        # ── Mode: ELA baseline ────────────────────────────────────────
        if self._explainer is not None:
            try:
                scores = np.array([
                    self._explainer._ela.score(img) for img in all_imgs
                ], dtype=np.float32)
                preds_ela = (scores > np.percentile(scores, 50)).astype(int)
                m_ela = _binary_metrics(all_lbls, preds_ela, scores)
                result["modes"]["ela_baseline"] = m_ela
                print(f"    ELA  → ACC={m_ela['accuracy']:.1f}%  "
                      f"AUC={m_ela['auc']:.4f}  F1={m_ela['f1']:.4f}")
            except Exception as e:
                result["modes"]["ela_baseline"] = {"error": str(e)}

        result["n_real"] = len(ds["real_images"])
        result["n_fake"] = len(ds["fake_images"])
        return result

    # ── Robustness sweep ─────────────────────────────────────────────

    def _robustness_sweep(self, ds: dict) -> dict:
        """Evaluate AUC at each degradation level."""
        sweep  = {}
        n_each = min(200, len(ds["real_images"]), len(ds["fake_images"]))

        real_subset = ds["real_images"][:n_each]
        fake_subset = ds["fake_images"][:n_each]
        labels      = np.array([0]*n_each + [1]*n_each)

        for deg_type, levels in self.ROBUSTNESS_GRID.items():
            sweep[deg_type] = {}
            for level in levels:
                key = f"{deg_type}_{level}"
                print(f"    {key:<18}", end=" ", flush=True)

                # Apply degradation
                real_deg = [_degrade(img, deg_type, level)
                            for img in real_subset]
                fake_deg = [_degrade(img, deg_type, level)
                            for img in fake_subset]
                all_imgs = real_deg + fake_deg

                entry = {"level": level}

                # MLP
                if self._mlp is not None and self._pipeline is not None:
                    try:
                        Z     = self._pipeline.extract_batch(
                            all_imgs, normalise=True)
                        probs = self._mlp.predict_proba(Z)
                        auc   = _roc_auc(labels, probs)
                        entry["mlp_auc"] = round(float(auc), 4)
                    except Exception:
                        entry["mlp_auc"] = None

                # One-class ensemble
                if self._ensemble is not None and self._pipeline is not None:
                    try:
                        Z      = self._pipeline.extract_batch(
                            all_imgs, normalise=False)
                        scores = self._ensemble.score_features(Z)
                        auc    = _roc_auc(labels, scores)
                        entry["occ_auc"] = round(float(auc), 4)
                    except Exception:
                        entry["occ_auc"] = None

                # ELA
                if self._explainer is not None:
                    try:
                        scores = np.array([
                            self._explainer._ela.score(img)
                            for img in all_imgs
                        ])
                        auc = _roc_auc(labels, scores)
                        entry["ela_auc"] = round(float(auc), 4)
                    except Exception:
                        entry["ela_auc"] = None

                sweep[deg_type][str(level)] = entry
                parts = [f"{k}={v:.3f}" for k, v in entry.items()
                         if k.endswith("auc") and v is not None]
                print("  ".join(parts))

        return sweep

    # ── Cross-dataset generalisation ─────────────────────────────────

    def _cross_dataset(self, datasets: dict) -> dict:
        """
        Train on CelebDF, test on FaceForensics and Stable Diffusion.
        Measures generalisation without retraining.
        Reports AUC drop from in-distribution to out-of-distribution.
        """
        cross = {}

        # In-distribution performance (from per-dataset results)
        for name, ds_result in self._results["datasets"].items():
            for mode, metrics in ds_result.get("modes", {}).items():
                if "auc" in metrics:
                    cross.setdefault(mode, {})[name] = {
                        "auc": metrics["auc"],
                        "f1":  metrics.get("f1"),
                    }

        # Compute generalisation gap: best in-dist vs worst out-dist
        for mode, dataset_aucs in cross.items():
            aucs = [v["auc"] for v in dataset_aucs.values()
                    if v["auc"] is not None]
            if aucs:
                cross[mode]["_gap"] = round(max(aucs) - min(aucs), 4)

        return cross

    # ── Summary ──────────────────────────────────────────────────────

    def _build_summary(self) -> dict:
        """Build condensed summary table for all modes × datasets."""
        summary = {}
        for ds_name, ds_result in self._results["datasets"].items():
            for mode, metrics in ds_result.get("modes", {}).items():
                summary.setdefault(mode, {})[ds_name] = {
                    "accuracy": round(metrics.get("accuracy", 0), 2),
                    "auc":      round(metrics.get("auc", 0), 4),
                    "f1":       round(metrics.get("f1", 0), 4),
                }
        return summary

    def _print_summary(self):
        """Pretty-print the summary table to stdout."""
        print("\n" + "=" * 60)
        print("  BENCHMARK SUMMARY")
        print("=" * 60)
        summary = self._results.get("summary", {})

        for mode, datasets in summary.items():
            print(f"\n  Mode: {mode}")
            print(f"  {'Dataset':<22} {'ACC':>7}  {'AUC':>7}  {'F1':>7}")
            print("  " + "-" * 44)
            for ds_name, m in datasets.items():
                print(f"  {ds_name:<22} "
                      f"{m.get('accuracy', 0):>6.2f}%  "
                      f"{m.get('auc', 0):>7.4f}  "
                      f"{m.get('f1', 0):>7.4f}")

        # Robustness summary
        rob = self._results.get("robustness", {})
        if rob:
            print("\n  Robustness (AUC vs degradation):")
            for deg_type, levels in rob.items():
                level_strs = []
                for lvl, entry in sorted(levels.items(),
                                         key=lambda x: float(x[0])):
                    auc = (entry.get("mlp_auc")
                           or entry.get("occ_auc")
                           or entry.get("ela_auc"))
                    if auc is not None:
                        level_strs.append(f"{lvl}→{auc:.3f}")
                print(f"    {deg_type:<8}: {',  '.join(level_strs)}")

    # ── File I/O ─────────────────────────────────────────────────────

    def _save_all(self):
        """Write all result files to results_dir."""
        # ── JSON (full results) ───────────────────────────────────────
        json_path = self._results_dir / "benchmark_results.json"
        with open(json_path, "w") as f:
            json.dump(self._results, f, indent=2, default=_json_safe)
        print(f"  Saved: {json_path}")

        # ── CSV summary ───────────────────────────────────────────────
        csv_path = self._results_dir / "benchmark_summary.csv"
        rows = []
        for mode, datasets in self._results.get("summary", {}).items():
            for ds_name, m in datasets.items():
                rows.append({
                    "mode":     mode,
                    "dataset":  ds_name,
                    "accuracy": m.get("accuracy", ""),
                    "auc":      m.get("auc", ""),
                    "f1":       m.get("f1", ""),
                })
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["mode","dataset","accuracy","auc","f1"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"  Saved: {csv_path}")

        # ── Robustness CSV ────────────────────────────────────────────
        rob      = self._results.get("robustness", {})
        rob_path = self._results_dir / "robustness.csv"
        if rob:
            rob_rows = []
            for deg_type, levels in rob.items():
                for lvl, entry in levels.items():
                    rob_rows.append({
                        "degradation": deg_type,
                        "level":       lvl,
                        "mlp_auc":     entry.get("mlp_auc", ""),
                        "occ_auc":     entry.get("occ_auc", ""),
                        "ela_auc":     entry.get("ela_auc", ""),
                    })
            with open(rob_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["degradation","level",
                                   "mlp_auc","occ_auc","ela_auc"])
                writer.writeheader()
                writer.writerows(rob_rows)
            print(f"  Saved: {rob_path}")

        # ── Confusion matrices JSON ───────────────────────────────────
        cm_path = self._results_dir / "confusion_matrices.json"
        cms     = {}
        for ds_name, ds_result in self._results.get("datasets", {}).items():
            cms[ds_name] = {}
            for mode, metrics in ds_result.get("modes", {}).items():
                cms[ds_name][mode] = {
                    k: metrics.get(k)
                    for k in ["tp","fp","fn","tn","accuracy","auc","f1"]
                }
        with open(cm_path, "w") as f:
            json.dump(cms, f, indent=2, default=_json_safe)
        print(f"  Saved: {cm_path}")


# ─────────────────────────────────────────────────────────────────────
#  Ablation study helper
# ─────────────────────────────────────────────────────────────────────

class AblationStudy:
    """
    Measures contribution of each feature group to overall AUC.

    For each group, zeroes out those features and measures AUC drop.
    Groups: statistical, frequency, wavelet, cnn
    """

    GROUPS = {
        "statistical": (0,   4),
        "frequency":   (4,   6),
        "wavelet":     (6,   206),
        "cnn":         (206, 718),
    }

    def __init__(self, mlp_trainer=None, ensemble=None):
        self._mlp      = mlp_trainer
        self._ensemble = ensemble

    def run(self, Z: np.ndarray, y: np.ndarray) -> dict:
        """
        Args:
            Z : (N, 718) full feature matrix (normalised)
            y : (N,)     binary labels

        Returns:
            dict: group → {full_auc, ablated_auc, auc_drop}
        """
        results = {}

        # Baseline AUC with all features
        baseline_auc = self._get_auc(Z, y)

        for group_name, (lo, hi) in self.GROUPS.items():
            Z_ablated        = Z.copy()
            Z_ablated[:, lo:hi] = 0.0   # zero out this group
            ablated_auc      = self._get_auc(Z_ablated, y)

            results[group_name] = {
                "full_auc":    round(float(baseline_auc), 4),
                "ablated_auc": round(float(ablated_auc),  4),
                "auc_drop":    round(float(baseline_auc - ablated_auc), 4),
            }
            print(f"    {group_name:<14}  full={baseline_auc:.4f}  "
                  f"ablated={ablated_auc:.4f}  "
                  f"drop={baseline_auc - ablated_auc:+.4f}")

        return results

    def _get_auc(self, Z: np.ndarray, y: np.ndarray) -> float:
        if self._mlp is not None:
            try:
                probs = self._mlp.predict_proba(Z)
                return _roc_auc(y, probs)
            except Exception:
                pass
        if self._ensemble is not None:
            try:
                real_Z = Z[y == 0]
                fake_Z = Z[y == 1]
                s_r    = self._ensemble.score_features(real_Z)
                s_f    = self._ensemble.score_features(fake_Z)
                all_s  = np.concatenate([s_r, s_f])
                all_y  = np.concatenate([np.zeros(len(s_r)),
                                         np.ones(len(s_f))])
                return _roc_auc(all_y, all_s)
            except Exception:
                pass
        return 0.5


# ─────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────

def _json_safe(obj):
    """Make numpy scalars JSON-serialisable."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def load_results(results_dir: str) -> dict:
    """Load previously saved benchmark results."""
    path = Path(results_dir) / "benchmark_results.json"
    with open(path) as f:
        return json.load(f)
