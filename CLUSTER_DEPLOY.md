# Cluster Deployment Guide
**RealCentric Generator-Agnostic Deepfake Detection**  
SVKM AI/ML HPC — Altair Access

---

## Project Paths

| Directory | Path |
|---|---|
| Project root | `/data/mpstme-naman/deepfake_detection` |
| Raw datasets | `/data/mpstme-naman/deepfake_detection/data/celebdf_v2` / `faceforensics` / `stable_diffusion` |
| Preprocessed | `/data/mpstme-naman/deepfake_detection/data/processed/celebdf` / `ff` / `sd` |
| Features | `/data/mpstme-naman/deepfake_detection/data/features/` |
| Checkpoints | `/data/mpstme-naman/deepfake_detection/checkpoints/` |
| Results | `/data/mpstme-naman/deepfake_detection/results/` |
| Logs | `/data/mpstme-naman/deepfake_detection/logs/` |

---

## First-Time Setup

```bash
# 1. Upload project to cluster
# (use Altair Access → Files → +Upload)

# 2. Install dependencies
pip install -r /data/mpstme-naman/deepfake_detection/requirements.txt

# 3. Verify config
python3 -c "
import sys; sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from config.config_loader import print_config_summary
print_config_summary()
"
```

---

## Run Order — Notebooks (Interactive)

Open each in Jupyter via Altair Access in this order:

| # | Notebook | Time |
|---|---|---|
| 1 | `01_preprocessing.ipynb` | 30–60 min |
| 2 | `02_feature_extraction.ipynb` | 45–90 min |
| 3 | `03_train_supervised.ipynb` | 10–20 min |
| 4 | `04_train_unsupervised.ipynb` | 5–15 min |
| 5 | `05_train_autoencoder.ipynb` | 30–60 min |
| 6 | `06_benchmark.ipynb` | 30–60 min |
| 7 | `07_inference.ipynb` | On demand |

---

## Run Order — PBS Jobs (Batch / Background)

```bash
cd /data/mpstme-naman/deepfake_detection
bash jobs/run_all.sh
```

This submits all 5 jobs with automatic dependency chaining:

```
01_preprocess ──────────────────────────┐
                                        ▼
             ┌── 02_supervised ──────┐
             ├── 03_unsupervised ────┼──▶ 05_benchmark
             └── 04_autoencoder ─────┘
```

---

## Monitoring

```bash
qstat -u mpstme-naman                          # all your jobs
tail -f /data/mpstme-naman/deepfake_detection/logs/05_benchmark.out          # benchmark log
ls -lh /data/mpstme-naman/deepfake_detection/checkpoints/                    # saved models
ls -lh /data/mpstme-naman/deepfake_detection/results/                        # outputs
```

---

## Expected Output Files

```
checkpoints/
├── pipeline_state.pkl          ← fitted feature pipeline
├── mlp_supervised_best.pt      ← MLP weights
├── ensemble.pkl                ← One-Class Ensemble
└── autoencoder_best.pt         ← Autoencoder weights

data/features/
├── Z_train.npy / y_train.npy  ← training features (CelebDF 70%)
├── Z_val.npy   / y_val.npy    ← validation
├── Z_test.npy  / y_test.npy   ← test
├── Z_ff.npy    / y_ff.npy     ← FaceForensics++ (cross-dataset)
├── Z_sd.npy    / y_sd.npy     ← Stable Diffusion (cross-dataset)
└── Z_celebdf.npy / y_celebdf.npy

results/
├── benchmark_summary.csv
├── robustness.csv
├── ablation.csv
├── robustness_plot.png
└── inference/
    └── *_result.png
```

---

## Quick Inference After Training

```python
import sys, cv2, numpy as np, pickle
sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from pathlib import Path
from config.config_loader          import load_config, BASE
from src.features.extractor        import FeatureFusionPipeline
from src.models.mlp_classifier     import MLPTrainer
from src.models.one_class_ensemble import OneClassEnsemble

cfg      = load_config()
CKPT_DIR = BASE / 'checkpoints'

pipeline = FeatureFusionPipeline(cfg=cfg, backbone='resnet18')
with open(CKPT_DIR / 'pipeline_state.pkl', 'rb') as f:
    pipeline.set_state(pickle.load(f))

mlp = MLPTrainer(cfg=cfg, input_dim=pipeline.output_dim)
mlp.load_checkpoint(str(CKPT_DIR / 'mlp_supervised_best.pt'))

img   = cv2.imread('/path/to/your/image.jpg')
Z     = pipeline.extract(img, normalise=True).reshape(1, -1)
prob  = float(mlp.predict_proba(Z)[0])
label = 'FAKE' if prob >= 0.5 else 'REAL'
print(f'{label}  (confidence={prob:.4f})')
```
