#!/bin/bash
#PBS -N df_supervised
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1:mem=65536mb
#PBS -l walltime=12:00:00
#PBS -o /data/mpstme-naman/deepfake_detection/logs/02_supervised.out
#PBS -e /data/mpstme-naman/deepfake_detection/logs/02_supervised.err

set -e
PROJ=/data/mpstme-naman/deepfake_detection
cd $PROJ
mkdir -p logs checkpoints results

echo "=========================================="
echo "  STEP 2 — SUPERVISED MLP TRAINING"
echo "  Node: $(hostname)"
echo "  GPU : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo CPU)"
echo "  Started: $(date)"
echo "=========================================="

python3 - << 'PYEOF'
import sys, numpy as np
sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from config.config_loader      import load_config, BASE
from src.models.mlp_classifier import MLPTrainer

cfg      = load_config()
FEAT_DIR = BASE / 'data' / 'features'
CKPT_DIR = BASE / 'checkpoints'

Z_train = np.load(FEAT_DIR / 'Z_train.npy');  y_train = np.load(FEAT_DIR / 'y_train.npy')
Z_val   = np.load(FEAT_DIR / 'Z_val.npy');    y_val   = np.load(FEAT_DIR / 'y_val.npy')

print(f'Train: {Z_train.shape}   Val: {Z_val.shape}')
trainer  = MLPTrainer(cfg=cfg, input_dim=Z_train.shape[1])
best_auc = trainer.train(Z_train, y_train, Z_val, y_val,
                         checkpoint_dir=str(CKPT_DIR), run_name='mlp_supervised')
print(f'Best val AUC: {best_auc:.4f}')
PYEOF

echo "Finished: $(date)"
