#!/bin/bash
#PBS -N df_unsupervised
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1:mem=65536mb
#PBS -l walltime=08:00:00
#PBS -o /data/mpstme-naman/deepfake_detection/logs/03_unsupervised.out
#PBS -e /data/mpstme-naman/deepfake_detection/logs/03_unsupervised.err

set -e
PROJ=/data/mpstme-naman/deepfake_detection
cd $PROJ

echo "=========================================="
echo "  STEP 3 — ONE-CLASS ENSEMBLE TRAINING"
echo "  Node: $(hostname)   Started: $(date)"
echo "=========================================="

python3 - << 'PYEOF'
import sys, numpy as np
sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from config.config_loader          import load_config, BASE
from src.models.one_class_ensemble import OneClassEnsemble

cfg      = load_config()
FEAT_DIR = BASE / 'data' / 'features'
CKPT_DIR = BASE / 'checkpoints'

Z_train = np.load(FEAT_DIR / 'Z_train.npy');  y_train = np.load(FEAT_DIR / 'y_train.npy')
Z_val   = np.load(FEAT_DIR / 'Z_val.npy');    y_val   = np.load(FEAT_DIR / 'y_val.npy')

Z_real_train = Z_train[y_train == 0]
Z_real_val   = Z_val[y_val   == 0]

class PassThrough:
    def extract_batch(self, arr, **kw): return np.array(arr)
    def extract(self, arr, **kw):       return arr

ensemble = OneClassEnsemble(cfg=cfg, use_augmentation=False, use_isolation_forest=True, target_fpr=0.05)
ensemble.fit([Z_real_train[i] for i in range(len(Z_real_train))], PassThrough(),
             real_val_images=[Z_real_val[i] for i in range(len(Z_real_val))])
ensemble.save(str(CKPT_DIR / 'ensemble.pkl'))
print(f'Saved  threshold τ={ensemble.threshold:.4f}')
PYEOF

echo "Finished: $(date)"
