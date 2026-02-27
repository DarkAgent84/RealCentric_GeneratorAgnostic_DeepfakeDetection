#!/bin/bash
#PBS -N df_autoencoder
#PBS -q workq
#PBS -l select=1:ncpus=4:ngpus=1:mem=65536mb
#PBS -l walltime=06:00:00
#PBS -o /data/mpstme-naman/deepfake_detection/logs/04_autoencoder.out
#PBS -e /data/mpstme-naman/deepfake_detection/logs/04_autoencoder.err

set -e
PROJ=/data/mpstme-naman/deepfake_detection
cd $PROJ

echo "=========================================="
echo "  STEP 4 — AUTOENCODER TRAINING"
echo "  Node: $(hostname)   Started: $(date)"
echo "=========================================="

python3 - << 'PYEOF'
import sys, cv2
sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from config.config_loader   import load_config, BASE
from src.models.autoencoder import DeepfakeExplainer
from tqdm import tqdm

cfg  = load_config()
PROC = BASE / 'data' / 'processed'
CKPT = BASE / 'checkpoints'

paths = sorted((PROC / 'celebdf' / 'real').glob('*.png'))[:20000]
imgs  = [cv2.imread(str(p)) for p in tqdm(paths, desc='Loading')]
imgs  = [x for x in imgs if x is not None]
print(f'Real images: {len(imgs):,}')

explainer = DeepfakeExplainer(cfg=cfg)
explainer.train_autoencoder(imgs, checkpoint_dir=str(CKPT), run_name='autoencoder')
print('Done.')
PYEOF

echo "Finished: $(date)"
