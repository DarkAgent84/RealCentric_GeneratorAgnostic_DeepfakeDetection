#!/bin/bash
#PBS -N df_benchmark
#PBS -q workq
#PBS -l select=1:ncpus=8:ngpus=1:mem=65536mb
#PBS -l walltime=06:00:00
#PBS -o /data/mpstme-naman/deepfake_detection/logs/05_benchmark.out
#PBS -e /data/mpstme-naman/deepfake_detection/logs/05_benchmark.err

set -e
PROJ=/data/mpstme-naman/deepfake_detection
cd $PROJ
mkdir -p results

echo "=========================================="
echo "  STEP 5 — FULL BENCHMARK"
echo "  Node: $(hostname)   Started: $(date)"
echo "=========================================="

python3 - << 'PYEOF'
import sys, numpy as np, pickle
sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from config.config_loader          import load_config, BASE
from src.features.extractor        import FeatureFusionPipeline
from src.models.mlp_classifier     import MLPTrainer
from src.models.one_class_ensemble import OneClassEnsemble
from src.models.autoencoder        import DeepfakeExplainer
from src.evaluation.benchmark      import Benchmarker

cfg      = load_config()
CKPT_DIR = BASE / 'checkpoints'
PROC     = BASE / 'data' / 'processed'
RES_DIR  = BASE / 'results'

pipeline = FeatureFusionPipeline(cfg=cfg, backbone='resnet18')
with open(CKPT_DIR / 'pipeline_state.pkl', 'rb') as f:
    pipeline.set_state(pickle.load(f))

mlp = MLPTrainer(cfg=cfg, input_dim=pipeline.output_dim)
mlp.load_checkpoint(str(CKPT_DIR / 'mlp_supervised_best.pt'))

ensemble = OneClassEnsemble(cfg=cfg)
ensemble.load(str(CKPT_DIR / 'ensemble.pkl'))

explainer = DeepfakeExplainer(cfg=cfg)
try:
    explainer.load_autoencoder(str(CKPT_DIR / 'autoencoder_best.pt'))
except Exception as e:
    print(f'Autoencoder not loaded ({e}) — ELA-only mode')

bm = Benchmarker(cfg=cfg, pipeline=pipeline, mlp_trainer=mlp,
                 ensemble=ensemble, explainer=explainer,
                 results_dir=str(RES_DIR))
bm.run_all(processed_root=str(PROC), max_per_class=None, run_robustness=True)
PYEOF

echo "Finished: $(date)"
