#!/bin/bash
#PBS -N df_preprocess
#PBS -q workq
#PBS -l select=1:ncpus=8:mem=65536mb
#PBS -l walltime=04:00:00
#PBS -o /data/mpstme-naman/deepfake_detection/logs/01_preprocess.out
#PBS -e /data/mpstme-naman/deepfake_detection/logs/01_preprocess.err

set -e
PROJ=/data/mpstme-naman/deepfake_detection
cd $PROJ
mkdir -p logs

echo "=========================================="
echo "  STEP 1 — PREPROCESSING"
echo "  Node   : $(hostname)"
echo "  Started: $(date)"
echo "=========================================="

python3 src/preprocessing/preprocess.py --dataset celebdf          --mode cluster
python3 src/preprocessing/preprocess.py --dataset faceforensics    --mode cluster
python3 src/preprocessing/preprocess.py --dataset stable_diffusion --mode cluster

echo "Finished: $(date)"
