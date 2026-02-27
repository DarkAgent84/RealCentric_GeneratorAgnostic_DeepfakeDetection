#!/bin/bash
# =============================================================================
#  Master Orchestration — submits all 5 jobs with PBS dependencies
#  Usage: cd /data/mpstme-naman/deepfake_detection && bash jobs/run_all.sh
# =============================================================================
set -e
PROJ=/data/mpstme-naman/deepfake_detection
cd $PROJ
mkdir -p logs checkpoints results

echo "=============================================="
echo "  DEEPFAKE DETECTION PIPELINE — SVKM HPC"
echo "  $(date)"
echo "  Project: $PROJ"
echo "=============================================="

JOB1=$(qsub jobs/01_preprocess.sh)
echo "[1/5] Preprocess      : $JOB1"

JOB2=$(qsub -W depend=afterok:$JOB1 jobs/02_train_supervised.sh)
echo "[2/5] Supervised MLP  : $JOB2  (waits for $JOB1)"

JOB3=$(qsub -W depend=afterok:$JOB1 jobs/03_train_unsupervised.sh)
echo "[3/5] One-Class Ensem : $JOB3  (waits for $JOB1)"

JOB4=$(qsub -W depend=afterok:$JOB1 jobs/04_train_autoencoder.sh)
echo "[4/5] Autoencoder     : $JOB4  (waits for $JOB1)"

JOB5=$(qsub -W depend=afterok:$JOB2:$JOB3:$JOB4 jobs/05_benchmark.sh)
echo "[5/5] Benchmark       : $JOB5  (waits for 2,3,4)"

echo ""
echo "  Monitor : qstat -u $USER"
echo "  Logs    : tail -f $PROJ/logs/05_benchmark.out"
echo "  Results : $PROJ/results/"
