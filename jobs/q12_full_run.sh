#!/bin/bash

#BSUB -J q12_full_run
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 04:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q12_full_run_%J.out
#BSUB -e batch_output/q12_full_run_%J.err
#BSUB -u s224473@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# Submit from the project root:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results reports/figures
#   bsub < jobs/q12_full_run.sh

PROJECT_DIR="$HOME/Documents/02613/miniproject"
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: Missing project dir: $PROJECT_DIR"
  exit 2
fi
cd "$PROJECT_DIR"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p batch_output results reports/figures

N=4571
WORKERS=8
CSV="results/q12_all_N${N}_W${WORKERS}.csv"

echo "Warming up Numba JIT with N=1"
python -u scripts/q11_numba_parallel.py 1 --workers 1 > /dev/null

START=$SECONDS
python -u scripts/q11_numba_parallel.py "$N" --workers "$WORKERS" > "$CSV"
ELAPSED=$((SECONDS - START))

echo "Elapsed seconds: $ELAPSED (N=$N, W=$WORKERS)"
echo "CSV written: $CSV"

echo ""
echo "Running Q12 analysis..."
python -u scripts/q12_analysis.py "$CSV" \
  --figures-dir reports/figures \
  --summary-path reports/q12_analysis_summary.txt
