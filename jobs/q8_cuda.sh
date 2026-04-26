#!/bin/bash

#BSUB -J q8_cuda
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q8_cuda_%J.out
#BSUB -e batch_output/q8_cuda_%J.err
#BSUB -u s224473@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# Submit from the project root:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results
#   bsub < jobs/q8_cuda.sh

PROJECT_DIR="$HOME/Documents/02613/miniproject"
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: Missing project dir: $PROJECT_DIR"
  exit 2
fi
cd "$PROJECT_DIR"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p batch_output results

N=10
TOTAL=4571

echo "Warming up CUDA JIT compilation with N=1"
python -u scripts/q8_cuda.py 1 > /dev/null

START=$SECONDS
python -u scripts/q8_cuda.py "$N" > "results/q8_cuda_N${N}.csv"
ELAPSED=$((SECONDS - START))

echo "Elapsed seconds: $ELAPSED (for N=$N, after warm-up)"
echo "Estimated seconds for all $TOTAL: $(( ELAPSED * TOTAL / N ))"
echo "CSV written: results/q8_cuda_N${N}.csv"
