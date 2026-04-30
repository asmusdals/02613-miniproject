#!/bin/bash

#BSUB -J q10_nsys_profile
#BSUB -q gpua100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 01:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -M 8GB
#BSUB -o batch_output/q10_nsys_profile_%J.out
#BSUB -e batch_output/q10_nsys_profile_%J.err
#BSUB -u s216137@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# Submit from the project root:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results reports
#   bsub < jobs/q10_nsys_profile.sh

PROJECT_DIR="$HOME/Documents/02613/miniproject"
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: Missing project dir: $PROJECT_DIR"
  exit 2
fi
cd "$PROJECT_DIR"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p batch_output results reports

# Warm up CUDA/CuPy before profiling to avoid capturing JIT compilation
echo "Warming up CuPy with N=1"
python -u scripts/q9_cupy.py 1 > /dev/null

# Profile q9 (naive, per-building transfers) with N=10
echo "Running nsys profile on q9_cupy (naive) with N=10"
nsys profile \
  --stats=true \
  --output=reports/q10_nsys_q9 \
  python -u scripts/q9_cupy.py 10 > /dev/null

# Export human-readable stats
nsys stats reports/q10_nsys_q9.nsys-rep > reports/q10_nsys_q9_stats.txt 2>&1

echo "nsys profile written: reports/q10_nsys_q9.nsys-rep"
echo "Text stats written:   reports/q10_nsys_q9_stats.txt"
