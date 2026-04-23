#!/bin/bash

#BSUB -J q7_numba_cpu
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q7_numba_cpu_%J.out
#BSUB -e batch_output/q7_numba_cpu_%J.err
#BSUB -u s224473@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# Submit from the project root:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results
#   bsub < jobs/q7_numba_cpu.sh

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

echo "Warming up Numba compilation/cache with N=1"
python -u scripts/q7_numba_cpu.py 1 > /dev/null

START=$SECONDS
python -u scripts/q7_numba_cpu.py "$N" > "results/q7_numba_cpu_N${N}.csv"
ELAPSED=$((SECONDS - START))

echo "Elapsed seconds: $ELAPSED (for N=$N, after warm-up)"
echo "Estimated seconds for all $TOTAL: $(( ELAPSED * TOTAL / N ))"
echo "CSV written: results/q7_numba_cpu_N${N}.csv"
