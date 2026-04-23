#!/bin/bash

#BSUB -J q6_dynamic
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 03:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q6_dynamic_%J.out
#BSUB -e batch_output/q6_dynamic_%J.err
#BSUB -u s224473@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# Submit from the project root:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results
#   bsub < jobs/q6_parallel_dynamic.sh

PROJECT_DIR="$HOME/Documents/02613/miniproject"
cd "$PROJECT_DIR"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p batch_output results

N=100
WORKERS_LIST="1 2 4 8"

echo "N,workers,elapsed_seconds"
for WORKERS in $WORKERS_LIST; do
  START=$SECONDS
  python -u scripts/q6_parallel_dynamic.py "$N" --workers "$WORKERS" > "results/q6_dynamic_N${N}_W${WORKERS}.csv"
  ELAPSED=$((SECONDS - START))
  echo "$N,$WORKERS,$ELAPSED"
done
