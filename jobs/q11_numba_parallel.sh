#!/bin/bash

#BSUB -J q11_numba_parallel
#BSUB -q hpc
#BSUB -n 8
#BSUB -W 03:00
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q11_numba_parallel_%J.out
#BSUB -e batch_output/q11_numba_parallel_%J.err
#BSUB -u s224473@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# Submit from the project root:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results
#   bsub < jobs/q11_numba_parallel.sh

PROJECT_DIR="$HOME/Documents/02613/miniproject"
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: Missing project dir: $PROJECT_DIR"
  exit 2
fi
cd "$PROJECT_DIR"

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p batch_output results

N=100
TOTAL=4571
WORKERS_LIST="1 2 4 8"

echo "N,workers,elapsed_seconds"
for WORKERS in $WORKERS_LIST; do
  START=$SECONDS
  python -u scripts/q11_numba_parallel.py "$N" --workers "$WORKERS" \
    > "results/q11_numba_parallel_N${N}_W${WORKERS}.csv"
  ELAPSED=$((SECONDS - START))
  echo "$N,$WORKERS,$ELAPSED"
done

# Extrapolate full-dataset estimate from the fastest run (W=8)
START=$SECONDS
python -u scripts/q11_numba_parallel.py "$N" --workers 8 \
  > "results/q11_numba_parallel_N${N}_W8.csv"
ELAPSED_BEST=$((SECONDS - START))

echo ""
echo "Best run (W=8): $ELAPSED_BEST seconds for N=$N"
echo "Estimated seconds for all $TOTAL: $(( ELAPSED_BEST * TOTAL / N ))"
