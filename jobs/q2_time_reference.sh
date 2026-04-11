#!/bin/bash

#BSUB -J q2_ref_time
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 00:20
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q2_ref_time_%J.out
#BSUB -e batch_output/q2_ref_time_%J.err

set -euo pipefail

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p batch_output results

N=20
TOTAL=4571

START=$SECONDS
python -u ~/Documents/02613/miniproject/scripts/simulate.py "$N" > "results/reference_${N}.csv"
ELAPSED=$((SECONDS - START))

echo "Elapsed seconds: $ELAPSED (for N=$N)"
echo "Estimated seconds for all $TOTAL: $(( ELAPSED * TOTAL / N ))"
echo "CSV written: results/reference_${N}.csv"

