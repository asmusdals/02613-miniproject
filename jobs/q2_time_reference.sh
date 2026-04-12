#!/bin/bash

#BSUB -J q2_ref_time
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q2_ref_time_%J.out
#BSUB -e batch_output/q2_ref_time_%J.err
#BSUB -u s224473@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# NOTE:
# LSF writes the -o/-e files relative to the directory you submit from.
# The batch_output directory must already exist before bsub starts the job.
# Use:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results
#   bsub < jobs/q2_time_reference.sh

PROJECT_DIR="$HOME/Documents/02613/miniproject"
if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "ERROR: Missing project dir: $PROJECT_DIR"
  exit 2
fi
cd "$PROJECT_DIR"

# Do not use `set -u` while sourcing DTU's conda init script:
# it references some optional environment variables.
source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

mkdir -p "$PROJECT_DIR/batch_output" "$PROJECT_DIR/results"

N=10
TOTAL=4571

START=$SECONDS
if [[ ! -f scripts/simulate.py ]]; then
  echo "ERROR: Missing scripts/simulate.py in $PROJECT_DIR"
  exit 2
fi
python -u scripts/simulate.py "$N" > "results/reference_${N}.csv"
ELAPSED=$((SECONDS - START))

echo "Elapsed seconds: $ELAPSED (for N=$N)"
echo "Estimated seconds for all $TOTAL: $(( ELAPSED * TOTAL / N ))"
echo "CSV written: results/reference_${N}.csv"
