#!/bin/bash

#BSUB -J q4_profile
#BSUB -q hpc
#BSUB -n 1
#BSUB -W 03:00
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 4GB
#BSUB -o batch_output/q4_profile_%J.out
#BSUB -e batch_output/q4_profile_%J.err
#BSUB -u s224473@student.dtu.dk
#BSUB -B
#BSUB -N

set -eo pipefail

# Submit from the project root:
#   cd ~/Documents/02613/miniproject
#   mkdir -p batch_output results
#   bsub < jobs/q4_profile_jacobi.sh

PROJECT_DIR="$HOME/Documents/02613/miniproject"
cd "$PROJECT_DIR"
mkdir -p batch_output results

source /dtu/projects/02613_2025/conda/conda_init.sh
conda activate 02613_2026

N=1

kernprof -l -v scripts/q4_profile_jacobi.py "$N" > "results/q4_kernprof_N${N}.txt"

echo "Profile written: results/q4_kernprof_N${N}.txt"

