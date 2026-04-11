#!/bin/bash

#BSUB -J autolab_2_1                 # Jobnavn (vises i jobkøen)
#BSUB -q hpc                         # Queue der bruges på DTU HPC
#BSUB -n 10                          # Antal CPU-cores der reserveres
#BSUB -W 10                          # Maksimal køretid (walltime) i minutter
#BSUB -R "select[model==XeonGold6126]"  # Kør på specifik CPU-model for reproducérbare benchmarks
#BSUB -R "span[hosts=1]"             # Alle cores skal ligge på samme node
#BSUB -R "rusage[mem=2GB]"           # Reserver 2GB RAM pr core
#BSUB -M 2GB                         # Hard memory limit pr proces
#BSUB -o job_%J.out                  # Standard output fil (%J = job-id)
#BSUB -e job_%J.err                  # Error output fil
#BSUB -u s224473@student.dtu.dk      # Email til jobnotifikationer
#BSUB -B                             # Send email når jobbet starter
#BSUB -N                             # Send email når jobbet slutter

source /dtu/projects/02613_2025/conda/conda_init.sh   # Load conda environment
conda activate 02613                                  # Aktiver kursus-miljø

time python -u pi_serial.py