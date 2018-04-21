#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 05:00:00 
#SBATCH -J gamalegacy_expSpec 
#SBATCH -o _gamalegacy_expSpec.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py
