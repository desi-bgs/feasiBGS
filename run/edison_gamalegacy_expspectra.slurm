#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
#SBATCH -J gleg_expSpec_dark
#SBATCH -o _gleg_expSpec_dark.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py bright 1
srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py dark 1
