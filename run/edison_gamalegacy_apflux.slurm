#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -C haswell
#SBATCH -t 02:00:00 
#SBATCH -J gamalegacy_apflux 
#SBATCH -o gamalegacy_apflux.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/tests/catalog_test.py
