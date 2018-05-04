#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 00:30:00 
#SBATCH -J SDSSspAllcut
#SBATCH -o _SDSSspAllcut.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/sdss_spAll_cut.py
