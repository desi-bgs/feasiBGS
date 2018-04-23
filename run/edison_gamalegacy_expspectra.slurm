#!/bin/bash -l 
#SBATCH -p regular 
#SBATCH -N 1 
#SBATCH -t 05:00:00 
#SBATCH -J gleg_expSpec_dark_z
#SBATCH -o _gleg_expSpec_dark_z.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

# exposed spectra
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra bright 1
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra dark 1

# run through redrock
#srun -n 1 -c 16 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py redshift bright 1 16
srun -n 1 -c 16 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py redshift dark 1 16
