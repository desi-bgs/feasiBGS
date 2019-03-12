#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 00:45:00 
#SBATCH -J gleg_g15
#SBATCH -o _gleg_g15.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

field="g15"

srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy.py gleg $field 

now=$(date +"%T") 
echo "end time ... $now"
