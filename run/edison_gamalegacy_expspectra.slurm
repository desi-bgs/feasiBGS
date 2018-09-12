#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 01:30:00 
#SBATCH -J synSpec_g15_dark
#SBATCH -o _synSpec_g15_dark.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

sky="dark"
field="g15"
seed=1
exptime=300
# exposed spectra (each block takes ~5 mins?) 
srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra $field $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_faintemline $field $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_fixedvdispEmline $field $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_noemline $sky $seed $exptime

now=$(date +"%T") 
echo "end time ... $now"
