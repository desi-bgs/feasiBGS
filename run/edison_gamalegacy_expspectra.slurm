#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 00:20:00 
#SBATCH -J gleg_expSpec_g15_dark_fixedvdisp
#SBATCH -o _gleg_expSpec_g15_dark_fixedvdisp.o 
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
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra $field $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_faintemline $field $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_fixedvdispEmline $field $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_noemline $sky $seed $exptime

now=$(date +"%T") 
echo "end time ... $now"
