#!/bin/bash -l 
#SBATCH -q debug 
#SBATCH -N 1 
#SBATCH -t 00:15:00 
#SBATCH -J gleg_expSpec_faintemline_bright
#SBATCH -o _gleg_expSpec_faintemline_bright.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

# exposed spectra
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra bright 1
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra dark 1
srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_faintemline bright 1
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_faintemline dark 1

#source /project/projectdirs/desi/software/desi_environment.sh 18.3

# run through redrock
#sky="bright"
#for iblock in 1; do # {1..40}; do 
#    f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed1."$iblock"of40blocks.fits"
#    f_redrock=$CSCRATCH"/feasibgs/redrock/gama_legacy.expSpectra."$sky"sky.seed1."$iblock"of40blocks.zbest.fits"
#    echo $f_spectra
#    echo $f_redrock
#    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock $f_spectra
#done 
