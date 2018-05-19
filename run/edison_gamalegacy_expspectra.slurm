#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
#SBATCH -J gleg_expSpec_faintemline_dark_z
#SBATCH -o _gleg_expSpec_faintemline_dark_z.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

sky="dark"
seed=1
# exposed spectra
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra $sky $seed
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_faintemline $sky $seed

source /project/projectdirs/desi/software/desi_environment.sh 18.3

# run through redrock
#for iblock in 1; do # {1..40}; do 
#    f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed1."$iblock"of40blocks.fits"
#    f_redrock=$CSCRATCH"/feasibgs/redrock/gama_legacy.expSpectra."$sky"sky.seed1."$iblock"of40blocks.zbest.fits"
#    echo $f_spectra
#    echo $f_redrock
#    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock $f_spectra
#done 

f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".faintEmLine.fits"
f_redrock=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".faintEmLine.zbest.fits"
f_zoutput=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".faintEmLine.zbest.h5"
echo $f_spectra
echo $f_redrock
echo $f_zoutput
srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra
