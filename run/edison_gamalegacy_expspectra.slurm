#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 01:30:00 
#SBATCH -J gleg_expSpec_noemline_bright
#SBATCH -o _gleg_expSpec_noemline_bright.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

sky="bright"
seed=1
exptime=300
# exposed spectra (takes ~5x40 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_faintemline $sky $seed $exptime
# exposed spectra (takes ~5 mins) 
#srun -n 1 -c 1 python /global/homes/c/chahah/projects/feasiBGS/run/gamalegacy_spectra.py spectra_noemline $sky $seed $exptime

#source /project/projectdirs/desi/software/desi_environment.sh 18.3

# run through redrock
#for iblock in 1; do # {1..40}; do 
#    f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed1."$iblock"of40blocks.fits"
#    f_redrock=$CSCRATCH"/feasibgs/redrock/gama_legacy.expSpectra."$sky"sky.seed1."$iblock"of40blocks.zbest.fits"
#    echo $f_spectra
#    echo $f_redrock
#    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock $f_spectra
#done 

# run through redrock (parallized; takes ~11mins) 
#f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.fits"
#f_redrock=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.zbest.fits"
#f_zoutput=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.zbest.h5"
#echo $f_spectra
#echo $f_redrock
#echo $f_zoutput
#srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

# run through redrock (parallized; takes ~11mins) 
f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.fits"
f_redrock=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.zbest.fits"
f_zoutput=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.zbest.h5"
srun -N 1 -n 3 -c 1 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

now=$(date +"%T") 
echo "end time ... $now"
