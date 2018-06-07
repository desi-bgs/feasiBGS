#!/bin/bash -l 
#SBATCH -q debug  
#SBATCH -N 1 
#SBATCH -t 00:15:00 
#SBATCH -J gleg_expSpec_g15_dark_fainteml_redrock
#SBATCH -o _gleg_expSpec_g15_dark_fainteml_redrock.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

source /project/projectdirs/desi/software/desi_environment.sh 18.3

sky="dark"
field="g15"
seed=1
exptime=300
if [ $field = "g15" ]; then
    nblocks=64
elif [ $field = "g09" ]; then
    nblocks=26
fi

dir_spec=$CSCRATCH"/feasibgs/spectra/"
dir_redr=$CSCRATCH"/feasibgs/redrock/"
# run through redrock
#for iblock in $(seq 20 $nblocks); do 
#    f_spectra=$dir_spec"GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.fits"
#    f_redrock=$dir_redr"GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.fits"
#    f_zoutput=$dir_redr"GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.h5"
#    echo "------------------------------"
#    echo "block "$iblock" of "$nblocks
#    #echo $f_spectra
#    #echo $f_redrock
#    #echo $f_zoutput
#    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra
#done 

# run through redrock (parallized; takes ~11mins) 
f_spectra=$CSCRATCH"/feasibgs/spectra/GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.fits"
f_redrock=$CSCRATCH"/feasibgs/redrock/GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.redrock.fits"
f_zoutput=$CSCRATCH"/feasibgs/redrock/GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.redrock.h5"
echo $f_spectra
echo $f_redrock
echo $f_zoutput
srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

# run through redrock (parallized; takes ~11mins) 
#f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.fits"
#f_redrock=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.zbest.fits"
#f_zoutput=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.zbest.h5"
#srun -N 1 -n 12 -c 1 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

now=$(date +"%T") 
echo "end time ... $now"
