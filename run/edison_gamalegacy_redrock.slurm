#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 05:00:00 
#SBATCH -J redrock_synSpec_g15_dark_p2of2
#SBATCH -o _redrock_synSpec_g15_dark_p2of2.o 
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
    nblocks=13
fi

dir_spec=$CSCRATCH"/feasibgs/spectra/gamadr3_legacydr7/"
# run blocks through redrock (takes mins for 5000 object block)
for iblock in $(seq 9 $nblocks); do 
    f_spectra=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.fits"
    f_redrock=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.fits"
    f_zoutput=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.h5"
    echo "------------------------------"
    echo "block "$iblock" of "$nblocks
    echo "running "$f_spectra" through redrock"
    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra
done 

# run through redrock (parallized; takes ~11mins) 
#f_spectra=$CSCRATCH"/feasibgs/spectra/GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.fits"
#f_redrock=$CSCRATCH"/feasibgs/redrock/GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.redrock.fits"
#f_zoutput=$CSCRATCH"/feasibgs/redrock/GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime".faintEmLine.redrock.h5"
#echo $f_spectra
#echo $f_redrock
#echo $f_zoutput
#srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

# run through redrock (parallized; takes ~11mins) 
#f_spectra=$CSCRATCH"/feasibgs/spectra/GamaLegacy.g15.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".5of64blocks_fixedvdispEmLine_150kms.fits"
#f_redrock=$CSCRATCH"/feasibgs/redrock/GamaLegacy.g15.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".5of64blocks_fixedvdispEmLine_150kms.zbest.fits"
#f_zoutput=$CSCRATCH"/feasibgs/redrock/GamaLegacy.g15.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".5of64blocks_fixedvdispEmLine_150kms.zbest.h5"
#echo $f_spectra
#echo $f_redrock
#echo $f_zoutput
#srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

# run through redrock (parallized; takes ~11mins) 
#f_spectra=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.fits"
#f_redrock=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.zbest.fits"
#f_zoutput=$CSCRATCH"/feasibgs/spectra/gama_legacy.expSpectra."$sky"sky.seed"$seed".exptime"$exptime".noEmLine.zbest.h5"
#srun -N 1 -n 12 -c 1 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

now=$(date +"%T") 
echo "end time ... $now"
