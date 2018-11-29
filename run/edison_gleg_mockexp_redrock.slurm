#!/bin/bash -l 
#SBATCH -q debug 
#SBATCH -N 1 
#SBATCH -t 00:10:00 
#SBATCH -J RR_g15_mockexp_sky
#SBATCH -o _RR_g15_mockexp_sky.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

source /project/projectdirs/desi/software/desi_environment.sh 18.3

export OMP_NUM_THREADS=1 
dir_spec=$CSCRATCH"/feasibgs/spectra/gamadr3_legacydr7/"

echo "-- running redrock --" 
for skymodel in "KS"; do #"newKS"; do 
    echo "-- $skymodel sky model --" 
    f_str="g15.sim_spectra.mockexp_block.1of64.480.iexp4729."$skymodel"sky"
    f_spec=$dir_spec$f_str".fits"
    f_redr=$dir_spec$f_str".rr.fits"
    f_zout=$dir_spec$f_str".rr.h5"
    echo "-- "$f_spec" --" 
    if [ ! -f $f_spec ]; then
        echo "$f_spec not found!"
        break 
    fi
    if [ ! -f $f_redr ]; then 
        rrdesi --mp 24 --zbest $f_redr --output $f_zout $f_spec
    fi 
done 
