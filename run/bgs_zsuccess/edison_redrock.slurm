#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
#SBATCH -J RR_g15_v2_iobs14
#SBATCH -o _RR_g15_v2_iobs14.o 
#SBATCH -L SCRATCH,project 

module load python/2.7-anaconda
source activate myenv0 

source /project/projectdirs/desi/software/desi_environment.sh 18.3

export OMP_NUM_THREADS=1 
dir_spec=$FEASIBGS_DIR"bgs_zsuccess/"

iexp=14
for skymodel in "old" "new"; do 
    now=$(date +"%T") 
    echo "-- $iexp exposure; block $iblock; $skymodel sky model -- $now" 
    f_str="g15.simSpectra.3000.texp_default.iexp"$iexp"of15spacefill."$skymodel"_sky.v2"
    f_spec=$dir_spec$f_str".fits"
    f_redr=$dir_spec$f_str".rr.fits"
    f_zout=$dir_spec$f_str".rr.h5"
    if [ ! -f $f_spec ]; then
        echo "$f_spec not found!"
        break 
    fi
    if [ ! -f $f_redr ]; then 
        rrdesi --mp 24 --zbest $f_redr --output $f_zout $f_spec
    fi 
done
now=$(date +"%T") 
echo "-- finished -- $now" 
