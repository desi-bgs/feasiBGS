#!/bin/bash -l 
#SBATCH -q debug 
#SBATCH -N 1 
#SBATCH -t 00:30:00 
#SBATCH -J RR_g15_mockexp_KSsky_lowHA
#SBATCH -o _RR_g15_mockexp_KSsky_lowHA.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

module load python/2.7-anaconda
source activate myenv0 

source /project/projectdirs/desi/software/desi_environment.sh 18.3

export OMP_NUM_THREADS=1 
dir_spec=$FEASIBGS_DIR"/spectra/gamadr3_legacydr7/"

# read in the exposure numbers from file 
f_exp=$FEASIBGS_DIR"bgs_survey_exposures.withsun.iexp_metabin.dat"

#while read -r line; do
#    iexp="$line"
#    echo "-- running redrock --" 
#    for skymodel in "KS"; do #"newKS"; do 
#        echo "-- $skymodel sky model --" 
#        f_str="g15.sim_spectra.mockexp_block.1of64.480.iexp4729."$skymodel"sky"
#        f_spec=$dir_spec$f_str".fits"
#        f_redr=$dir_spec$f_str".rr.fits"
#        f_zout=$dir_spec$f_str".rr.h5"
#        echo "-- "$f_spec" --" 
#        if [ ! -f $f_spec ]; then
#            echo "$f_spec not found!"
#            break 
#        fi
#        if [ ! -f $f_redr ]; then 
#            # takes ~9 mins 
#            rrdesi --mp 24 --zbest $f_redr --output $f_zout $f_spec
#        fi 
#    done 
#done < "$f_exp"

# low Halpha 
f_str="g15.sim_spectra.mockexp_block.lowHalpha.texp_default.iexp2910.KSsky"
f_spec=$dir_spec$f_str".fits"
f_redr=$dir_spec$f_str".rr.fits"
f_zout=$dir_spec$f_str".rr.h5"

rrdesi --mp 24 --zbest $f_redr --output $f_zout $f_spec
