#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
#SBATCH -C haswell
#SBATCH -J RR_surveysim_exp4
#SBATCH -o _RR_surveysim_exp4.o 
#SBATCH -L SCRATCH,project 

source /project/projectdirs/desi/software/desi_environment.sh 19.2

export OMP_NUM_THREADS=1 
dir_spec=$FEASIBGS_DIR"bgs_zsuccess/"

now=$(date +"%T") 

dir_spec=$CSCRATCH"/feasibgs/bgs_zsuccess/"

f_str="GALeg.g15.sourceSpec.exposures_surveysim_fork_150sv0p3.subset.14spacefill.4"
f_spec=$dir_spec$f_str".fits"
f_redr=$dir_spec$f_str".rr.fits"
f_zout=$dir_spec$f_str".rr.h5"
rrdesi --mp 32 --zbest $f_redr --output $f_zout $f_spec

now=$(date +"%T") 
echo "-- finished -- $now" 
