#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 03:00:00 
#SBATCH -C haswell
#SBATCH -J RR_uparchuptemp_g15_dark
#SBATCH -o _RR_uparchuptemp_g15_dark.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

source /project/projectdirs/desi/software/desi_environment.sh 18.3
module unload redrock/0.10.1
export PATH="/global/homes/c/chahah/.local/edison/2.7-anaconda-4.4/bin:$PATH"

sky="dark"
field="g15"
seed=1
exptime=300
if [ $field = "g15" ]; then
    nblocks=13
fi

#dir_archetypes=$HOME"/projects/packages/redrock-archetypes/"

dir_spec=$CSCRATCH"/feasibgs/spectra/gamadr3_legacydr7/"
# run blocks through redrock (takes mins for 5000 object block)
#for iblock in $(seq 1 1); do 
#    f_spectra=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.fits"
#    f_redrock=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.archetype.fits"
#    f_zoutput=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.archetype.h5"
#    echo "------------------------------"
#    echo "block "$iblock" of "$nblocks
#    echo "running "$f_spectra" through redrock"
#    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --archetypes $dir_archetypes --output $f_zoutput $f_spectra
#done 

#dir_archetypes=$HOME"/projects/redrock-archetypes/"
# run blocks through redrock (takes mins for 5000 object block)
#for iblock in $(seq 1 1); do 
#    f_spectra=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.fits"
#    f_redrock=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.uparchetype.fits"
#    f_zoutput=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.uparchetype.h5"
#    echo "------------------------------"
#    echo "block "$iblock" of "$nblocks
#    echo "running "$f_spectra" through redrock"
#    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --archetypes $dir_archetypes --output $f_zoutput $f_spectra
#done 

# run redrock with updated template and archetype from 
# https://github.com/desihub/redrock-archetypes/pull/7
# https://github.com/desihub/redrock-templates/pull/8
dir_archetypes=$HOME"/projects/packages/redrock-archetypes/"
dir_templates=$HOME"/projects/packages/redrock-templates/"
for iblock in $(seq 1 1); do 
    f_spectra=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.fits"
    f_redrock=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.uparch.uptemp.fits"
    f_zoutput=$dir_spec$field".synSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.uparch.uptemp.h5"
    echo "------------------------------"
    echo "block "$iblock" of "$nblocks
    echo "running "$f_spectra" through redrock"
    srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --archetypes $dir_archetypes --templates $dir_templates --output $f_zoutput $f_spectra
done 

now=$(date +"%T") 
echo "end time ... $now"
