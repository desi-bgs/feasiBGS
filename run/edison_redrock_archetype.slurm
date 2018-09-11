#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 00:30:00 
#SBATCH -J archetype_test 
#SBATCH -o _archetype_test.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

#module load python/2.7-anaconda

source /project/projectdirs/desi/software/desi_environment.sh 18.3
module unload redrock/0.10.1
export PATH="/global/homes/c/chahah/.local/edison/2.7-anaconda-4.4/bin:$PATH"

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

iblock=5
f_spectra=$dir_spec"GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.fits"
f_redrock=$dir_redr"GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.archetype.newgaltemp.fits"
f_zoutput=$dir_redr"GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.redrock.archetype.newgaltemp.h5"
echo "------------------------------"
echo "block "$iblock" of "$nblocks
echo $f_spectra
echo $f_redrock
echo $f_zoutput
path_to_archetypes=$HOME"/projects/packages/redrock-archetypes/"
srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --archetype $path_to_archetypes --output $f_zoutput $f_spectra

now=$(date +"%T") 
echo "end time ... $now"
