#!/bin/bash -l 
#SBATCH -q debug 
#SBATCH -N 1 
#SBATCH -t 00:15:00 
#SBATCH -J zfail_redrock 
#SBATCH -o zfail_redrock.o 
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
f_str="GamaLegacy."$field".expSpectra."$sky"sky.seed"$seed".exptime"$exptime"."$iblock"of"$nblocks"blocks.zfail"
f_spectra=$dir_spec$f_str".fits"
f_redrock=$dir_redr$f_str".redrock.fits"
f_zoutput=$dir_redr$f_str".redrock.h5"

# regular redrock run
srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --output $f_zoutput $f_spectra

f_redrock=$dir_redr$f_str".redrock.archetype.fits"
f_zoutput=$dir_redr$f_str".redrock.archetype.h5"
path_to_archetypes=$HOME"/projects/packages/redrock-archetypes/"

# regular redrock run with archetype 
srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --archetype $path_to_archetypes --output $f_zoutput $f_spectra

f_redrock=$dir_redr$f_str".redrock.archetype.newgaltemp.fits"
f_zoutput=$dir_redr$f_str".redrock.archetype.newgaltemp.h5"
template_file=$CSCRATCH"/feasibgs/rrtemplate-galaxy.fits"
# regular redrock run with archetype and updated galaxy template
srun -N 1 -n 12 -c 2 rrdesi_mpi --zbest $f_redrock --templates $template_file --archetype $path_to_archetypes --output $f_zoutput $f_spectra
now=$(date +"%T") 
echo "end time ... $now"
