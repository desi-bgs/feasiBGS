#!/bin/bash -l 
#SBATCH -q regular 
#SBATCH -N 1 
#SBATCH -t 01:00:00 
#SBATCH -C knl 
#SBATCH -J surveysim_200_v2
#SBATCH -o _surveysim_200_v2.o 
#SBATCH -L SCRATCH,project 

now=$(date +"%T") 
echo "start time ... $now"

#######################################################################
# forked branch desi_surveysim_branch which includes bright time exposure
# factor  
#######################################################################
conda activate desi_surveysim_branch

DIR=/global/homes/c/chahah/projects/feasiBGS/run/survey_sim

## 130s with twilight
#srun -n 1 -c 1 python $DIR/survey_sim.py 130s_skybranch_v2 config_130s.yaml True True
## 130s without twilight
#srun -n 1 -c 1 python $DIR/survey_sim.py 130s_skybranch_v2 config_130s.yaml False True

# 150s with twilight
#srun -n 1 -c 1 python $DIR/survey_sim.py 150s_skybranch_v2 config_150s.yaml True True
# 150s without twilight
#srun -n 1 -c 1 python $DIR/survey_sim.py 150s_skybranch_v2 config_150s.yaml False True

# 200s without twilight
srun -n 1 -c 1 python $DIR/survey_sim.py 200s_skybranch_v2 config_200s.yaml False True
# 200s with twilight
srun -n 1 -c 1 python $DIR/survey_sim.py 200s_skybranch_v2 config_200s.yaml True True

# 250s without twilight
#srun -n 1 -c 1 python $DIR/survey_sim.py 250s_skybranch config_250s.yaml False True
# 250s with twilight
#srun -n 1 -c 1 python $DIR/survey_sim.py 250s_skybranch config_250s.yaml True True

## 300s without twilight 
#srun -n 1 -c 1 python $DIR/survey_sim.py 150s_skybranch config_300s.yaml False True
## 300s with twilight
#srun -n 1 -c 1 python $DIR/survey_sim.py 150s_skybranch config_300s.yaml True True

now=$(date +"%T") 
echo "end time ... $now"
