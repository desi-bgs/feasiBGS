#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# python survey_sim.py name fconfig twilight 

#######################################################################
# master branch desi_surveysim 
#######################################################################
#source activate desi_surveysim
# default
#python $DIR/survey_sim.py default config_default.yaml False
#python $DIR/survey_sim.py default config_default.yaml True
# 150s 
#python $DIR/survey_sim.py 150s config_150s.yaml False
#python $DIR/survey_sim.py 150s config_150s.yaml True
#conda deactivate 

#######################################################################
# forked branch desi_surveysim_branch
#######################################################################
source ~/.bashrc    
conda activate desi_surveysim_branch
# default
#python $DIR/survey_sim.py default config_default.yaml False
#python $DIR/survey_sim.py default config_default.yaml True

# 130
#python $DIR/survey_sim.py 130s_skybranch_v3 config_130s.yaml False True
#python $DIR/survey_sim.py 130s_skybranch_v3 config_130s.yaml True True

# 150s 
#python $DIR/survey_sim.py 150s_skybranch_v4 config_150s.yaml False True 
#python $DIR/survey_sim.py 150s_skybranch_v4 config_150s.yaml True True

# 200s 
#python $DIR/survey_sim.py 200s_skybranch_v3 config_200s.yaml False True
#python $DIR/survey_sim.py 200s_skybranch_v3 config_200s.yaml True True

#######################################################################
# reduced footprint (does not include tiles near the ecliptic) 
#######################################################################
# 130s
#python $DIR/survey_sim.py 130s_bgs12000_skybranch_v4 config_130s.yaml False True 12000
#python $DIR/survey_sim.py 130s_bgs13000_skybranch_v4 config_130s.yaml False True 13000
#python $DIR/survey_sim.py 130s_bgs14000_skybranch_v4 config_130s.yaml False True

# 140s
#python $DIR/survey_sim.py 140s_bgs13000_skybranch_v4 config_140s.yaml False True 13000

# 150s
#python $DIR/survey_sim.py 150s_bgs10000_skybranch_v4 config_150s.yaml True True 10000
#python $DIR/survey_sim.py 150s_bgs11000_skybranch_v4 config_150s.yaml True True 11000
#python $DIR/survey_sim.py 150s_bgs12000_skybranch_v4 config_150s.yaml True True 12000
#python $DIR/survey_sim.py 150s_bgs13000_skybranch_v4 config_150s.yaml True True 13000

# 170s (this corresponds to 95% redshift completeness at r~19.5
#python $DIR/survey_sim.py 170s_bgs10000_skybranch_v6 config_170s.yaml False True 10000
#python $DIR/survey_sim.py 170s_bgs11000_skybranch_v6 config_170s.yaml False True 11000
#python $DIR/survey_sim.py 170s_bgs12000_skybranch_v6 config_170s.yaml False True 12000
#python $DIR/survey_sim.py 170s_bgs14000_skybranch_v7 config_170s.yaml False True False # full footprintk
#python $DIR/survey_sim.py 170s_bgs14000_skybranch_v6 config_170s.yaml False True True # full footprintk

# 180s
#python $DIR/survey_sim.py 180s_bgs14000_skybranch_v6 config_180s.yaml False True False # full footprintk
#python $DIR/survey_sim.py 180s_bgs14000_skybranch_v6 config_180s.yaml False True True # full footprintk


#python $DIR/survey_sim.py 190s_bgs14000_skybranch_v6 config_190s.yaml False True # full footprintk
#python $DIR/survey_sim.py 190s_bgs14000_skybranch_v6 config_190s.yaml False True True # full footprintk
#python $DIR/survey_sim.py 250s_bgs12000_skybranch_v6 config_250s.yaml False True 12000

# 200s (this corresponds to 95% redshift completeness at r~20.0
#python $DIR/survey_sim.py 200s_bgs10000_skybranch_v6 config_200s.yaml False True 10000
#python $DIR/survey_sim.py 200s_bgs11000_skybranch_v6 config_200s.yaml False True 11000
#python $DIR/survey_sim.py 200s_bgs12000_skybranch_v6 config_200s.yaml False True 12000
#python $DIR/survey_sim.py 200s_bgs13000_skybranch_v6 config_200s.yaml False True 13000
#python $DIR/survey_sim.py 200s_bgs13000_skybranch_v6 config_200s.yaml False True True 13000
#python $DIR/survey_sim.py 200s_bgs13000_skybranch_v7 config_200s.yaml False True False 13000
#python $DIR/survey_sim.py 200s_bgs14000_skybranch_v7 config_200s.yaml False True False 

# 270s (this corresponds to 95% redshift completeness at r~20.0
#python $DIR/survey_sim.py 270s_bgs10000_skybranch_v6 config_270s.yaml False True 10000
#python $DIR/survey_sim.py 270s_bgs11000_skybranch_v6 config_270s.yaml False True 11000

#python $DIR/survey_sim.py 300s_bgs10000_skybranch_v6 config_300s.yaml False True 10000
python $DIR/survey_sim.py 270s_bgs13000_skybranch_v7 config_270s.yaml False True False 10000

#conda deactivate 
