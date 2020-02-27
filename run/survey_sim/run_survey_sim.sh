#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# default
python $DIR/survey_sim.py default config_default.yaml 
# 150s 
#python $DIR/survey_sim.py 150s config_150s.yaml

