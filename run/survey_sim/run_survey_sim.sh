#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# python survey_sim.py name fconfig twilight 
# default
python $DIR/survey_sim.py default config_default.yaml False
python $DIR/survey_sim.py default config_default.yaml True
# 150s 
python $DIR/survey_sim.py 150s config_150s.yaml False
python $DIR/survey_sim.py 150s config_150s.yaml True

