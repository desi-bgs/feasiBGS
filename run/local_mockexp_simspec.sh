#!/bin/bash/
for iblock in {3..5}; do  
    echo "-- "$iblock" --"
    python /Users/ChangHoon/projects/feasiBGS/run/bgs_mockexp_spectra.py $iblock 0 480
    python /Users/ChangHoon/projects/feasiBGS/run/bgs_mockexp_spectra.py $iblock 2268 480
done 
