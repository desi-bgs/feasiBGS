#!/bin/bash/

dir_spec=$FEASIBGS_DIR"spectra/gamadr3_legacydr7/"

# read in the exposure numbers from file 
#f_exp=$FEASIBGS_DIR"bgs_survey_exposures.withsun.iexp_metabin.dat"
f_exp=$FEASIBGS_DIR"bgs_survey_exposures.withsun.obscond_random.dat"

nblock=64
#export OMP_NUM_THREADS=1 

while read -r line; do
    iexp="$line"
    for iblock in {2..3}; do 
        echo "-- mock exposure #$iexp; block $iblock --"
        for skymodel in "KS" "newKS"; do 
            f_str="g15.sim_spectra.mockexp_block."$iblock"of64.texp_default.iexp"$iexp"."$skymodel"sky"
            #f_str="g15.sim_spectra.mockexp_block."$iblock"of64.480.iexp"$iexp"."$skymodel"sky"
            #f_str="g15.sim_spectra.mockexp_block.480.iexp"$iexp"."$skymodel"sky"
            f_spec=$dir_spec$f_str".fits"

            # create mock spectra with old KS sky model and new sky model  
            # exposure time set by survey sim 
            if [ ! -f $f_spec ]; then
                echo "-- generating simulated spectra --" 
                echo "-- $f_spec --" 
                python /users/changhoon/projects/feasibgs/run/bgs_mockexp_spectra.py $iblock $iexp default 
            fi
            # specified exposure time  
            #python /Users/ChangHoon/projects/feasiBGS/run/bgs_mockexp_spectra.py $iblock $iexp 480
            
            # run the simulated spectra through redrock
            #echo "-- running redrock --" 
            #f_redr=$dir_spec$f_str".rr.fits"
            #f_zout=$dir_spec$f_str".rr.h5"
            #if [ ! -f $f_redr ]; then 
            #    rrdesi --mp 4 --zbest $f_redr --output $f_zout $f_spec
            #fi 
        done 
    done 
done < "$f_exp"

#for iexp in 1009 1179 1180 1194 1366; do 
#    for iblock in 2; do 
#        echo "-- mock exposure #$iexp; block $iblock --"
#        # create mock spectra with old KS sky model and new sky model  
#        echo "-- generating simulated spectra --" 
#        python /users/changhoon/projects/feasibgs/run/bgs_mockexp_spectra.py $iblock $iexp default 
#    done 
#done 


#for iblock in {3..5}; do  
#    echo "-- "$iblock" --"
#    python /Users/ChangHoon/projects/feasiBGS/run/bgs_mockexp_spectra.py $iblock 0 480
#    python /Users/ChangHoon/projects/feasiBGS/run/bgs_mockexp_spectra.py $iblock 2268 480
#done 

# low Halpha block 
#python /users/changhoon/projects/feasibgs/run/bgs_mockexp_spectra.py lowHA 2910 
