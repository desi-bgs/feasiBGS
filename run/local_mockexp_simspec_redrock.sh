#!/bin/bash/

dir_spec="/Users/ChangHoon/data/feasiBGS/spectra/gamadr3_legacydr7/"

export OMP_NUM_THREADS=1 
nblock=64

for iblock in {3..5}; do 
    for iexp in 0 2268; do 
        for skymodel in "KS" "newKS"; do 
            echo "-- mock exposure #"$iexp" --"
            f_str="g15.sim_spectra.mockexp_block."$iblock"of"$nblock".480.iexp"$iexp"."$skymodel"sky"
            #f_str="g15.sim_spectra.mockexp_block.480.iexp"$iexp"."$skymodel"sky"
            f_spec=$dir_spec$f_str".fits"
            f_redr=$dir_spec$f_str".rr.fits"
            f_zout=$dir_spec$f_str".rr.h5"
            echo "-- "$f_spec" --" 
            rrdesi --mp 4 --zbest $f_redr --output $f_zout $f_spec
        done
    done 
done 
