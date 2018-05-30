#!/bin/usr/python 
import os
import sys
import numpy as np 
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat


def constructGamaLegacy(field, data_release=3): 
    ''' Construct the GAMA-Legacy catalogs using
    methods in the GamaLegacy class object. 
    '''
    gleg = Cat.GamaLegacy()
    gleg._Build(field, dr_gama=data_release, 
            sweep_dir='/global/project/projectdirs/cosmo/data/legacysurvey/dr5/sweep/5.0/', 
            silent=False)
    return None 


if __name__=='__main__':
    field = sys.argv[1]
    dr = sys.argv[2]
    constructGamaLegacy(field, data_release=dr)
