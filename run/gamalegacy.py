#!/bin/usr/python 
import os
import sys
import numpy as np 
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat


def constructGamaLegacy(field, gama_dr=3, legacy_dr=7): 
    ''' Construct the GAMA-Legacy catalogs using
    methods in the GamaLegacy class object. 
    '''
    gleg = Cat.GamaLegacy()
    gleg._Build(field, dr_gama=gama_dr, dr_legacy=legcay_dr, silent=False)
    return None 


if __name__=='__main__':
    field = sys.argv[1]
    gama_dr = int(sys.argv[2])
    legacy_dr = int(sys.argv[3])
    constructGamaLegacy(field, gama_dr=gama_dr, legacy_dr=legacy_dr)
