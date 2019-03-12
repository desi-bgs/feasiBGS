#!/bin/usr/python 
import os
import sys
import numpy as np 
# -- feasibgs -- 
from feasibgs import util as UT
from feasibgs import catalogs as Cat


def constructGAMA(gama_dr=3): 
    gama = Cat.GAMA() 
    gama._Build(data_release=gama_dr, silent=False) # build the catalog
    gama._fieldSplit(data_release=gama_dr, silent=False) # split into the different GAMA fields
    return None 


def constructGamaLegacy(field, gama_dr=3, legacy_dr=7): 
    ''' Construct the GAMA-Legacy catalogs using
    methods in the GamaLegacy class object. 
    '''
    gleg = Cat.GamaLegacy()
    gleg._Build(field, dr_gama=gama_dr, dr_legacy=legacy_dr, silent=False)
    return None 


if __name__=='__main__':
    catalog = sys.argv[1] 
    if catalog == 'gama': 
        constructGAMA(gama_dr=3)
    elif catalog == 'gleg': 
        field = sys.argv[2]
        constructGamaLegacy(field, gama_dr=3, legacy_dr=7)
