#!/bin/python 
'''
script to build catalogs
'''
from feasibgs import catalogs as Cat 


def build_GAMA(): 
    ''' build GAMA datasets
    '''
    gama = Cat.GAMA()
    gama._Build(data_release=3, silent=False) 
    gama._fieldSplit(data_release=3, silent=False) 
    return None


def build_GamaLegacy():
    galega = GamaLegacy()
    for field in ['g09', 'g12', 'g15']: 
        galega._Build(field, dr_gama=3, dr_legacy=7, sweep_dir=None, silent=False)
    return None 

if __name__=='__main__': 
    #build_GAMA()
    build_GamaLegacy()
