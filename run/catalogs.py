#!/bin/python
'''
'''
from feasibgs import catalogs as Cat

def legacy_sweeps():
    ''' construct combined sweeps catalog from legacy 
    '''
    Cat._collect_Legacy_sweeps(dr=8)
    return None


if __name__=="__main__":
    legacy_sweeps()

